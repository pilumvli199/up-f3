"""
Data Manager: Upstox API + Redis Memory
FIXED: OI parsing, 24hr expiry, better error handling
"""

import asyncio
import aiohttp
import json
import time as time_module
from datetime import datetime, timedelta
from urllib.parse import quote
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config import *
from utils import IST, setup_logger

logger = setup_logger("data_manager")

# ==================== Memory TTL (24 Hours) ====================
MEMORY_TTL_HOURS = 24
MEMORY_TTL_SECONDS = MEMORY_TTL_HOURS * 3600  # 86400 seconds


# ==================== Upstox Client ====================
class UpstoxClient:
    """Upstox API V2 Client with dynamic instrument detection"""
    
    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0
        
        # Instrument keys
        self.spot_key = None
        self.index_key = None
        self.futures_key = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.detect_instruments()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _get_headers(self):
        return {
            'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
            'Accept': 'application/json'
        }
    
    async def _rate_limit(self):
        elapsed = asyncio.get_event_loop().time() - self._last_request
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request = asyncio.get_event_loop().time()
    
    async def _request(self, url, params=None):
        """Make API request with retry"""
        await self._rate_limit()
        
        for attempt in range(3):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.get(url, headers=self._get_headers(), 
                                           params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        logger.warning(f"⚠️ Rate limit, retry {attempt+1}/3")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        text = await resp.text()
                        logger.error(f"❌ API error {resp.status}: {text[:300]}")
                        return None
            
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout (attempt {attempt + 1}/3)")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return None
            
            except Exception as e:
                logger.error(f"❌ Request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return None
        
        return None
    
    async def detect_instruments(self):
        """Auto-detect NIFTY instrument keys"""
        logger.info("🔍 Auto-detecting NIFTY instruments...")
        
        try:
            url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
            
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.error(f"❌ Instruments fetch failed: {resp.status}")
                    return False
                
                import gzip
                content = await resp.read()
                json_text = gzip.decompress(content).decode('utf-8')
                instruments = json.loads(json_text)
            
            # Find NIFTY spot
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_INDEX':
                    continue
                
                name = instrument.get('name', '').upper()
                symbol = instrument.get('trading_symbol', '').upper()
                
                if 'NIFTY 50' in name or 'NIFTY 50' in symbol or symbol == 'NIFTY':
                    self.spot_key = instrument.get('instrument_key')
                    self.index_key = self.spot_key
                    logger.info(f"✅ Spot: {self.spot_key}")
                    break
            
            if not self.spot_key:
                logger.error("❌ NIFTY spot not found")
                return False
            
            # Find nearest futures
            now = datetime.now(IST)
            futures_list = []
            
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') != 'FUT':
                    continue
                if instrument.get('name') != 'NIFTY':
                    continue
                
                expiry_ms = instrument.get('expiry', 0)
                if not expiry_ms:
                    continue
                
                try:
                    expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                    if expiry_dt > now:
                        futures_list.append({
                            'key': instrument.get('instrument_key'),
                            'expiry': expiry_dt,
                            'symbol': instrument.get('trading_symbol', '')
                        })
                except:
                    continue
            
            if not futures_list:
                logger.error("❌ No futures found")
                return False
            
            futures_list.sort(key=lambda x: x['expiry'])
            nearest = futures_list[0]
            
            self.futures_key = nearest['key']
            logger.info(f"✅ Futures: {nearest['symbol']} (Expiry: {nearest['expiry'].strftime('%Y-%m-%d')})")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return False
    
    async def get_quote(self, instrument_key):
        """Get market quote"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_QUOTE_URL}?symbol={encoded}"
        
        data = await self._request(url)
        
        if not data or 'data' not in data:
            return None
        
        quotes = data['data']
        
        # Try exact match
        if instrument_key in quotes:
            return quotes[instrument_key]
        
        # Try colon format
        alt_key = instrument_key.replace('|', ':')
        if alt_key in quotes:
            return quotes[alt_key]
        
        # Try segment match
        segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
        for key in quotes.keys():
            if key.startswith(segment):
                return quotes[key]
        
        logger.error(f"❌ Instrument not found in: {list(quotes.keys())[:3]}")
        return None
    
    async def get_candles(self, instrument_key, interval='1minute'):
        """Get historical candles"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_HISTORICAL_URL}/intraday/{encoded}/{interval}"
        
        data = await self._request(url)
        
        if not data or 'data' not in data:
            return None
        
        return data['data']
    
    async def get_option_chain(self, instrument_key, expiry_date):
        """Get option chain - FIXED PARSING"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        
        logger.info(f"📡 Fetching option chain...")
        data = await self._request(url)
        
        if not data:
            logger.error("❌ Option chain API returned None")
            return None
        
        if 'data' not in data:
            logger.error(f"❌ No 'data' key. Keys: {list(data.keys())}")
            logger.error(f"Response sample: {json.dumps(data, indent=2)[:500]}")
            return None
        
        # ✅ LOG RAW RESPONSE FOR DEBUGGING
        chain_data = data['data']
        logger.info(f"📋 RAW API Response Type: {type(chain_data)}")
        if isinstance(chain_data, list) and len(chain_data) > 0:
            logger.info(f"📋 Sample item: {json.dumps(chain_data[0], indent=2)[:800]}")
        elif isinstance(chain_data, dict):
            sample_key = list(chain_data.keys())[0] if chain_data else None
            if sample_key:
                logger.info(f"📋 Sample key: {sample_key}")
                logger.info(f"📋 Sample value: {json.dumps(chain_data[sample_key], indent=2)[:800]}")
        
        return chain_data


# ==================== Redis Brain (24hr expiry) ====================
class RedisBrain:
    """Memory manager with 24 hour TTL"""
    
    def __init__(self):
        self.client = None
        self.memory = {}
        self.memory_timestamps = {}
        self.snapshot_count = 0
        self.startup_time = datetime.now(IST)
        self.premarket_loaded = False
        
        if REDIS_AVAILABLE and REDIS_URL:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info(f"✅ Redis connected (TTL: {MEMORY_TTL_HOURS}h)")
            except Exception as e:
                logger.warning(f"⚠️ Redis failed: {e}. Using RAM.")
                self.client = None
        else:
            logger.info(f"💾 RAM mode (TTL: {MEMORY_TTL_HOURS}h)")
    
    def save_total_oi(self, ce, pe):
        """Save total OI snapshot with 24hr expiry"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:total:{now.strftime('%Y%m%d_%H%M')}"
        value = json.dumps({'ce': ce, 'pe': pe, 'timestamp': now.isoformat()})
        
        if self.client:
            try:
                self.client.setex(key, MEMORY_TTL_SECONDS, value)
            except:
                self.memory[key] = value
                self.memory_timestamps[key] = time_module.time()
        else:
            self.memory[key] = value
            self.memory_timestamps[key] = time_module.time()
        
        self.snapshot_count += 1
        
        # Log first save
        if self.snapshot_count == 1:
            logger.info(f"💾 First snapshot saved: CE={ce:,.0f}, PE={pe:,.0f}")
        
        self._cleanup()
    
    def get_total_oi_change(self, current_ce, current_pe, minutes_ago=15):
        """Get OI change with better validation"""
        target = datetime.now(IST) - timedelta(minutes=minutes_ago)
        target = target.replace(second=0, microsecond=0)
        key = f"nifty:total:{target.strftime('%Y%m%d_%H%M')}"
        
        past_str = None
        if self.client:
            try:
                past_str = self.client.get(key)
            except:
                pass
        
        if not past_str:
            past_str = self.memory.get(key)
        
        # Try tolerance
        if not past_str:
            for offset in [-1, 1, -2, 2, -3, 3]:
                alt = target + timedelta(minutes=offset)
                alt_key = f"nifty:total:{alt.strftime('%Y%m%d_%H%M')}"
                
                if self.client:
                    try:
                        past_str = self.client.get(alt_key)
                        if past_str:
                            break
                    except:
                        pass
                
                if not past_str:
                    past_str = self.memory.get(alt_key)
                    if past_str:
                        break
        
        if not past_str:
            return 0.0, 0.0, False
        
        try:
            past = json.loads(past_str)
            past_ce = past.get('ce', 0)
            past_pe = past.get('pe', 0)
            
            # Handle zero baseline
            if past_ce == 0 and current_ce == 0:
                ce_chg = 0.0
            elif past_ce == 0:
                ce_chg = 100.0  # New OI
            else:
                ce_chg = ((current_ce - past_ce) / past_ce * 100)
            
            if past_pe == 0 and current_pe == 0:
                pe_chg = 0.0
            elif past_pe == 0:
                pe_chg = 100.0
            else:
                pe_chg = ((current_pe - past_pe) / past_pe * 100)
            
            return round(ce_chg, 1), round(pe_chg, 1), True
        
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
            return 0.0, 0.0, False
    
    def save_strike(self, strike, data):
        """Save strike OI with 24hr expiry"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:strike:{strike}:{now.strftime('%Y%m%d_%H%M')}"
        
        # Add timestamp
        data_with_ts = data.copy()
        data_with_ts['timestamp'] = now.isoformat()
        value = json.dumps(data_with_ts)
        
        if self.client:
            try:
                self.client.setex(key, MEMORY_TTL_SECONDS, value)
            except:
                self.memory[key] = value
                self.memory_timestamps[key] = time_module.time()
        else:
            self.memory[key] = value
            self.memory_timestamps[key] = time_module.time()
    
    def get_strike_oi_change(self, strike, current_data, minutes_ago=15):
        """Get strike OI change with validation"""
        target = datetime.now(IST) - timedelta(minutes=minutes_ago)
        target = target.replace(second=0, microsecond=0)
        key = f"nifty:strike:{strike}:{target.strftime('%Y%m%d_%H%M')}"
        
        past_str = None
        if self.client:
            try:
                past_str = self.client.get(key)
            except:
                pass
        
        if not past_str:
            past_str = self.memory.get(key)
        
        # Tolerance
        if not past_str:
            for offset in [-1, 1, -2, 2, -3, 3]:
                alt = target + timedelta(minutes=offset)
                alt_key = f"nifty:strike:{strike}:{alt.strftime('%Y%m%d_%H%M')}"
                
                if self.client:
                    try:
                        past_str = self.client.get(alt_key)
                        if past_str:
                            break
                    except:
                        pass
                
                if not past_str:
                    past_str = self.memory.get(alt_key)
                    if past_str:
                        break
        
        if not past_str:
            return 0.0, 0.0, False
        
        try:
            past = json.loads(past_str)
            
            ce_past = past.get('ce_oi', 0)
            pe_past = past.get('pe_oi', 0)
            ce_curr = current_data.get('ce_oi', 0)
            pe_curr = current_data.get('pe_oi', 0)
            
            # Handle zeros
            if ce_past == 0 and ce_curr == 0:
                ce_chg = 0.0
            elif ce_past == 0:
                ce_chg = 100.0
            else:
                ce_chg = ((ce_curr - ce_past) / ce_past * 100)
            
            if pe_past == 0 and pe_curr == 0:
                pe_chg = 0.0
            elif pe_past == 0:
                pe_chg = 100.0
            else:
                pe_chg = ((pe_curr - pe_past) / pe_past * 100)
            
            return round(ce_chg, 1), round(pe_chg, 1), True
        
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
            return 0.0, 0.0, False
    
    def is_warmed_up(self, minutes=10):
        """Check warmup with data validation"""
        elapsed = (datetime.now(IST) - self.startup_time).total_seconds() / 60
        
        if elapsed < minutes:
            return False
        
        # Verify data exists
        test_time = datetime.now(IST) - timedelta(minutes=minutes)
        test_key = f"nifty:total:{test_time.strftime('%Y%m%d_%H%M')}"
        
        has_data = False
        if self.client:
            try:
                has_data = self.client.exists(test_key) > 0
            except:
                has_data = test_key in self.memory
        else:
            has_data = test_key in self.memory
        
        return has_data
    
    def get_stats(self):
        """Get stats with data check"""
        elapsed = (datetime.now(IST) - self.startup_time).total_seconds() / 60
        return {
            'snapshot_count': self.snapshot_count,
            'elapsed_minutes': elapsed,
            'warmed_up_5m': self.is_warmed_up(5),
            'warmed_up_10m': self.is_warmed_up(10),
            'warmed_up_15m': self.is_warmed_up(15)
        }
    
    def _cleanup(self):
        """Clean expired RAM (24hr)"""
        if not self.memory:
            return
        now = time_module.time()
        expired = [k for k, ts in self.memory_timestamps.items() 
                  if now - ts > MEMORY_TTL_SECONDS]
        for key in expired:
            self.memory.pop(key, None)
            self.memory_timestamps.pop(key, None)
        
        if expired:
            logger.info(f"🧹 Cleaned {len(expired)} expired entries")
    
    async def load_previous_day_data(self):
        """Load previous day data"""
        if self.premarket_loaded:
            return
        logger.info("📚 Loading previous session data...")
        self.premarket_loaded = True


# ==================== Data Fetcher ====================
class DataFetcher:
    """High-level data fetching with fixes"""
    
    def __init__(self, client):
        self.client = client
    
    async def fetch_spot(self):
        """Fetch spot price"""
        try:
            if not self.client.spot_key:
                logger.error("❌ Spot key missing")
                return None
            
            data = await self.client.get_quote(self.client.spot_key)
            
            if not data:
                return None
            
            ltp = data.get('last_price')
            if not ltp:
                logger.error(f"❌ No 'last_price'. Keys: {list(data.keys())}")
                return None
            
            return float(ltp)
            
        except Exception as e:
            logger.error(f"❌ Spot error: {e}")
            return None
    
    async def fetch_futures(self):
        """Fetch futures candles"""
        try:
            if not self.client.futures_key:
                return None
            
            data = await self.client.get_candles(self.client.futures_key, '1minute')
            
            if not data or 'candles' not in data:
                return None
            
            candles = data['candles']
            if not candles:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Futures error: {e}")
            return None
    
    async def fetch_option_chain(self, spot_price):
        """Fetch option chain - COMPLETELY FIXED"""
        try:
            if not self.client.index_key:
                return None
            
            expiry = get_next_tuesday_expiry()
            atm = calculate_atm_strike(spot_price)
            min_strike, max_strike = get_strike_range(atm, num_strikes=3)
            
            data = await self.client.get_option_chain(self.client.index_key, expiry)
            
            if not data:
                return None
            
            strike_data = {}
            
            # ✅ FIXED: Try multiple response formats
            if isinstance(data, list):
                logger.info(f"📊 Parsing list format ({len(data)} items)")
                
                for item in data:
                    strike = item.get('strike_price') or item.get('strike')
                    if not strike:
                        continue
                    
                    strike = float(strike)
                    if strike < min_strike or strike > max_strike:
                        continue
                    
                    # Try format 1: Nested call_options/put_options
                    ce_data = item.get('call_options', {})
                    pe_data = item.get('put_options', {})
                    
                    # Try format 2: Direct CE/PE keys
                    if not ce_data:
                        ce_data = item.get('CE', {})
                    if not pe_data:
                        pe_data = item.get('PE', {})
                    
                    # Extract values with fallbacks
                    ce_oi = (ce_data.get('open_interest') or 
                            ce_data.get('oi') or 
                            ce_data.get('market_data', {}).get('oi') or 0)
                    
                    pe_oi = (pe_data.get('open_interest') or 
                            pe_data.get('oi') or 
                            pe_data.get('market_data', {}).get('oi') or 0)
                    
                    ce_vol = (ce_data.get('volume') or 
                             ce_data.get('market_data', {}).get('volume') or 0)
                    
                    pe_vol = (pe_data.get('volume') or 
                             pe_data.get('market_data', {}).get('volume') or 0)
                    
                    ce_ltp = (ce_data.get('last_price') or 
                             ce_data.get('ltp') or 
                             ce_data.get('market_data', {}).get('ltp') or 0)
                    
                    pe_ltp = (pe_data.get('last_price') or 
                             pe_data.get('ltp') or 
                             pe_data.get('market_data', {}).get('ltp') or 0)
                    
                    strike_data[strike] = {
                        'ce_oi': float(ce_oi),
                        'pe_oi': float(pe_oi),
                        'ce_vol': float(ce_vol),
                        'pe_vol': float(pe_vol),
                        'ce_ltp': float(ce_ltp),
                        'pe_ltp': float(pe_ltp)
                    }
            
            elif isinstance(data, dict):
                logger.info(f"📊 Parsing dict format ({len(data)} keys)")
                
                for key, item in data.items():
                    strike = item.get('strike_price') or item.get('strike')
                    if not strike:
                        continue
                    
                    strike = float(strike)
                    if strike < min_strike or strike > max_strike:
                        continue
                    
                    # Same parsing logic as above
                    ce_data = item.get('call_options', {}) or item.get('CE', {})
                    pe_data = item.get('put_options', {}) or item.get('PE', {})
                    
                    ce_oi = (ce_data.get('open_interest') or ce_data.get('oi') or 0)
                    pe_oi = (pe_data.get('open_interest') or pe_data.get('oi') or 0)
                    ce_vol = (ce_data.get('volume') or 0)
                    pe_vol = (pe_data.get('volume') or 0)
                    ce_ltp = (ce_data.get('last_price') or ce_data.get('ltp') or 0)
                    pe_ltp = (pe_data.get('last_price') or pe_data.get('ltp') or 0)
                    
                    strike_data[strike] = {
                        'ce_oi': float(ce_oi),
                        'pe_oi': float(pe_oi),
                        'ce_vol': float(ce_vol),
                        'pe_vol': float(pe_vol),
                        'ce_ltp': float(ce_ltp),
                        'pe_ltp': float(pe_ltp)
                    }
            
            # ✅ VALIDATION
            if not strike_data:
                logger.error("❌ No strikes parsed!")
                return None
            
            # Check if all OI is zero
            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                logger.error("❌ ALL OI VALUES ARE ZERO - API response issue!")
                logger.error(f"Sample strike data: {list(strike_data.items())[0] if strike_data else 'Empty'}")
                return None
            
            logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI: {total_oi:,.0f})")
            
            # Log sample for debugging
            sample_strike = list(strike_data.keys())[0]
            sample_data = strike_data[sample_strike]
            logger.info(f"📊 Sample {sample_strike}: CE_OI={sample_data['ce_oi']:,.0f}, PE_OI={sample_data['pe_oi']:,.0f}")
            
            return atm, strike_data
        
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}", exc_info=True)
            return None
