"""
Data Manager: Upstox API + Redis Memory
FIXED: LIVE volume fetching, timestamp debugging, proper candle handling
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

MEMORY_TTL_SECONDS = MEMORY_TTL_HOURS * 3600


# ==================== Upstox Client ====================
class UpstoxClient:
    """Upstox API V2 Client with MONTHLY futures detection"""
    
    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0
        
        # Instrument keys
        self.spot_key = None
        self.index_key = None
        self.futures_key = None
        self.futures_expiry = None
        self.futures_symbol = None
    
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
        """Auto-detect NIFTY instruments (spot + MONTHLY futures)"""
        logger.info("🔍 Auto-detecting NIFTY instruments...")
        
        try:
            url = UPSTOX_INSTRUMENTS_URL
            
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
            
            # Find MONTHLY futures (SMART DETECTION)
            now = datetime.now(IST)
            all_futures = []
            
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
                        days_to_expiry = (expiry_dt - now).days
                        all_futures.append({
                            'key': instrument.get('instrument_key'),
                            'expiry': expiry_dt,
                            'symbol': instrument.get('trading_symbol', ''),
                            'days_to_expiry': days_to_expiry,
                            'weekday': expiry_dt.strftime('%A')
                        })
                except:
                    continue
            
            if not all_futures:
                logger.error("❌ No futures contracts found")
                return False
            
            # Sort by expiry date
            all_futures.sort(key=lambda x: x['expiry'])
            
            # Select MONTHLY contract (>10 days)
            monthly_futures = None
            
            for fut in all_futures:
                if fut['days_to_expiry'] > 10:
                    monthly_futures = fut
                    break
            
            # Fallback: nearest
            if not monthly_futures:
                monthly_futures = all_futures[0]
                logger.warning(f"⚠️ Using nearest futures (no contract > 10 days found)")
            
            self.futures_key = monthly_futures['key']
            self.futures_expiry = monthly_futures['expiry']
            self.futures_symbol = monthly_futures['symbol']
            
            logger.info(f"✅ Futures (MONTHLY): {monthly_futures['symbol']}")
            logger.info(f"   Expiry: {monthly_futures['expiry'].strftime('%Y-%m-%d %A')} ({monthly_futures['days_to_expiry']} days)")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return False
    
    async def get_quote(self, instrument_key):
        """Get market quote (for spot/futures LIVE price)"""
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
        """Get historical candles (for technical analysis only)"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_HISTORICAL_URL}/intraday/{encoded}/{interval}"
        
        data = await self._request(url)
        
        if not data or 'data' not in data:
            return None
        
        return data['data']
    
    async def get_option_chain(self, instrument_key, expiry_date):
        """Get option chain (WEEKLY options)"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        
        data = await self._request(url)
        
        if not data:
            logger.error("❌ Option chain API returned None")
            return None
        
        if 'data' not in data:
            logger.error(f"❌ No 'data' key. Keys: {list(data.keys())}")
            return None
        
        return data['data']


# ==================== Redis Brain ====================
class RedisBrain:
    """Memory manager with 24 hour TTL"""
    
    def __init__(self):
        self.client = None
        self.memory = {}
        self.memory_timestamps = {}
        self.snapshot_count = 0
        self.first_snapshot_time = None
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
        """Save total OI snapshot"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:total:{now.strftime('%Y%m%d_%H%M')}"
        value = json.dumps({'ce': ce, 'pe': pe, 'timestamp': now.isoformat()})
        
        if self.snapshot_count == 0:
            self.first_snapshot_time = now
            logger.info(f"📍 FIRST SNAPSHOT at {now.strftime('%H:%M')} - BASE REFERENCE")
        
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
        
        if self.snapshot_count == 1:
            logger.info(f"💾 First snapshot saved: CE={ce:,.0f}, PE={pe:,.0f}")
        
        self._cleanup()
    
    def get_total_oi_change(self, current_ce, current_pe, minutes_ago=15):
        """Get OI change with tolerance"""
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
            for offset in [-1, 1, -2, 2]:
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
            
            if past_ce == 0:
                ce_chg = 100.0 if current_ce > 0 else 0.0
            else:
                ce_chg = ((current_ce - past_ce) / past_ce * 100)
            
            if past_pe == 0:
                pe_chg = 100.0 if current_pe > 0 else 0.0
            else:
                pe_chg = ((current_pe - past_pe) / past_pe * 100)
            
            return round(ce_chg, 1), round(pe_chg, 1), True
        
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
            return 0.0, 0.0, False
    
    def save_strike(self, strike, data):
        """Save strike OI"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:strike:{strike}:{now.strftime('%Y%m%d_%H%M')}"
        
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
        """Get strike OI change"""
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
        
        if not past_str:
            for offset in [-1, 1, -2, 2]:
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
            
            if ce_past == 0:
                ce_chg = 100.0 if ce_curr > 0 else 0.0
            else:
                ce_chg = ((ce_curr - ce_past) / ce_past * 100)
            
            if pe_past == 0:
                pe_chg = 100.0 if pe_curr > 0 else 0.0
            else:
                pe_chg = ((pe_curr - pe_past) / pe_past * 100)
            
            return round(ce_chg, 1), round(pe_chg, 1), True
        
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
            return 0.0, 0.0, False
    
    def is_warmed_up(self, minutes=15):
        """Check warmup from first snapshot"""
        if not self.first_snapshot_time:
            return False
        
        elapsed = (datetime.now(IST) - self.first_snapshot_time).total_seconds() / 60
        
        if elapsed < minutes:
            return False
        
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
        """Get memory stats"""
        if not self.first_snapshot_time:
            elapsed = 0
        else:
            elapsed = (datetime.now(IST) - self.first_snapshot_time).total_seconds() / 60
        
        return {
            'snapshot_count': self.snapshot_count,
            'elapsed_minutes': elapsed,
            'first_snapshot_time': self.first_snapshot_time,
            'warmed_up_5m': self.is_warmed_up(5),
            'warmed_up_10m': self.is_warmed_up(10),
            'warmed_up_15m': self.is_warmed_up(15)
        }
    
    def _cleanup(self):
        """Clean expired RAM entries"""
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
        """Skip previous day data"""
        if self.premarket_loaded:
            return
        logger.info("📚 Skipping previous day data (fresh start at 9:16)")
        self.premarket_loaded = True


# ==================== Data Fetcher ====================
class DataFetcher:
    """High-level data fetching with LIVE volume support"""
    
    def __init__(self, client):
        self.client = client
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
    
    async def fetch_spot(self):
        """Fetch spot price (for ATM calculation)"""
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
    
    async def fetch_futures_candles(self):
        """
        🔧 FIXED: Fetch MONTHLY futures candles + detect timestamp freezing
        """
        try:
            if not self.client.futures_key:
                return None
            
            data = await self.client.get_candles(self.client.futures_key, '1minute')
            
            if not data or 'candles' not in data:
                return None
            
            candles = data['candles']
            if not candles:
                return None
            
            # Handle both list and dict formats
            if isinstance(candles[0], dict):
                df = pd.DataFrame(candles)
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"❌ Missing columns. Available: {df.columns.tolist()}")
                    return None
                
                # Convert types explicitly
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
            else:
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 🆕 TIMESTAMP FREEZE DETECTION
            latest_timestamp = df['timestamp'].iloc[-1]
            
            if self.last_candle_timestamp == latest_timestamp:
                self.candle_repeat_count += 1
                logger.warning(f"⚠️ CANDLE FROZEN! Same timestamp repeated {self.candle_repeat_count} times: {latest_timestamp}")
            else:
                if self.candle_repeat_count > 0:
                    logger.info(f"✅ Candle updated after {self.candle_repeat_count} repeats")
                self.candle_repeat_count = 0
                self.last_candle_timestamp = latest_timestamp
            
            # 🆕 DETAILED CANDLE DEBUG
            logger.info(f"📊 CANDLE DEBUG:")
            logger.info(f"   Total bars: {len(df)}")
            logger.info(f"   Latest candle time: {latest_timestamp}")
            logger.info(f"   Latest close: ₹{df['close'].iloc[-1]:.2f}")
            logger.info(f"   Latest volume: {df['volume'].iloc[-1]:,.0f}")
            logger.info(f"   Last 3 timestamps: {df['timestamp'].tail(3).dt.strftime('%H:%M:%S').tolist()}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Futures candles error: {e}", exc_info=True)
            return None
    
    async def fetch_futures_ltp(self):
        """Fetch MONTHLY futures LIVE price (for entry/exit decisions)"""
        try:
            if not self.client.futures_key:
                logger.error("❌ Futures key missing")
                return None
            
            data = await self.client.get_quote(self.client.futures_key)
            
            if not data:
                logger.error("❌ Futures quote returned None")
                return None
            
            ltp = data.get('last_price')
            if not ltp:
                logger.error(f"❌ No 'last_price' in futures quote. Keys: {list(data.keys())}")
                return None
            
            return float(ltp)
            
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None
    
    async def fetch_futures_live_volume(self):
        """
        🆕 FETCH LIVE VOLUME from quote API (not candles!)
        This gets TODAY'S cumulative volume, NOT per-candle
        """
        try:
            if not self.client.futures_key:
                return None
            
            data = await self.client.get_quote(self.client.futures_key)
            
            if not data:
                return None
            
            # Get volume from quote
            volume = data.get('volume', 0)
            
            if volume == 0:
                logger.warning("⚠️ Live volume = 0 from quote API")
                return None
            
            logger.info(f"📊 LIVE VOLUME from quote: {volume:,.0f}")
            
            return float(volume)
            
        except Exception as e:
            logger.error(f"❌ Live volume error: {e}")
            return None
    
    async def fetch_option_chain(self, reference_price):
        """
        Fetch WEEKLY option chain - 11 strikes
        reference_price can be spot OR futures (caller decides)
        """
        try:
            if not self.client.index_key:
                return None
            
            expiry = get_next_weekly_expiry()
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)
            
            logger.info(f"📡 Fetching option chain: Expiry={expiry}, ATM={atm}, Range={min_strike}-{max_strike}")
            
            data = await self.client.get_option_chain(self.client.index_key, expiry)
            
            if not data:
                return None
            
            # DEBUG: Log response structure
            logger.info(f"🔍 DEBUG: Response type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"🔍 DEBUG: Top-level keys: {list(data.keys())[:5]}")
            elif isinstance(data, list):
                logger.info(f"🔍 DEBUG: List length: {len(data)}")
                if len(data) > 0:
                    logger.info(f"🔍 DEBUG: First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    logger.info(f"🔍 DEBUG: First item sample: {str(data[0])[:200]}")
            
            strike_data = {}
            
            # Parse response (handle both list and dict formats)
            if isinstance(data, list):
                for item in data:
                    strike = item.get('strike_price') or item.get('strike')
                    if not strike:
                        continue
                    
                    strike = float(strike)
                    if strike < min_strike or strike > max_strike:
                        continue
                    
                    ce_data = item.get('call_options', {}) or item.get('CE', {})
                    pe_data = item.get('put_options', {}) or item.get('PE', {})
                    
                    # DEBUG: Log first strike structure
                    if len(strike_data) == 0:
                        logger.info(f"🔍 DEBUG: CE data keys: {list(ce_data.keys()) if ce_data else 'Empty'}")
                        logger.info(f"🔍 DEBUG: PE data keys: {list(pe_data.keys()) if pe_data else 'Empty'}")
                        logger.info(f"🔍 DEBUG: CE sample: {str(ce_data)[:200]}")
                    
                    # Extract from nested market_data object
                    ce_market = ce_data.get('market_data', {})
                    pe_market = pe_data.get('market_data', {})
                    
                    strike_data[strike] = {
                        'ce_oi': float(ce_market.get('oi') or 0),
                        'pe_oi': float(pe_market.get('oi') or 0),
                        'ce_vol': float(ce_market.get('volume') or 0),
                        'pe_vol': float(pe_market.get('volume') or 0),
                        'ce_ltp': float(ce_market.get('ltp') or 0),
                        'pe_ltp': float(pe_market.get('ltp') or 0)
                    }
            
            elif isinstance(data, dict):
                for key, item in data.items():
                    strike = item.get('strike_price') or item.get('strike')
                    if not strike:
                        continue
                    
                    strike = float(strike)
                    if strike < min_strike or strike > max_strike:
                        continue
                    
                    ce_data = item.get('call_options', {}) or item.get('CE', {})
                    pe_data = item.get('put_options', {}) or item.get('PE', {})
                    
                    # DEBUG: Log first strike structure
                    if len(strike_data) == 0:
                        logger.info(f"🔍 DEBUG: CE data keys: {list(ce_data.keys()) if ce_data else 'Empty'}")
                        logger.info(f"🔍 DEBUG: PE data keys: {list(pe_data.keys()) if pe_data else 'Empty'}")
                    
                    # Extract from nested market_data object
                    ce_market = ce_data.get('market_data', {})
                    pe_market = pe_data.get('market_data', {})
                    
                    strike_data[strike] = {
                        'ce_oi': float(ce_market.get('oi') or 0),
                        'pe_oi': float(pe_market.get('oi') or 0),
                        'ce_vol': float(ce_market.get('volume') or 0),
                        'pe_vol': float(pe_market.get('volume') or 0),
                        'ce_ltp': float(ce_market.get('ltp') or 0),
                        'pe_ltp': float(pe_market.get('ltp') or 0)
                    }
            
            if not strike_data:
                logger.error("❌ No strikes parsed!")
                return None
            
            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                logger.error("❌ ALL OI VALUES ARE ZERO!")
                logger.error(f"🔍 DEBUG: Strike data sample: {list(strike_data.items())[:2]}")
                return None
            
            logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI: {total_oi:,.0f})")
            
            return atm, strike_data
        
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}", exc_info=True)
            return None
