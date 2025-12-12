"""
Data Manager: Upstox API + Redis Memory
V2 COMPLETE FIX: Volume delta tracking, candle fallback, live VWAP, graceful degradation
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
            
            # Find MONTHLY futures
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
            
            all_futures.sort(key=lambda x: x['expiry'])
            
            monthly_futures = None
            for fut in all_futures:
                if fut['days_to_expiry'] > 10:
                    monthly_futures = fut
                    break
            
            if not monthly_futures:
                monthly_futures = all_futures[0]
                logger.warning(f"⚠️ Using nearest futures")
            
            self.futures_key = monthly_futures['key']
            self.futures_expiry = monthly_futures['expiry']
            self.futures_symbol = monthly_futures['symbol']
            
            logger.info(f"✅ Futures (MONTHLY): {monthly_futures['symbol']}")
            logger.info(f"   Expiry: {monthly_futures['expiry'].strftime('%Y-%m-%d')} ({monthly_futures['days_to_expiry']} days)")
            
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
        
        if instrument_key in quotes:
            return quotes[instrument_key]
        
        alt_key = instrument_key.replace('|', ':')
        if alt_key in quotes:
            return quotes[alt_key]
        
        segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
        for key in quotes.keys():
            if key.startswith(segment):
                return quotes[key]
        
        logger.error(f"❌ Instrument not found")
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
        """Get option chain (WEEKLY options)"""
        if not instrument_key:
            return None
        
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        
        data = await self._request(url)
        
        if not data:
            return None
        
        if 'data' not in data:
            return None
        
        return data['data']


# ==================== Redis Brain (same as before) ====================
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
        logger.info("📚 Skipping previous day data")
        self.premarket_loaded = True


# ==================== Data Fetcher V2 ====================
class DataFetcher:
    """
    🔥 V2 COMPLETE FIX:
    - Volume delta tracking
    - Candle freeze detection with fallback
    - Live VWAP calculation
    - Graceful degradation
    """
    
    def __init__(self, client):
        self.client = client
        
        # Candle tracking
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
        self.candle_frozen = False
        self.candle_freeze_start = None
        
        # Volume tracking (for delta calculation)
        self.previous_cumulative_volume = 0
        self.previous_volume_time = None
        self.volume_history = []  # Store last 10 deltas
        
        # Live VWAP tracking
        self.live_vwap = None
        self.vwap_cumulative_vol_price = 0
        self.vwap_cumulative_volume = 0
        
        # Live price history (for synthetic ATR)
        self.live_price_history = []
    
    async def fetch_spot(self):
        """Fetch spot price"""
        try:
            if not self.client.spot_key:
                return None
            
            data = await self.client.get_quote(self.client.spot_key)
            
            if not data:
                return None
            
            ltp = data.get('last_price')
            if not ltp:
                return None
            
            return float(ltp)
            
        except Exception as e:
            logger.error(f"❌ Spot error: {e}")
            return None
    
    async def fetch_futures_candles(self):
        """
        🔧 FIXED: Fetch candles with freeze detection
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
            
            # Parse candles
            if isinstance(candles[0], dict):
                df = pd.DataFrame(candles)
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    return None
                
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
            else:
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 🆕 FREEZE DETECTION
            latest_timestamp = df['timestamp'].iloc[-1]
            current_time = datetime.now(IST)
            
            if self.last_candle_timestamp == latest_timestamp:
                self.candle_repeat_count += 1
                
                if not self.candle_frozen and self.candle_repeat_count >= 5:
                    self.candle_frozen = True
                    self.candle_freeze_start = current_time
                    logger.error(f"🚨 CANDLE API FROZEN! Timestamp stuck at {latest_timestamp}")
                    logger.error(f"   Switching to LIVE DATA MODE...")
                elif self.candle_frozen:
                    freeze_duration = (current_time - self.candle_freeze_start).total_seconds() / 60
                    logger.warning(f"⚠️ CANDLE FROZEN for {freeze_duration:.1f} min (repeat #{self.candle_repeat_count})")
            else:
                if self.candle_repeat_count > 0:
                    logger.info(f"✅ Candle updated after {self.candle_repeat_count} repeats")
                if self.candle_frozen:
                    logger.info(f"✅ CANDLE API RECOVERED!")
                    
                self.candle_repeat_count = 0
                self.candle_frozen = False
                self.candle_freeze_start = None
                self.last_candle_timestamp = latest_timestamp
            
            # Log candle status
            if not self.candle_frozen:
                logger.info(f"📊 CANDLE STATUS: Active (Latest: {latest_timestamp.strftime('%H:%M')})")
            else:
                logger.warning(f"⚠️ CANDLE STATUS: FROZEN MODE - Using live data only")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Futures candles error: {e}")
            return None
    
    async def fetch_futures_ltp(self):
        """Fetch LIVE futures price"""
        try:
            if not self.client.futures_key:
                return None
            
            data = await self.client.get_quote(self.client.futures_key)
            
            if not data:
                return None
            
            ltp = data.get('last_price')
            if not ltp:
                return None
            
            price = float(ltp)
            
            # 🆕 V2: Track price history for synthetic ATR
            if price > 0:
                self.live_price_history.append({
                    'price': price,
                    'timestamp': datetime.now(IST)
                })
                
                # Keep last 50 prices only
                if len(self.live_price_history) > 50:
                    self.live_price_history = self.live_price_history[-50:]
            
            return price
            
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None
    
    async def fetch_daily_candles(self, num_candles=20):
        """
        🆕 V3: Fetch 7 days historical data for daily S/R
        Uses HISTORICAL API endpoint directly!
        """
        try:
            if not self.client.futures_symbol:
                logger.error("❌ Futures symbol missing!")
                return None
            
            # Calculate dates (last 10 days to get enough 30-min candles)
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=10)
            
            # Format dates for API
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')
            
            # Historical endpoint
            url = f"https://api.upstox.com/v2/historical-candle/{self.client.futures_symbol}/30minute/{to_date_str}/{from_date_str}"
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
            }
            
            async with self.client.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'success' and 'data' in data:
                        candles = data['data'].get('candles', [])
                        
                        if candles:
                            # Historical API returns: [timestamp, open, high, low, close, volume, oi]
                            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df['open'] = pd.to_numeric(df['open'], errors='coerce')
                            df['high'] = pd.to_numeric(df['high'], errors='coerce')
                            df['low'] = pd.to_numeric(df['low'], errors='coerce')
                            df['close'] = pd.to_numeric(df['close'], errors='coerce')
                            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                            
                            # Sort and sample for daily-like levels
                            df = df.sort_values('timestamp')
                            if len(df) > 100:
                                step = max(1, len(df) // 30)
                                df = df.iloc[::step].head(30)
                            
                            df = df.reset_index(drop=True)
                            
                            logger.info(f"✅ Daily S/R: {len(df)} bars from {from_date_str} to {to_date_str}")
                            return df
                
                logger.warning(f"⚠️ Historical API returned {response.status}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Historical daily error: {e}")
            return None
    
    async def fetch_15min_candles(self, num_candles=90):
        """
        🆕 V3: Fetch 7 days 1-min data and resample to 15-min
        Uses HISTORICAL API for multi-day data!
        """
        try:
            if not self.client.futures_symbol:
                logger.error("❌ Futures symbol missing!")
                return None
            
            # Get last 7 days of 1-min candles
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=7)
            
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')
            
            # Historical 1-minute endpoint
            url = f"https://api.upstox.com/v2/historical-candle/{self.client.futures_symbol}/1minute/{to_date_str}/{from_date_str}"
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
            }
            
            async with self.client.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'success' and 'data' in data:
                        candles = data['data'].get('candles', [])
                        
                        if candles and len(candles) >= 30:
                            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df['open'] = pd.to_numeric(df['open'], errors='coerce')
                            df['high'] = pd.to_numeric(df['high'], errors='coerce')
                            df['low'] = pd.to_numeric(df['low'], errors='coerce')
                            df['close'] = pd.to_numeric(df['close'], errors='coerce')
                            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                            
                            # Resample to 15-min
                            df = df.set_index('timestamp')
                            df_15min = df.resample('15T').agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            }).dropna()
                            
                            df_15min = df_15min.reset_index()
                            df_15min = df_15min.tail(num_candles)
                            
                            logger.info(f"✅ 15-min S/R: {len(df_15min)} bars (7 days resampled)")
                            return df_15min
                
                logger.warning(f"⚠️ Historical 15-min returned {response.status}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Historical 15-min error: {e}")
            return None
            
            # Track for synthetic ATR calculation
            self.live_price_history.append({
                'price': price,
                'time': datetime.now(IST)
            })
            
            # Keep only last 20 prices
            self.live_price_history = self.live_price_history[-20:]
            
            return price
            
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None
    
    async def fetch_futures_live_volume(self):
        """
        🆕 FETCH LIVE VOLUME with DELTA calculation
        Returns: (cumulative_volume, delta_volume, avg_delta)
        """
        try:
            if not self.client.futures_key:
                return None, None, None
            
            data = await self.client.get_quote(self.client.futures_key)
            
            if not data:
                return None, None, None
            
            cumulative_volume = data.get('volume', 0)
            
            if cumulative_volume == 0:
                logger.warning("⚠️ Live volume = 0")
                return None, None, None
            
            current_time = datetime.now(IST)
            cumulative_volume = float(cumulative_volume)
            
            # 🔥 CALCULATE DELTA (1-minute volume)
            delta_volume = None
            if self.previous_cumulative_volume > 0:
                delta_volume = cumulative_volume - self.previous_cumulative_volume
                
                # Store delta in history
                self.volume_history.append({
                    'delta': delta_volume,
                    'time': current_time
                })
                
                # Keep only last 10 deltas
                self.volume_history = self.volume_history[-10:]
            
            # Calculate average delta
            avg_delta = None
            if len(self.volume_history) >= 3:
                deltas = [v['delta'] for v in self.volume_history]
                avg_delta = sum(deltas) / len(deltas)
            
            # Update tracking
            self.previous_cumulative_volume = cumulative_volume
            self.previous_volume_time = current_time
            
            logger.info(f"📊 VOLUME (DELTA MODE):")
            logger.info(f"   Cumulative: {cumulative_volume:,.0f}")
            if delta_volume is not None:
                logger.info(f"   Delta (1-min): {delta_volume:,.0f}")
            if avg_delta is not None:
                logger.info(f"   Avg delta: {avg_delta:,.0f}")
            
            return cumulative_volume, delta_volume, avg_delta
            
        except Exception as e:
            logger.error(f"❌ Live volume error: {e}")
            return None, None, None
    
    def update_live_vwap(self, price, volume):
        """
        🆕 UPDATE VWAP using live data (when candles frozen)
        """
        if volume is None or volume <= 0:
            return self.live_vwap
        
        try:
            # Incremental VWAP calculation
            self.vwap_cumulative_vol_price += (price * volume)
            self.vwap_cumulative_volume += volume
            
            if self.vwap_cumulative_volume > 0:
                self.live_vwap = self.vwap_cumulative_vol_price / self.vwap_cumulative_volume
            
            return self.live_vwap
        except Exception as e:
            logger.error(f"❌ Live VWAP error: {e}")
            return self.live_vwap
    
    def calculate_synthetic_atr(self, periods=14):
        """
        🆕 CALCULATE ATR from live price history (when candles frozen)
        """
        if len(self.live_price_history) < periods:
            return ATR_FALLBACK
        
        try:
            recent_prices = [p['price'] for p in self.live_price_history[-periods:]]
            
            # Calculate True Range from price movements
            ranges = []
            for i in range(1, len(recent_prices)):
                price_range = abs(recent_prices[i] - recent_prices[i-1])
                ranges.append(price_range)
            
            if not ranges:
                return ATR_FALLBACK
            
            atr = sum(ranges) / len(ranges)
            
            logger.info(f"📊 SYNTHETIC ATR: {atr:.1f} (from {len(ranges)} price moves)")
            
            return round(atr, 2)
        
        except Exception as e:
            logger.error(f"❌ Synthetic ATR error: {e}")
            return ATR_FALLBACK
    
    def is_candle_frozen(self):
        """Check if candle API is frozen"""
        return self.candle_frozen
    
    async def fetch_option_chain(self, reference_price):
        """Fetch option chain"""
        try:
            if not self.client.index_key:
                return None
            
            expiry = get_next_weekly_expiry()
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)
            
            data = await self.client.get_option_chain(self.client.index_key, expiry)
            
            if not data:
                return None
            
            strike_data = {}
            
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
                return None
            
            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                return None
            
            logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI: {total_oi:,.0f})")
            
            return atm, strike_data
        
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None
