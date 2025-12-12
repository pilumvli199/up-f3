"""
Data Manager: Upstox API + Redis Memory (FIXED)
- Robust instrument detection (use canonical instrument_key)
- Option-chain calls use index_key (NSE_INDEX) and expiry YYYY-MM-DD
- fetch_spot() added
- Graceful Redis fallback to RAM
- Live VWAP, delta volume, synthetic ATR
"""

import asyncio
import aiohttp
import json
import time as time_module
from datetime import datetime, timedelta
from urllib.parse import quote
import pandas as pd
import gzip
import io
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

from config import *
from utils import IST, setup_logger

logger = setup_logger("data_manager")

MEMORY_TTL_SECONDS = MEMORY_TTL_HOURS * 3600

# ==================== Upstox Client ====================
class UpstoxClient:
    """Upstox API V2 Client with robust instrument detection (monthly futures)"""

    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0

        # Instrument keys
        self.spot_key = None     # canonical instrument_key for index/spot (NSE_INDEX|...)
        self.index_key = None    # alias for spot_key used for option chain calls
        self.futures_key = None  # canonical instrument_key for futures (NSE_FO|...)
        self.futures_expiry = None
        self.futures_symbol = None

        # cached instruments JSON (decompressed)
        self._instruments = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._fetch_instruments_json()
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
        """Make API request with retry and basic error logging"""
        await self._rate_limit()

        for attempt in range(3):
            try:
                timeout = aiohttp.ClientTimeout(total=12)
                async with self.session.get(url, headers=self._get_headers(), params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(text)
                        except:
                            return None
                    elif resp.status == 429:
                        logger.warning(f"⚠️ Rate limit, retry {attempt+1}/3")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"❌ API error {resp.status}: {text[:400]}")
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

    async def _fetch_instruments_json(self):
        """Download and decompress instruments JSON from Upstox assets if available"""
        try:
            url = UPSTOX_INSTRUMENTS_URL
            logger.info("🔁 Fetching instruments JSON...")
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"⚠️ Instruments fetch returned {resp.status}")
                    return False
                content = await resp.read()
                # content is gz - decompress
                try:
                    decompressed = gzip.decompress(content).decode('utf-8')
                    self._instruments = json.loads(decompressed)
                    logger.info(f"✅ Instruments JSON loaded ({len(self._instruments)} entries)")
                    return True
                except Exception as e:
                    # fallback: try reading as text
                    try:
                        text = content.decode('utf-8')
                        self._instruments = json.loads(text)
                        logger.info(f"✅ Instruments JSON loaded (no gzip)")
                        return True
                    except Exception as ee:
                        logger.error(f"❌ Instruments parse error: {e} / {ee}")
                        self._instruments = None
                        return False
        except Exception as e:
            logger.error(f"❌ Instruments fetch failed: {e}")
            self._instruments = None
            return False

    async def detect_instruments(self):
        """Auto-detect NIFTY/BANKNIFTY instruments (spot + monthly futures) with robust matching"""
        logger.info("🔍 Auto-detecting NIFTY instruments...")

        now = datetime.now(IST)

        instruments = self._instruments or []

        # fallbacks: try API if instruments JSON not loaded
        if not instruments:
            # Try to fetch instruments again synchronously via HTTP (best-effort)
            try:
                url = UPSTOX_INSTRUMENTS_URL
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        try:
                            decompressed = gzip.decompress(content).decode('utf-8')
                            instruments = json.loads(decompressed)
                            self._instruments = instruments
                        except:
                            instruments = json.loads(content.decode('utf-8'))
                            self._instruments = instruments
                    else:
                        logger.error(f"❌ Instruments fetch failed with status: {resp.status}")
            except Exception as e:
                logger.error(f"❌ Instruments fetch exception: {e}")

        # Utility to normalize strings
        def norm(s):
            return (s or "").upper()

        # --- FIND SPOT / INDEX (NSE_INDEX) ---
        preferred_tokens = ['NIFTY', 'NIFTY 50', 'BANKNIFTY', 'BANK NIFTY', 'BANK NIFTY FUT', 'NIFTY100']
        found_spot = False

        for instrument in instruments:
            seg = norm(instrument.get('segment'))
            sym = norm(instrument.get('trading_symbol'))
            name = norm(instrument.get('name'))

            # Prefer NSE_INDEX
            if seg == 'NSE_INDEX':
                # Try direct trading_symbol match for common tokens
                if any(token.replace(' ', '') in sym.replace(' ', '') for token in ['NIFTY', 'BANKNIFTY']):
                    self.spot_key = instrument.get('instrument_key')
                    self.index_key = self.spot_key
                    logger.info(f"✅ Spot/Index detected (by trading_symbol): {sym} -> {self.spot_key}")
                    found_spot = True
                    break
                # or name contains token
                for t in preferred_tokens:
                    if t in name:
                        self.spot_key = instrument.get('instrument_key')
                        self.index_key = self.spot_key
                        logger.info(f"✅ Spot/Index detected (by name): {name} -> {self.spot_key}")
                        found_spot = True
                        break
            if found_spot:
                break

        # fallback attempt: any NSE_INDEX containing 'NIFTY'
        if not found_spot:
            for instrument in instruments:
                if norm(instrument.get('segment')) == 'NSE_INDEX' and 'NIFTY' in norm(instrument.get('name')):
                    self.spot_key = instrument.get('instrument_key')
                    self.index_key = self.spot_key
                    logger.warning(f"⚠️ Fallback spot chosen: {self.spot_key} ({instrument.get('name')})")
                    found_spot = True
                    break

        if not found_spot:
            logger.error("❌ NIFTY spot not found in instruments JSON")
            # don't return False here — allow further attempts but caller should handle missing index_key
        else:
            logger.info(f"   Spot key set to: {self.spot_key}")

        # --- FIND MONTHLY FUTURES (NSE_FO FUT) ---
        all_futures = []
        for instrument in instruments:
            try:
                seg = norm(instrument.get('segment'))
                itype = norm(instrument.get('instrument_type'))
                name = norm(instrument.get('name'))
                sym = norm(instrument.get('trading_symbol'))

                if seg != 'NSE_FO' or itype != 'FUT':
                    continue

                # Accept if it's the desired family (NIFTY/BANKNIFTY)
                if ('NIFTY' not in name and 'NIFTY' not in sym and 'BANKNIFTY' not in name and 'BANKNIFTY' not in sym):
                    continue

                expiry_val = instrument.get('expiry') or instrument.get('expiry_date') or None
                if not expiry_val:
                    continue

                # expiry may be ms epoch or ISO string
                expiry_dt = None
                if isinstance(expiry_val, (int, float)):
                    expiry_dt = datetime.fromtimestamp(int(expiry_val)/1000, tz=IST)
                else:
                    try:
                        expiry_dt = datetime.fromisoformat(expiry_val)
                        # ensure timezone
                        expiry_dt = expiry_dt.astimezone(IST)
                    except:
                        # try parse as int string
                        try:
                            expiry_dt = datetime.fromtimestamp(int(expiry_val)/1000, tz=IST)
                        except:
                            continue

                if expiry_dt <= now:
                    continue

                all_futures.append({
                    'key': instrument.get('instrument_key'),
                    'expiry': expiry_dt,
                    'symbol': instrument.get('trading_symbol'),
                    'name': instrument.get('name'),
                    'days_to_expiry': (expiry_dt - now).days
                })
            except Exception:
                continue

        if not all_futures:
            logger.error("❌ No futures contracts found in instruments JSON")
            return False

        # sort by nearest expiry ascending
        all_futures.sort(key=lambda x: x['expiry'])

        # Prefer monthly (days_to_expiry > 10) else nearest
        monthly_futures = None
        for fut in all_futures:
            if fut['days_to_expiry'] > 10:
                monthly_futures = fut
                break
        if not monthly_futures:
            monthly_futures = all_futures[0]
            logger.warning("⚠️ No distant-monthly future found; using nearest future as monthly")

        self.futures_key = monthly_futures['key']
        self.futures_expiry = monthly_futures['expiry']
        self.futures_symbol = monthly_futures['symbol']
        logger.info(f"✅ Futures (MONTHLY): {monthly_futures['symbol']}")
        logger.info(f"   Expiry: {monthly_futures['expiry'].strftime('%Y-%m-%d')} ({monthly_futures['days_to_expiry']} days)")

        return True

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
        # direct lookup
        if instrument_key in quotes:
            return quotes[instrument_key]

        # try alt form replace | with :
        alt_key = instrument_key.replace('|', ':')
        if alt_key in quotes:
            return quotes[alt_key]

        # best-effort: return any quote that matches the segment prefix
        segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
        for key, val in quotes.items():
            if key.startswith(segment):
                return val

        logger.error("❌ Instrument not found in quote response")
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
        """
        Get option chain using canonical instrument_key (prefer index_key)
        instrument_key must be the canonical instrument_key from instruments JSON
        expiry_date must be YYYY-MM-DD or datetime
        """
        if not instrument_key:
            return None

        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.strftime('%Y-%m-%d')

        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"

        logger.info(f"🔎 Trying option chain for key: {instrument_key} expiry: {expiry_date}")
        data = await self._request(url)
        if not data:
            logger.warning(f"⚠️ Option chain empty for key: {instrument_key}")
            return None
        if 'data' not in data:
            logger.warning(f"⚠️ Option chain response has no 'data' for key: {instrument_key}")
            return None
        return data['data']


# ==================== Redis Brain ====================
class RedisBrain:
    """Memory manager with 24 hour TTL (Redis preferred, RAM fallback)"""

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
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:total:{now.strftime('%Y%m%d_%H%M')}"
        value = json.dumps({'ce': ce, 'pe': pe, 'timestamp': now.isoformat()})

        if self.snapshot_count == 0:
            self.first_snapshot_time = now
            logger.info(f"📍 FIRST SNAPSHOT at {now.strftime('%H:%M')} - BASE REFERENCE")

        if self.client:
            try:
                self.client.setex(key, MEMORY_TTL_SECONDS, value)
            except Exception:
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
        target = datetime.now(IST) - timedelta(minutes=minutes_ago)
        target = target.replace(second=0, microsecond=0)
        key = f"nifty:total:{target.strftime('%Y%m%d_%H%M')}"

        past_str = None
        if self.client:
            try:
                past_str = self.client.get(key)
            except Exception:
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
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:strike:{strike}:{now.strftime('%Y%m%d_%H%M')}"
        data_with_ts = data.copy()
        data_with_ts['timestamp'] = now.isoformat()
        value = json.dumps(data_with_ts)

        if self.client:
            try:
                self.client.setex(key, MEMORY_TTL_SECONDS, value)
            except Exception:
                self.memory[key] = value
                self.memory_timestamps[key] = time_module.time()
        else:
            self.memory[key] = value
            self.memory_timestamps[key] = time_module.time()

    def get_strike_oi_change(self, strike, current_data, minutes_ago=15):
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
        if not self.first_snapshot_time:
            return False
        elapsed = (datetime.now(IST) - self.first_snapshot_time).total_seconds() / 60
        if elapsed < minutes:
            return False
        test_time = datetime.now(IST) - timedelta(minutes=minutes)
        test_time = test_time.replace(second=0, microsecond=0)
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
        if not self.memory:
            return
        now = time_module.time()
        expired = [k for k, ts in self.memory_timestamps.items() if now - ts > MEMORY_TTL_SECONDS]
        for key in expired:
            self.memory.pop(key, None)
            self.memory_timestamps.pop(key, None)
        if expired:
            logger.info(f"🧹 Cleaned {len(expired)} expired entries")

    async def load_previous_day_data(self):
        if self.premarket_loaded:
            return
        logger.info("📚 Skipping previous day data")
        self.premarket_loaded = True


# ==================== Data Fetcher V2 ====================
class DataFetcher:
    """
    V2 FIXED DataFetcher
    - fetch_spot() present
    - fetch_futures_candles() robust parsing
    - fetch_futures_ltp()
    - fetch_futures_live_volume()
    - update_live_vwap(), calculate_synthetic_atr()
    - fetch_option_chain() uses index_key and returns (atm, strike_data)
    """

    def __init__(self, client: UpstoxClient):
        self.client = client

        # Candle tracking
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
        self.candle_frozen = False
        self.candle_freeze_start = None

        # Volume tracking (for delta calculation)
        self.previous_cumulative_volume = 0
        self.previous_volume_time = None
        self.volume_history = []  # list of dicts with 'delta' and 'time'

        # Live VWAP tracking
        self.live_vwap = None
        self.vwap_cumulative_vol_price = 0.0
        self.vwap_cumulative_volume = 0.0

        # Live price history (for synthetic ATR)
        self.live_price_history = []

    async def fetch_spot(self):
        """Fetch spot price using client's spot_key"""
        try:
            if not self.client.spot_key and not self.client.index_key:
                logger.error("⚠️ No spot/index key available to fetch spot price")
                return None
            key = self.client.spot_key or self.client.index_key
            data = await self.client.get_quote(key)
            if not data:
                return None
            ltp = data.get('last_price') or data.get('ltp') or data.get('lastPrice')
            if ltp is None:
                return None
            return float(ltp)
        except Exception as e:
            logger.error(f"❌ Spot error: {e}")
            return None

    async def fetch_futures_candles(self):
        """Fetch futures candles with freeze detection and robust parsing"""
        try:
            if not self.client.futures_key:
                logger.error("⚠️ No futures_key available to fetch candles")
                return None

            data = await self.client.get_candles(self.client.futures_key, '1minute')
            if not data:
                return None

            # data could be dict with 'candles' or direct list
            candles = data.get('candles') if isinstance(data, dict) and 'candles' in data else data
            if not candles:
                return None

            # If items are dicts with named fields
            if isinstance(candles[0], dict):
                df = pd.DataFrame(candles)
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    # try to map common variations
                    df_columns = [c.lower() for c in df.columns]
                    if all(x in df_columns for x in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
                        df.columns = [c.lower() for c in df.columns]
                    else:
                        logger.error("❌ Candles missing required columns")
                        return None

                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
            else:
                # list of lists -> assume standard shape
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

            # normalize timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_timestamp = df['timestamp'].iloc[-1]
            current_time = datetime.now(IST)

            # Freeze detection
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
                    logger.info("✅ CANDLE API RECOVERED!")
                self.candle_repeat_count = 0
                self.candle_frozen = False
                self.candle_freeze_start = None
                self.last_candle_timestamp = latest_timestamp

            if not self.candle_frozen:
                logger.info(f"📊 CANDLE STATUS: Active (Latest: {latest_timestamp.strftime('%H:%M')})")
            else:
                logger.warning("⚠️ CANDLE STATUS: FROZEN MODE - Using live data only")

            return df

        except Exception as e:
            logger.error(f"❌ Futures candles error: {e}")
            return None

    async def fetch_futures_ltp(self):
        """Fetch LIVE futures price (last traded)"""
        try:
            if not self.client.futures_key:
                logger.error("⚠️ No futures_key available to fetch LTP")
                return None

            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None

            ltp = data.get('last_price') or data.get('ltp') or data.get('lastPrice')
            if ltp is None:
                return None
            price = float(ltp)

            # Track for synthetic ATR calculation
            self.live_price_history.append({
                'price': price,
                'time': datetime.now(IST)
            })
            self.live_price_history = self.live_price_history[-50:]

            return price

        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None

    async def fetch_futures_live_volume(self):
        """
        Fetch LIVE cumulative volume and compute delta (1-min)
        Returns: (cumulative_volume, delta_volume, avg_delta)
        """
        try:
            if not self.client.futures_key:
                logger.error("⚠️ No futures_key to fetch live volume")
                return None, None, None

            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None, None, None

            cumulative_volume = data.get('volume') or data.get('total_volume') or 0
            try:
                cumulative_volume = float(cumulative_volume)
            except:
                cumulative_volume = 0.0

            if cumulative_volume == 0:
                logger.warning("⚠️ Live volume = 0")
                return cumulative_volume, None, None

            current_time = datetime.now(IST)
            delta_volume = None
            if self.previous_cumulative_volume > 0:
                delta_volume = cumulative_volume - self.previous_cumulative_volume
                if delta_volume < 0:
                    # sometime cumulative resets on new day; reset history
                    self.volume_history = []
                    delta_volume = None
                else:
                    self.volume_history.append({'delta': delta_volume, 'time': current_time})
                    self.volume_history = self.volume_history[-20:]

            avg_delta = None
            if len(self.volume_history) >= 3:
                deltas = [v['delta'] for v in self.volume_history]
                avg_delta = sum(deltas) / len(deltas)

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
        """Incremental VWAP update using live delta volume"""
        if price is None or volume is None or volume <= 0:
            return self.live_vwap
        try:
            self.vwap_cumulative_vol_price += (price * volume)
            self.vwap_cumulative_volume += volume
            if self.vwap_cumulative_volume > 0:
                self.live_vwap = round(self.vwap_cumulative_vol_price / self.vwap_cumulative_volume, 2)
            return self.live_vwap
        except Exception as e:
            logger.error(f"❌ Live VWAP error: {e}")
            return self.live_vwap

    def calculate_synthetic_atr(self, periods=14):
        """Calculate ATR-like measure from live price changes"""
        if len(self.live_price_history) < 3:
            return ATR_FALLBACK
        try:
            recent_prices = [p['price'] for p in self.live_price_history[-periods:]]
            ranges = []
            for i in range(1, len(recent_prices)):
                ranges.append(abs(recent_prices[i] - recent_prices[i-1]))
            if not ranges:
                return ATR_FALLBACK
            atr = sum(ranges) / len(ranges)
            logger.info(f"📊 SYNTHETIC ATR: {atr:.1f} (from {len(ranges)} price moves)")
            return round(atr, 2)
        except Exception as e:
            logger.error(f"❌ Synthetic ATR error: {e}")
            return ATR_FALLBACK

    def is_candle_frozen(self):
        return self.candle_frozen

    async def fetch_option_chain(self, reference_price):
        """
        Fetch option chain for weekly expiry using client's index_key (preferred).
        Returns: (atm, strike_data) or None
        """
        try:
            if not self.client.index_key and not self.client.spot_key:
                logger.error("❌ No index_key/spot to fetch option chain")
                return None

            expiry = get_next_weekly_expiry()
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)

            # prefer using index_key (canonical)
            tried_keys = []
            candidate_keys = []
            if self.client.index_key:
                candidate_keys.append(self.client.index_key)
            # also try client.spot_key if different
            if self.client.spot_key and self.client.spot_key not in candidate_keys:
                candidate_keys.append(self.client.spot_key)
            # as last resort try futures_key trading_symbol (but usually invalid)
            if self.client.futures_key and self.client.futures_key not in candidate_keys:
                candidate_keys.append(self.client.futures_key)

            strike_data = {}
            successful = False
            for key in candidate_keys:
                tried_keys.append(key)
                data = await self.client.get_option_chain(key, expiry)
                if not data:
                    logger.warning(f"⚠️ Option chain empty for instrument_key attempts: {tried_keys}")
                    continue

                # parse response: could be list or dict (consistent mapping below)
                try:
                    if isinstance(data, list):
                        for item in data:
                            # item may be nested or structured - try to extract strike
                            strike = item.get('strike_price') or item.get('strike') or item.get('strikePrice') or None
                            if strike is None:
                                continue
                            strike = float(strike)
                            if strike < min_strike or strike > max_strike:
                                continue
                            ce_data = item.get('call_options', {}) or item.get('CE', {}) or {}
                            pe_data = item.get('put_options', {}) or item.get('PE', {}) or {}
                            ce_market = ce_data.get('market_data', {}) if isinstance(ce_data, dict) else ce_data
                            pe_market = pe_data.get('market_data', {}) if isinstance(pe_data, dict) else pe_data
                            strike_data[strike] = {
                                'ce_oi': float(ce_market.get('oi') or 0),
                                'pe_oi': float(pe_market.get('oi') or 0),
                                'ce_vol': float(ce_market.get('volume') or 0),
                                'pe_vol': float(pe_market.get('volume') or 0),
                                'ce_ltp': float(ce_market.get('ltp') or 0),
                                'pe_ltp': float(pe_market.get('ltp') or 0)
                            }
                    elif isinstance(data, dict):
                        for key_k, item in data.items():
                            strike = item.get('strike_price') or item.get('strike') or None
                            if strike is None:
                                continue
                            strike = float(strike)
                            if strike < min_strike or strike > max_strike:
                                continue
                            ce_data = item.get('call_options', {}) or item.get('CE', {}) or {}
                            pe_data = item.get('put_options', {}) or item.get('PE', {}) or {}
                            ce_market = ce_data.get('market_data', {}) if isinstance(ce_data, dict) else ce_data
                            pe_market = pe_data.get('market_data', {}) if isinstance(pe_data, dict) else pe_data
                            strike_data[strike] = {
                                'ce_oi': float(ce_market.get('oi') or 0),
                                'pe_oi': float(pe_market.get('oi') or 0),
                                'ce_vol': float(ce_market.get('volume') or 0),
                                'pe_vol': float(pe_market.get('volume') or 0),
                                'ce_ltp': float(ce_market.get('ltp') or 0),
                                'pe_ltp': float(pe_market.get('ltp') or 0)
                            }
                    else:
                        logger.warning("⚠️ Option chain data format unexpected")
                        continue

                    if strike_data:
                        successful = True
                        logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI approx: {sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values()):,.0f})")
                        break

                except Exception as e:
                    logger.error(f"❌ Option chain parse error for key {key}: {e}")
                    continue

            if not successful:
                logger.error(f"❌ Option chain returned None for tried keys: {tried_keys}")
                return None

            # final total OI check
            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                logger.warning("⚠️ Option chain returned strikes but total OI == 0")
                return None

            # return ATM as int (strike step assumed STRIKE_GAP)
            return int(atm), strike_data

        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None
