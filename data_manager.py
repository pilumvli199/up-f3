# data_manager.py
"""
Data Manager: Upstox API + Redis Memory (FIXED)
- Robust instrument detection (index/futures)
- Option chain: tries multiple candidate instrument_key formats
- Handles 400/invalid-instrument errors gracefully
- Exposes UpstoxClient, RedisBrain, DataFetcher for main app imports
"""

import asyncio
import aiohttp
import json
import time as time_module
from datetime import datetime, timedelta
from urllib.parse import quote
import gzip
import pandas as pd
import logging

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
    """Upstox API V2 Client with robust instrument detection & option chain helper"""

    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.12
        self._last_request = 0

        # Instrument keys discovered
        self.spot_key = None
        self.index_key = None
        self.futures_key = None
        self.futures_expiry = None
        self.futures_symbol = None

        # local instruments cache
        self.instruments = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._load_instruments_json()
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
        """Make API request with retry & basic logging"""
        await self._rate_limit()

        for attempt in range(3):
            try:
                timeout = aiohttp.ClientTimeout(total=12)
                async with self.session.get(url, headers=self._get_headers(), params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    status = resp.status
                    if status == 200:
                        try:
                            return json.loads(text)
                        except Exception:
                            return None
                    elif status == 429:
                        logger.warning(f"⚠️ Rate limited ({status}), retry {attempt+1}/3")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        # return structured info for caller to handle
                        logger.error(f"❌ API error {status}: {text[:400]}")
                        return {'__status': status, 'text': text}
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout (attempt {attempt+1}/3) for {url}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return None
            except Exception as e:
                logger.error(f"❌ Request failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(1.5)
                    continue
                return None
        return None

    async def _load_instruments_json(self):
        """Load instruments JSON (gz) once for robust mapping"""
        try:
            url = UPSTOX_INSTRUMENTS_URL
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"⚠️ Instruments fetch failed: {resp.status}")
                    self.instruments = []
                    return
                content = await resp.read()
                try:
                    decompressed = gzip.decompress(content).decode('utf-8')
                except Exception:
                    # already plain
                    decompressed = content.decode('utf-8')
                self.instruments = json.loads(decompressed)
                logger.info(f"✅ Instruments JSON loaded ({len(self.instruments)} entries)")
        except Exception as e:
            logger.error(f"❌ Instruments load failed: {e}")
            self.instruments = []

    async def detect_instruments(self):
        """Auto-detect NIFTY/BANKNIFTY instruments (index + monthly futures)"""

        logger.info("🔍 Auto-detecting instruments...")

        # Safe guard: instruments may already be loaded
        instruments = self.instruments or []

        # Utility to normalise strings
        def norm(s):
            return (s or "").strip().upper()

        # First, try to find a canonical index (prefer 'NIFTY 50' then 'NIFTY' then 'BANKNIFTY')
        spot_candidate = None
        for inst in instruments:
            seg = inst.get('segment', '')
            name = norm(inst.get('name') or inst.get('display_name') or '')
            tsym = norm(inst.get('trading_symbol') or '')
            # Prefer NIFTY 50 exact
            if seg == 'NSE_INDEX' and ('NIFTY 50' in name or 'NIFTY 50' in tsym or tsym == 'NIFTY'):
                spot_candidate = inst
                break

        # fallback: any NSE_INDEX containing 'NIFTY' or 'BANKNIFTY'
        if not spot_candidate:
            for inst in instruments:
                if inst.get('segment') == 'NSE_INDEX':
                    nm = norm(inst.get('name') or '')
                    ts = norm(inst.get('trading_symbol') or '')
                    if 'NIFTY' in nm or 'NIFTY' in ts or 'BANKNIFTY' in nm or 'BANKNIFTY' in ts:
                        spot_candidate = inst
                        break

        if spot_candidate:
            self.spot_key = spot_candidate.get('instrument_key')
            self.index_key = self.spot_key
            logger.info(f"✅ Spot/Index detected: {spot_candidate.get('name') or spot_candidate.get('trading_symbol')} -> {self.spot_key}")
        else:
            logger.error("❌ Spot/index not found in instruments list")
            self.spot_key = None
            self.index_key = None

        # Detect futures (prefer monthly futures with >10 days to expiry)
        now = datetime.now(IST)
        all_futures = []
        for inst in instruments:
            try:
                if inst.get('segment') != 'NSE_FO':
                    continue
                if inst.get('instrument_type') != 'FUT':
                    continue
                # name/trading_symbol check for index name match
                if self.spot_key:
                    # If spot trading symbol exists, filter futures by same underlying 'name' field
                    pass
                expiry_ms = inst.get('expiry') or 0
                if not expiry_ms:
                    continue
                expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                if expiry_dt > now:
                    days_to_exp = (expiry_dt - now).days
                    all_futures.append({
                        'key': inst.get('instrument_key'),
                        'expiry': expiry_dt,
                        'symbol': inst.get('trading_symbol', ''),
                        'days_to_expiry': days_to_exp,
                        'name': inst.get('name', '')
                    })
            except Exception:
                continue

        if not all_futures:
            logger.warning("⚠️ No futures found in instruments list")
            self.futures_key = None
            self.futures_expiry = None
            self.futures_symbol = None
            return False

        all_futures.sort(key=lambda x: x['expiry'])
        monthly_fut = None
        for fut in all_futures:
            if fut['days_to_expiry'] > 10:
                monthly_fut = fut
                break

        if not monthly_fut:
            monthly_fut = all_futures[0]
            logger.warning("⚠️ Using nearest futures contract as fallback")

        self.futures_key = monthly_fut['key']
        self.futures_expiry = monthly_fut['expiry']
        self.futures_symbol = monthly_fut['symbol']
        logger.info(f"✅ Futures (MONTHLY): {self.futures_symbol}")
        logger.info(f"   Expiry: {self.futures_expiry.strftime('%Y-%m-%d')} ({monthly_fut['days_to_expiry']} days)")
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
        # Prefer exact
        if instrument_key in quotes:
            return quotes[instrument_key]

        # try alt forms
        alt_key = instrument_key.replace('|', ':')
        if alt_key in quotes:
            return quotes[alt_key]

        # fallback: return first quote with same segment prefix
        segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
        for key in quotes.keys():
            if key.startswith(segment):
                return quotes[key]

        logger.error("❌ Instrument quote not found in response")
        return None

    async def get_candles(self, instrument_key, interval='1minute'):
        """Get historical candles (intraday)"""
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
        Get option chain.
        Upstox expects 'instrument_key' (e.g. 'NSE_INDEX|Nifty 50') and expiry_date 'YYYY-MM-DD'.
        We'll call endpoint and return parsed 'data' if available; otherwise return structured error.
        """
        if not instrument_key or not expiry_date:
            return {'__error': 'missing_params'}

        params = {'instrument_key': instrument_key, 'expiry_date': expiry_date}
        url = UPSTOX_OPTION_CHAIN_URL
        data = await self._request(url, params=params)

        # If API returned structured error (status wrapper), propagate it
        if data is None:
            return None

        if isinstance(data, dict) and data.get('__status') is not None:
            # pass raw info back to caller for handling
            return data

        if 'data' not in data:
            return None

        return data['data']


# ==================== Redis Brain ====================
class RedisBrain:
    """Memory manager with 24 hour TTL (Redis if available, else in-memory)"""

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
            # tolerance offsets
            for offset in [-1, 1, -2, 2]:
                alt = target + timedelta(minutes=offset)
                alt_key = f"nifty:total:{alt.strftime('%Y%m%d_%H%M')}"
                if self.client:
                    try:
                        past_str = self.client.get(alt_key)
                        if past_str:
                            break
                    except Exception:
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
            except Exception:
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
                    except Exception:
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
            except Exception:
                pass
        if not has_data:
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
        logger.info("📚 Skipping previous day data (placeholder)")
        self.premarket_loaded = True


# ==================== Data Fetcher V2 ====================
class DataFetcher:
    """
    DataFetcher with:
    - fetch_spot (from client.spot_key)
    - fetch_futures_candles
    - fetch_futures_ltp
    - fetch_futures_live_volume (delta)
    - fetch_option_chain (tries candidate instrument_keys)
    - live VWAP update & synthetic ATR
    """

    def __init__(self, client: UpstoxClient):
        self.client = client

        # Candle tracking
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
        self.candle_frozen = False
        self.candle_freeze_start = None

        # Volume delta tracking
        self.previous_cumulative_volume = 0
        self.previous_volume_time = None
        self.volume_history = []

        # Live VWAP tracking
        self.live_vwap = None
        self.vwap_cumulative_vol_price = 0.0
        self.vwap_cumulative_volume = 0.0

        # Live price history (for synthetic ATR)
        self.live_price_history = []

    async def fetch_spot(self):
        """Fetch spot price (from detected spot_key)"""
        try:
            if not self.client.spot_key:
                logger.warning("⚠️ No spot_key detected")
                return None
            data = await self.client.get_quote(self.client.spot_key)
            if not data:
                return None
            ltp = data.get('last_price') or data.get('ltp') or data.get('last')
            if ltp is None:
                return None
            return float(ltp)
        except Exception as e:
            logger.error(f"❌ Spot error: {e}")
            return None

    async def fetch_futures_candles(self):
        """Fetch futures candles with freeze detection"""
        try:
            if not self.client.futures_key:
                logger.warning("⚠️ No futures_key detected")
                return None
            data = await self.client.get_candles(self.client.futures_key, '1minute')
            if not data:
                return None
            candles = data.get('candles') if isinstance(data, dict) else data
            if not candles:
                return None

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

            latest_timestamp = df['timestamp'].iloc[-1]
            current_time = datetime.now(IST)

            if self.last_candle_timestamp == latest_timestamp:
                self.candle_repeat_count += 1
                if not self.candle_frozen and self.candle_repeat_count >= 5:
                    self.candle_frozen = True
                    self.candle_freeze_start = current_time
                    logger.error(f"🚨 CANDLE API FROZEN! Timestamp stuck at {latest_timestamp}")
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
            ltp = data.get('last_price') or data.get('ltp') or data.get('last')
            if ltp is None:
                return None
            price = float(ltp)
            self.live_price_history.append({'price': price, 'time': datetime.now(IST)})
            self.live_price_history = self.live_price_history[-40:]
            return price
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None

    async def fetch_futures_live_volume(self):
        """Fetch LIVE cumulative volume and compute delta (1-min)"""
        try:
            if not self.client.futures_key:
                return None, None, None
            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None, None, None
            cumulative_volume = data.get('volume') or data.get('cumulative_volume') or 0
            if cumulative_volume is None or cumulative_volume == 0:
                logger.warning("⚠️ Live volume = 0 or missing")
                return None, None, None
            cumulative_volume = float(cumulative_volume)
            current_time = datetime.now(IST)
            delta_volume = None
            if self.previous_cumulative_volume > 0:
                delta_volume = cumulative_volume - self.previous_cumulative_volume
                self.volume_history.append({'delta': delta_volume, 'time': current_time})
                self.volume_history = self.volume_history[-20:]
            avg_delta = None
            if len(self.volume_history) >= 3:
                deltas = [v['delta'] for v in self.volume_history if v.get('delta') is not None]
                if deltas:
                    avg_delta = sum(deltas) / len(deltas)
            self.previous_cumulative_volume = cumulative_volume
            self.previous_volume_time = current_time
            logger.info(f"📊 VOLUME (DELTA MODE): Cumulative: {cumulative_volume:,.0f}")
            if delta_volume is not None:
                logger.info(f"   Delta (1-min): {delta_volume:,.0f}")
            if avg_delta is not None:
                logger.info(f"   Avg delta: {avg_delta:,.0f}")
            return cumulative_volume, delta_volume, avg_delta
        except Exception as e:
            logger.error(f"❌ Live volume error: {e}")
            return None, None, None

    def update_live_vwap(self, price, volume):
        """Incremental VWAP update from live tick volume"""
        if volume is None or volume <= 0 or price is None:
            return self.live_vwap
        try:
            self.vwap_cumulative_vol_price += (price * volume)
            self.vwap_cumulative_volume += volume
            if self.vwap_cumulative_volume > 0:
                self.live_vwap = self.vwap_cumulative_vol_price / self.vwap_cumulative_volume
            return self.live_vwap
        except Exception as e:
            logger.error(f"❌ Live VWAP error: {e}")
            return self.live_vwap

    def calculate_synthetic_atr(self, periods=14):
        """Calculate ATR-like value from live price history"""
        if len(self.live_price_history) < 3:
            return ATR_FALLBACK
        try:
            recent = [p['price'] for p in self.live_price_history[-periods:]]
            ranges = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
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
        Fetch option chain for weekly expiry using robust instrument key attempts.
        Returns (atm, strike_data) on success, or None on failure.
        """

        try:
            if not self.client.index_key:
                # fallback: try to derive candidate instrument keys from instruments list
                logger.warning("⚠️ index_key missing; building candidates from instruments JSON")

            expiry = get_next_weekly_expiry()
            # ATM calculated from reference (futures price)
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)

            # Build candidate instrument keys to try for option chain (ordered)
            candidates = []

            # 1) If index_key present, try that exact value
            if self.client.index_key:
                candidates.append(self.client.index_key)

            # 2) Friendly name variations from detected spot name
            if self.client.spot_key and isinstance(self.client.spot_key, str):
                # spot_key often like 'NSE_INDEX|Nifty 50' or 'NSE_INDEX|NIFTY'
                candidates.append(self.client.spot_key)

            # 3) Try canonical patterns using known names from instruments list
            # Example canonical: 'NSE_INDEX|Nifty 50' or 'NSE_INDEX|NIFTY 50'
            instruments = self.client.instruments or []
            seen = set()
            for inst in instruments:
                seg = inst.get('segment', '')
                if seg != 'NSE_INDEX':
                    continue
                name = inst.get('name') or inst.get('display_name') or inst.get('trading_symbol') or ''
                if not name:
                    continue
                key = f"{seg}|{name}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
                # also try trading_symbol
                tsym = inst.get('trading_symbol')
                if tsym:
                    k2 = f"{seg}|{tsym}"
                    if k2 not in seen:
                        seen.add(k2)
                        candidates.append(k2)

            # 4) Add some common manual fallbacks (Nifty/BANKNIFTY)
            candidates.extend([
                "NSE_INDEX|Nifty 50",
                "NSE_INDEX|NIFTY 50",
                "NSE_INDEX|NIFTY",
                "NSE_INDEX|BANKNIFTY",
                "NSE_INDEX|BANK NIFTY"
            ])

            # Deduplicate while preserving order
            final_candidates = []
            for c in candidates:
                if c and c not in final_candidates:
                    final_candidates.append(c)

            tried_keys = []
            strike_data = {}

            for candidate in final_candidates:
                tried_keys.append(candidate)
                logger.info(f"🔎 Trying option chain for key: {candidate} expiry: {expiry}")
                result = await self.client.get_option_chain(candidate, expiry)
                # result may be None, dict error, or actual data
                if result is None:
                    logger.warning(f"⚠️ Option chain empty for instrument_key attempts: {tried_keys}")
                    continue
                # if API returned a structured error wrapper
                if isinstance(result, dict) and result.get('__status') is not None:
                    status = result.get('__status')
                    text = result.get('text', '')
                    logger.error(f"❌ API error for {candidate}: {status} {str(text)[:300]}")
                    continue

                # if we have 'data' (parsed earlier in client) it will be a dict/list of strikes
                # Normalize into strike_data dict: strike -> {ce_oi, pe_oi, ce_vol, pe_vol, ce_ltp, pe_ltp}
                data = result
                parsed = {}
                if isinstance(data, list):
                    for item in data:
                        strike = item.get('strike_price') or item.get('strike')
                        if strike is None:
                            continue
                        strike = float(strike)
                        if strike < min_strike or strike > max_strike:
                            continue
                        ce_data = item.get('call_options') or item.get('CE') or {}
                        pe_data = item.get('put_options') or item.get('PE') or {}
                        ce_market = ce_data.get('market_data', {}) if isinstance(ce_data, dict) else {}
                        pe_market = pe_data.get('market_data', {}) if isinstance(pe_data, dict) else {}
                        parsed[strike] = {
                            'ce_oi': float(ce_market.get('oi') or 0),
                            'pe_oi': float(pe_market.get('oi') or 0),
                            'ce_vol': float(ce_market.get('volume') or 0),
                            'pe_vol': float(pe_market.get('volume') or 0),
                            'ce_ltp': float(ce_market.get('ltp') or 0),
                            'pe_ltp': float(pe_market.get('ltp') or 0)
                        }
                elif isinstance(data, dict):
                    for key, item in data.items():
                        # item might be nested dict representing strike
                        strike = item.get('strike_price') or item.get('strike')
                        if strike is None:
                            continue
                        strike = float(strike)
                        if strike < min_strike or strike > max_strike:
                            continue
                        ce_data = item.get('call_options') or item.get('CE') or {}
                        pe_data = item.get('put_options') or item.get('PE') or {}
                        ce_market = ce_data.get('market_data', {}) if isinstance(ce_data, dict) else {}
                        pe_market = pe_data.get('market_data', {}) if isinstance(pe_data, dict) else {}
                        parsed[strike] = {
                            'ce_oi': float(ce_market.get('oi') or 0),
                            'pe_oi': float(pe_market.get('oi') or 0),
                            'ce_vol': float(ce_market.get('volume') or 0),
                            'pe_vol': float(pe_market.get('volume') or 0),
                            'ce_ltp': float(ce_market.get('ltp') or 0),
                            'pe_ltp': float(pe_market.get('ltp') or 0)
                        }
                else:
                    # unexpected format
                    logger.warning(f"⚠️ Unexpected option chain format for {candidate}")
                    continue

                if not parsed:
                    logger.warning(f"⚠️ Parsed option chain empty for {candidate}")
                    continue

                total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in parsed.values())
                if total_oi == 0:
                    logger.warning(f"⚠️ Option chain from {candidate} has zero total OI")
                    continue

                logger.info(f"✅ Parsed {len(parsed)} strikes (Total OI: {total_oi:,.0f}) using key {candidate}")
                # success!
                return atm, parsed

            # tried all candidates
            logger.error(f"❌ Option chain returned None for tried keys: {tried_keys}")
            return None

        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None
