"""
Fixed data_manager.py
- Robust instrument detection (prefer NIFTY 50 exact or configured keys)
- Option chain fetch using proper instrument_key formats (NSE_INDEX|Nifty 50) and expiry
- Clear fallbacks and retries
- RedisBrain included (RAM fallback)
- DataFetcher provides fetch_spot, fetch_futures_candles, fetch_futures_ltp, fetch_futures_live_volume, fetch_option_chain
- Better parsing of Upstox responses and helpful logging
"""

import asyncio
import aiohttp
import json
import time as time_module
import gzip
from datetime import datetime, timedelta
from urllib.parse import quote
import pandas as pd
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config import *
from utils import IST, setup_logger

logger = setup_logger("data_manager")

MEMORY_TTL_SECONDS = MEMORY_TTL_HOURS * 3600


# ==================== Redis Brain (kept self-contained) ====================
class RedisBrain:
    """Memory manager with 24 hour TTL (Redis if available, else RAM)
    This is intentionally self-contained so other modules can import RedisBrain from data_manager.
    """
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
        test_key = f"nifty:total:{test_time.strftime('%Y%m%d_%H%M')}"

        has_data = False
        if self.client:
            try:
                has_data = self.client.exists(test_key) > 0
            except Exception:
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


# ==================== Upstox Client ====================
class UpstoxClient:
    """Upstox API V2 Client with robust instrument detection."""
    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.12
        self._last_request = 0

        # Instrument keys
        self.spot_key = NIFTY_SPOT_KEY or None
        self.index_key = NIFTY_INDEX_KEY or None
        self.futures_key = NIFTY_FUTURES_KEY or None
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
        await self._rate_limit()
        for attempt in range(3):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.get(url, headers=self._get_headers(), params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(text)
                        except Exception:
                            return None
                    elif resp.status == 429:
                        logger.warning(f"⚠️ Rate limit, retry {attempt+1}/3")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"❌ API error {resp.status}: {text}")
                        return {'status_code': resp.status, 'text': text}
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
        logger.info("🔍 Auto-detecting NIFTY instruments...")
        # If user configured keys present, trust them
        if self.spot_key and self.futures_key:
            logger.info("ℹ️ Using configured instrument keys from config")
            return True

        # Fetch instruments JSON (compressed) and parse
        try:
            url = UPSTOX_INSTRUMENTS_URL
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.error(f"❌ Instruments fetch failed: {resp.status}")
                    return False
                content = await resp.read()
                try:
                    json_text = gzip.decompress(content).decode('utf-8')
                except Exception:
                    json_text = content.decode('utf-8')
                instruments = json.loads(json_text)

            # Prefer exact match 'Nifty 50' to avoid picking index variants
            found_spot = None
            for instrument in instruments:
                seg = instrument.get('segment', '')
                name = instrument.get('name', '') or ''
                sym = instrument.get('trading_symbol', '') or ''
                if seg == 'NSE_INDEX' and ('NIFTY 50' in name.upper() or sym.upper() == 'NIFTY'):
                    found_spot = instrument
                    break

            # Fallback: try trading_symbol contains 'NIFTY'
            if not found_spot:
                for instrument in instruments:
                    seg = instrument.get('segment', '')
                    name = instrument.get('name', '') or ''
                    sym = instrument.get('trading_symbol', '') or ''
                    if seg == 'NSE_INDEX' and 'NIFTY' in sym.upper():
                        found_spot = instrument
                        break

            if not found_spot:
                logger.error("❌ NIFTY spot not found in instruments JSON")
                return False

            self.spot_key = found_spot.get('instrument_key')
            self.index_key = self.spot_key
            logger.info(f"✅ Spot: {self.spot_key}")

            # Find FUTURES (MONTHLY) for same underlying name 'NIFTY' in NSE_FO
            now = datetime.now(IST)
            all_futures = []
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') != 'FUT':
                    continue
                if instrument.get('name') and 'NIFTY' in (instrument.get('name') or '').upper():
                    expiry_ms = instrument.get('expiry') or 0
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
                                'days_to_expiry': days_to_expiry
                            })
                    except Exception:
                        continue

            if not all_futures:
                logger.warning("⚠️ No future contracts found for NIFTY in instruments list")
                return True

            all_futures.sort(key=lambda x: x['expiry'])

            monthly_futures = None
            for fut in all_futures:
                # choose first with >10 days to expiry to avoid weekly
                if fut['days_to_expiry'] > 10:
                    monthly_futures = fut
                    break
            if not monthly_futures:
                monthly_futures = all_futures[0]
                logger.warning("⚠️ Using nearest futures")

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
        # fallback: return first quote for same segment
        segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
        for key in quotes.keys():
            if key.startswith(segment):
                return quotes[key]
        logger.error(f"❌ Instrument not found in quote response for {instrument_key}")
        return None

    async def get_candles(self, instrument_key, interval='1minute'):
        if not instrument_key:
            return None
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_HISTORICAL_URL}/intraday/{encoded}/{interval}"
        data = await self._request(url)
        if not data or 'data' not in data:
            return None
        return data['data']

    async def get_option_chain(self, instrument_key, expiry_date):
        if not instrument_key:
            return None
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        data = await self._request(url)
        if not data:
            return None
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return None


# ==================== Data Fetcher ====================
class DataFetcher:
    """Centralized fetcher with volume delta, candle freeze detection, live VWAP support."""
    def __init__(self, client: UpstoxClient):
        self.client = client
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
        self.candle_frozen = False
        self.candle_freeze_start = None
        self.previous_cumulative_volume = 0
        self.previous_volume_time = None
        self.volume_history = []
        self.live_vwap = None
        self.vwap_cumulative_vol_price = 0
        self.vwap_cumulative_volume = 0
        self.live_price_history = []

    # --- spot ---
    async def fetch_spot(self):
        try:
            if not self.client.spot_key:
                logger.warning("⚠️ No spot key available")
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

    # --- futures candles ---
    async def fetch_futures_candles(self):
        try:
            if not self.client.futures_key:
                return None
            data = await self.client.get_candles(self.client.futures_key, '1minute')
            if not data:
                return None
            # Upstox may return a dict with 'candles' or a list
            candles = data.get('candles') if isinstance(data, dict) and 'candles' in data else data
            if not candles:
                return None
            if isinstance(candles[0], dict):
                df = pd.DataFrame(candles)
            else:
                # list form [ts, open, high, low, close, volume, oi]
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            # Normalize types
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
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

    # --- futures ltp ---
    async def fetch_futures_ltp(self):
        try:
            if not self.client.futures_key:
                return None
            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None
            ltp = data.get('last_price') or data.get('ltp')
            if ltp is None:
                return None
            price = float(ltp)
            self.live_price_history.append({'price': price, 'time': datetime.now(IST)})
            self.live_price_history = self.live_price_history[-50:]
            return price
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None

    # --- live volume delta ---
    async def fetch_futures_live_volume(self):
        try:
            if not self.client.futures_key:
                return None, None, None
            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None, None, None
            cumulative_volume = data.get('volume') or data.get('total_volume') or 0
            if cumulative_volume == 0:
                logger.warning("⚠️ Live volume = 0")
                return None, None, None
            current_time = datetime.now(IST)
            cumulative_volume = float(cumulative_volume)
            delta_volume = None
            if self.previous_cumulative_volume > 0:
                delta_volume = cumulative_volume - self.previous_cumulative_volume
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

    # --- live vwap updater ---
    def update_live_vwap(self, price, volume):
        if volume is None or volume <= 0:
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
        if len(self.live_price_history) < 2:
            return ATR_FALLBACK
        try:
            recent_prices = [p['price'] for p in self.live_price_history[-periods:]]
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
        return self.candle_frozen

    # --- option chain fetch with robust key attempts ---
    async def fetch_option_chain(self, reference_price):
        """
        Returns: (atm, strike_data) or None
        strike_data: dict mapping strike-> {ce_oi, pe_oi, ce_vol, pe_vol, ce_ltp, pe_ltp}
        """
        try:
            if not self.client.index_key and not self.client.spot_key:
                logger.error("❌ No index/spot key available to fetch option chain")
                return None

            # prefer index_key (NSE_INDEX|Nifty 50)
            tried_keys = []
            expiry = get_next_weekly_expiry()

            candidates = []
            if self.client.index_key:
                candidates.append(self.client.index_key)
            # try normalized known tokens (common variants)
            if self.client.spot_key and self.client.spot_key not in candidates:
                candidates.append(self.client.spot_key)
            # lastly try futures symbol name (some setups use underlying name)
            if self.client.futures_symbol:
                candidates.append(self.client.futures_symbol)

            strike_data = {}
            atm = None
            min_strike = None
            max_strike = None

            for key in candidates:
                if not key:
                    continue
                tried_keys.append(key)
                logger.info(f"🔎 Trying option chain for key: {key} expiry: {expiry}")
                data = await self.client.get_option_chain(key, expiry)
                # Accept dictionary of strikes or list
                if not data:
                    logger.warning(f"⚠️ Option chain empty for instrument_key attempts: {tried_keys}")
                    continue

                parsed = {}
                # data may be list or dict keyed by strike
                if isinstance(data, list):
                    for item in data:
                        strike = item.get('strike_price') or item.get('strike')
                        if strike is None:
                            continue
                        strike = float(strike)
                        # parse CE/PE nodes carefully
                        ce_node = item.get('call_options') or item.get('CE') or {}
                        pe_node = item.get('put_options') or item.get('PE') or {}
                        ce_market = ce_node.get('market_data', {}) if isinstance(ce_node, dict) else {}
                        pe_market = pe_node.get('market_data', {}) if isinstance(pe_node, dict) else {}
                        parsed[strike] = {
                            'ce_oi': float(ce_market.get('oi') or 0),
                            'pe_oi': float(pe_market.get('oi') or 0),
                            'ce_vol': float(ce_market.get('volume') or 0),
                            'pe_vol': float(pe_market.get('volume') or 0),
                            'ce_ltp': float(ce_market.get('ltp') or 0),
                            'pe_ltp': float(pe_market.get('ltp') or 0),
                        }
                elif isinstance(data, dict):
                    # some responses use strike keys
                    for k, item in data.items():
                        # try to extract strike value
                        strike = item.get('strike_price') or item.get('strike') or k
                        try:
                            strike = float(strike)
                        except Exception:
                            continue
                        ce_node = item.get('call_options') or item.get('CE') or {}
                        pe_node = item.get('put_options') or item.get('PE') or {}
                        ce_market = ce_node.get('market_data', {}) if isinstance(ce_node, dict) else {}
                        pe_market = pe_node.get('market_data', {}) if isinstance(pe_node, dict) else {}
                        parsed[strike] = {
                            'ce_oi': float(ce_market.get('oi') or 0),
                            'pe_oi': float(pe_market.get('oi') or 0),
                            'ce_vol': float(ce_market.get('volume') or 0),
                            'pe_vol': float(pe_market.get('volume') or 0),
                            'ce_ltp': float(ce_market.get('ltp') or 0),
                            'pe_ltp': float(pe_market.get('ltp') or 0),
                        }
                # Filter strikes by desired fetch range
                if not parsed:
                    logger.warning(f"⚠️ Parsed option chain empty for {key}")
                    continue

                # Build ATM from reference_price (futures-based)
                atm_calc = calculate_atm_strike(reference_price)
                min_fetch, max_fetch = get_strike_range_fetch(atm_calc)

                strike_data = {s: v for s, v in parsed.items() if s >= min_fetch and s <= max_fetch}

                if not strike_data:
                    logger.warning(f"⚠️ No strikes in desired range for {key} (expected {min_fetch}-{max_fetch})")
                    continue

                # success
                atm = atm_calc
                logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI: {sum(d['ce_oi']+d['pe_oi'] for d in strike_data.values()):,.0f})")
                return atm, strike_data

            # if loop finishes
            logger.error(f"❌ Option chain returned None for tried keys: {tried_keys}")
            return None

        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None


# End of file
