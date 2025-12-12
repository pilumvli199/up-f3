"""
Data Manager: Upstox API + Redis Memory (FIXED)
- Improved instrument detection (spot/index/futures)
- Robust option-chain fetching with fallbacks (try index_key, futures_key, trading_symbol)
- Better parsing of API shapes (list/dict)
- Retries + 429 handling already present in _request()
- Returns (atm, strike_data) or None
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
    """Upstox API V2 Client with improved instrument detection & option-chain fallback"""

    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0

        # Instrument keys
        self.spot_key = None      # index / spot instrument_key
        self.index_key = None     # prefer index instrument_key used for option chain
        self.futures_key = None   # futures instrument_key (monthly)
        self.futures_expiry = None
        self.futures_symbol = None
        self.futures_underlying = None  # underlying index name for futures (e.g., "NIFTY" or "BANKNIFTY")

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
        """Make API request with retry (keeps previous behavior)"""
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
                            return text
                    elif resp.status == 429:
                        logger.warning(f"⚠️ Rate limit (429), retry {attempt+1}/3")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
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
        """Auto-detect NIFTY / BANKNIFTY instruments (spot + MONTHLY futures)"""
        logger.info("🔍 Auto-detecting instruments (robust)...")
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

            # Normalize search keys
            def name_matches(name, patterns):
                if not name:
                    return False
                name_up = name.upper()
                for p in patterns:
                    if p in name_up:
                        return True
                return False

            # Try to find common index names first: NIFTY, NIFTY 50, BANKNIFTY
            index_candidates = ['NIFTY 50', 'NIFTY', 'BANKNIFTY', 'BANK NIFTY', 'NIFTY50']

            # Find spot/index instruments (NSE_INDEX segment)
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_INDEX':
                    continue
                name = instrument.get('name', '') or ''
                symbol = instrument.get('trading_symbol', '') or ''
                for cand in index_candidates:
                    if cand in name.upper() or cand in symbol.upper():
                        self.spot_key = instrument.get('instrument_key')
                        self.index_key = instrument.get('instrument_key')
                        self.futures_underlying = cand.replace(' ', '').upper()
                        logger.info(f"✅ Spot/Index detected: {symbol or name} -> {self.spot_key}")
                        break
                if self.spot_key:
                    break

            # Fallback: if no index found, try any NSE_INDEX
            if not self.spot_key:
                for instrument in instruments:
                    if instrument.get('segment') == 'NSE_INDEX':
                        self.spot_key = instrument.get('instrument_key')
                        self.index_key = self.spot_key
                        self.futures_underlying = (instrument.get('trading_symbol') or instrument.get('name') or '').upper()
                        logger.warning(f"⚠️ Fallback spot detected: {self.spot_key}")
                        break

            if not self.spot_key:
                logger.error("❌ Spot/index not found")
                # We still continue to attempt finding futures though
            # Find MONTHLY futures (NSE_FO FUT)
            now = datetime.now(IST)
            all_futures = []
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') != 'FUT':
                    continue
                # Accept futures whose trading symbol contains NIFTY or BANKNIFTY or matches spot underlying
                trading_symbol = (instrument.get('trading_symbol') or '').upper()
                name = (instrument.get('name') or '').upper()
                # Determine if this future is a candidate for the detected index
                acceptable = False
                if self.futures_underlying:
                    if self.futures_underlying in trading_symbol or self.futures_underlying in name:
                        acceptable = True
                else:
                    # try common names
                    if 'NIFTY' in trading_symbol or 'BANKNIFTY' in trading_symbol or 'NIFTY' in name or 'BANKNIFTY' in name:
                        acceptable = True
                if not acceptable:
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
                            'symbol': trading_symbol,
                            'days_to_expiry': days_to_expiry,
                            'weekday': expiry_dt.strftime('%A')
                        })
                except:
                    continue

            if not all_futures:
                # fallback: pick earliest FUT in NSE_FO if none matched
                for instrument in instruments:
                    if instrument.get('segment') == 'NSE_FO' and instrument.get('instrument_type') == 'FUT':
                        expiry_ms = instrument.get('expiry', 0)
                        if not expiry_ms:
                            continue
                        try:
                            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                            days_to_expiry = (expiry_dt - now).days
                            all_futures.append({
                                'key': instrument.get('instrument_key'),
                                'expiry': expiry_dt,
                                'symbol': (instrument.get('trading_symbol') or '').upper(),
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
                # pick the first future with >10 days to expiry as MONTHLY
                if fut['days_to_expiry'] > 10:
                    monthly_futures = fut
                    break

            if not monthly_futures:
                monthly_futures = all_futures[0]
                logger.warning("⚠️ Using nearest futures (no >10d found)")

            self.futures_key = monthly_futures['key']
            self.futures_expiry = monthly_futures['expiry']
            self.futures_symbol = monthly_futures['symbol']

            # If index_key missing, try to set a sensible index_key from futures_symbol
            if not self.index_key and self.futures_symbol:
                # try derive index name
                base = self.futures_symbol.split()[0]
                self.index_key = self.spot_key  # still best-effort
                logger.warning("⚠️ index_key missing; using spot_key as fallback for option chain")

            logger.info(f"✅ Futures (MONTHLY): {self.futures_symbol}")
            logger.info(f"   Expiry: {self.futures_expiry.strftime('%Y-%m-%d')} ({monthly_futures['days_to_expiry']} days)")
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
        if not data:
            return None
        # data may come as {'data': { ... }} or direct mapping
        if isinstance(data, dict) and 'data' in data:
            quotes = data['data']
        else:
            quotes = data
        if not quotes:
            return None
        # Prefer exact key
        if isinstance(quotes, dict):
            if instrument_key in quotes:
                return quotes[instrument_key]
            alt_key = instrument_key.replace('|', ':')
            if alt_key in quotes:
                return quotes[alt_key]
            # fallback: return first quote matching segment prefix
            segment = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key.split(':')[0]
            for key in quotes.keys():
                if key.startswith(segment):
                    return quotes[key]
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
        Get option chain for the given instrument_key + expiry.
        Robust: will try alternative keys if initial attempt returns empty.
        """
        if not instrument_key:
            logger.debug("get_option_chain: instrument_key empty")
            return None
        # ensure expiry format YYYY-MM-DD
        try:
            # allow both date or string; format properly
            if isinstance(expiry_date, datetime):
                expiry = expiry_date.strftime('%Y-%m-%d')
            else:
                expiry = str(expiry_date)
        except Exception:
            expiry = str(expiry_date)

        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry}"
        data = await self._request(url)
        # If no result, try fallbacks: use futures_key (without instrument_key encoding issues) or trading symbol
        if not data or (isinstance(data, dict) and not data.get('data')):
            # try using futures_key if different
            tried = [instrument_key]
            if self.futures_key and self.futures_key not in tried:
                tried.append(self.futures_key)
                encoded2 = quote(self.futures_key, safe='')
                url2 = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded2}&expiry_date={expiry}"
                data = await self._request(url2)
                if data and data.get('data'):
                    logger.info("✅ Option chain fetched using futures_key fallback")
            # try using trading symbol (non-key) as last resort
            if (not data or (isinstance(data, dict) and not data.get('data'))) and self.futures_symbol:
                tried.append(self.futures_symbol)
                encoded3 = quote(self.futures_symbol, safe='')
                url3 = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded3}&expiry_date={expiry}"
                data = await self._request(url3)
                if data and data.get('data'):
                    logger.info("✅ Option chain fetched using symbol fallback")

            if not data or (isinstance(data, dict) and not data.get('data')):
                logger.warning(f"⚠️ Option chain empty for instrument_key attempts: {tried}")
                return None

        # Normalize returned shape
        try:
            payload = data.get('data') if isinstance(data, dict) and 'data' in data else data
            if not payload:
                return None
            # payload could be list of strikes or dict keyed by strike
            strike_data = {}
            if isinstance(payload, list):
                for item in payload:
                    strike = item.get('strike_price') or item.get('strike') or item.get('strikePrice')
                    if strike is None:
                        continue
                    try:
                        strike = float(strike)
                    except:
                        continue
                    # try CE/PE fields in multiple shapes
                    ce_block = item.get('call_options') or item.get('CE') or item.get('ce') or {}
                    pe_block = item.get('put_options') or item.get('PE') or item.get('pe') or {}
                    ce_market = ce_block.get('market_data', {}) if isinstance(ce_block, dict) else {}
                    pe_market = pe_block.get('market_data', {}) if isinstance(pe_block, dict) else {}
                    # Some APIs provide direct fields
                    ce_oi = float((ce_market.get('oi') or ce_block.get('oi') or 0) or 0)
                    pe_oi = float((pe_market.get('oi') or pe_block.get('oi') or 0) or 0)
                    ce_vol = float((ce_market.get('volume') or ce_block.get('volume') or 0) or 0)
                    pe_vol = float((pe_market.get('volume') or pe_block.get('volume') or 0) or 0)
                    ce_ltp = float((ce_market.get('ltp') or ce_block.get('ltp') or 0) or 0)
                    pe_ltp = float((pe_market.get('ltp') or pe_block.get('ltp') or 0) or 0)
                    strike_data[strike] = {
                        'ce_oi': ce_oi,
                        'pe_oi': pe_oi,
                        'ce_vol': ce_vol,
                        'pe_vol': pe_vol,
                        'ce_ltp': ce_ltp,
                        'pe_ltp': pe_ltp
                    }
            elif isinstance(payload, dict):
                # keys might be strike strings
                for key, item in payload.items():
                    # item may contain strike_price inside
                    strike = item.get('strike_price') or item.get('strike') or key
                    try:
                        strike = float(strike)
                    except:
                        continue
                    ce_block = item.get('call_options') or item.get('CE') or item.get('ce') or {}
                    pe_block = item.get('put_options') or item.get('PE') or item.get('pe') or {}
                    ce_market = ce_block.get('market_data', {}) if isinstance(ce_block, dict) else {}
                    pe_market = pe_block.get('market_data', {}) if isinstance(pe_block, dict) else {}
                    ce_oi = float((ce_market.get('oi') or ce_block.get('oi') or 0) or 0)
                    pe_oi = float((pe_market.get('oi') or pe_block.get('oi') or 0) or 0)
                    ce_vol = float((ce_market.get('volume') or ce_block.get('volume') or 0) or 0)
                    pe_vol = float((pe_market.get('volume') or pe_block.get('volume') or 0) or 0)
                    ce_ltp = float((ce_market.get('ltp') or ce_block.get('ltp') or 0) or 0)
                    pe_ltp = float((pe_market.get('ltp') or pe_block.get('ltp') or 0) or 0)
                    strike_data[strike] = {
                        'ce_oi': ce_oi,
                        'pe_oi': pe_oi,
                        'ce_vol': ce_vol,
                        'pe_vol': pe_vol,
                        'ce_ltp': ce_ltp,
                        'pe_ltp': pe_ltp
                    }
            else:
                logger.error("❌ Unknown option-chain payload shape")
                return None
            if not strike_data:
                logger.warning("⚠️ Parsed strike_data empty after processing payload")
                return None
            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                logger.warning("⚠️ Option chain total OI is zero (possible stale API or wrong expiry)")
                # still return data (caller can decide), but to keep previous behavior return None
                return None
            return strike_data
        except Exception as e:
            logger.error(f"❌ Option chain parse error: {e}")
            return None


# ==================== Redis Brain (same as before, kept unchanged) ====================
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
        expired = [k for k, ts in self.memory_timestamps.items() if now - ts > MEMORY_TTL_SECONDS]
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


# ==================== DataFetcher (option-chain usage) ====================
class DataFetcher:
    """
    DataFetcher uses UpstoxClient.get_option_chain() which now tries robust fallbacks.
    The fetch_option_chain below expects a return of strike_data (dict) or None.
    """
    def __init__(self, client):
        self.client = client
        # keep previous attributes like freeze detection, volume tracking etc.
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

    # ... (other methods like fetch_spot, fetch_futures_candles, fetch_futures_ltp remain unchanged)
    # We'll include just fetch_option_chain here since it's the fixed piece.

    async def fetch_option_chain(self, reference_price):
        """Fetch option chain with robust fallback and return (atm, strike_data)"""
        try:
            if not self.client:
                logger.error("❌ No Upstox client")
                return None

            expiry = get_next_weekly_expiry()
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)

            # Try using index_key first (preferred for option chain)
            option_data = None
            tried_keys = []
            keys_to_try = []
            if getattr(self.client, "index_key", None):
                keys_to_try.append(self.client.index_key)
            # also try futures_key
            if getattr(self.client, "futures_key", None) and self.client.futures_key not in keys_to_try:
                keys_to_try.append(self.client.futures_key)
            # also try spot_key
            if getattr(self.client, "spot_key", None) and self.client.spot_key not in keys_to_try:
                keys_to_try.append(self.client.spot_key)
            # as last resort try futures_symbol
            if getattr(self.client, "futures_symbol", None):
                keys_to_try.append(self.client.futures_symbol)

            for key in keys_to_try:
                tried_keys.append(key)
                logger.info(f"🔎 Trying option chain for key: {key} expiry: {expiry}")
                option_data = await self.client.get_option_chain(key, expiry)
                if option_data:
                    logger.info(f"✅ Option chain returned for key: {key}")
                    break

            if not option_data:
                logger.error(f"❌ Option chain returned None for tried keys: {tried_keys}")
                return None

            # option_data is a mapping strike -> data
            strike_data = {}
            for strike, dat in option_data.items():
                # ensure numeric strike and filter by requested range
                try:
                    strike_f = float(strike)
                except:
                    continue
                if strike_f < min_strike or strike_f > max_strike:
                    continue
                # Ensure keys exist
                ce_oi = float(dat.get('ce_oi', 0))
                pe_oi = float(dat.get('pe_oi', 0))
                ce_vol = float(dat.get('ce_vol', 0))
                pe_vol = float(dat.get('pe_vol', 0))
                ce_ltp = float(dat.get('ce_ltp', 0))
                pe_ltp = float(dat.get('pe_ltp', 0))
                strike_data[strike_f] = {
                    'ce_oi': ce_oi,
                    'pe_oi': pe_oi,
                    'ce_vol': ce_vol,
                    'pe_vol': pe_vol,
                    'ce_ltp': ce_ltp,
                    'pe_ltp': pe_ltp
                }

            if not strike_data:
                logger.error("❌ No strikes parsed after filtering by range")
                return None

            total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
            if total_oi == 0:
                logger.warning("⚠️ Total OI for parsed strikes is zero")
                return None

            logger.info(f"✅ Parsed {len(strike_data)} strikes (Total OI: {total_oi:,.0f})")
            return atm, strike_data

        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None
