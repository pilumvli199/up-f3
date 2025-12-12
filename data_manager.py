"""
data_manager.py
Fixed & complete version for your bot
- UpstoxClient: detects instruments robustly
- RedisBrain: memory manager (24h TTL)
- DataFetcher: includes fetch_spot, fetch_futures_candles, fetch_futures_ltp,
  fetch_futures_live_volume, update_live_vwap, calculate_synthetic_atr,
  is_candle_frozen, fetch_option_chain
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
        self.index_key = None     # preferred key for option chain
        self.futures_key = None   # futures instrument_key (monthly)
        self.futures_expiry = None
        self.futures_symbol = None
        self.futures_underlying = None

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

            index_candidates = ['NIFTY 50', 'NIFTY', 'BANKNIFTY', 'BANK NIFTY', 'NIFTY50']

            # Find spot/index
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

            if not self.spot_key:
                for instrument in instruments:
                    if instrument.get('segment') == 'NSE_INDEX':
                        self.spot_key = instrument.get('instrument_key')
                        self.index_key = self.spot_key
                        self.futures_underlying = (instrument.get('trading_symbol') or instrument.get('name') or '').upper()
                        logger.warning(f"⚠️ Fallback spot detected: {self.spot_key}")
                        break

            now = datetime.now(IST)
            all_futures = []
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') != 'FUT':
                    continue
                trading_symbol = (instrument.get('trading_symbol') or '').upper()
                name = (instrument.get('name') or '').upper()
                acceptable = False
                if self.futures_underlying:
                    if self.futures_underlying in trading_symbol or self.futures_underlying in name:
                        acceptable = True
                else:
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
                if fut['days_to_expiry'] > 10:
                    monthly_futures = fut
                    break

            if not monthly_futures:
                monthly_futures = all_futures[0]
                logger.warning("⚠️ Using nearest futures (no >10d found)")

            self.futures_key = monthly_futures['key']
            self.futures_expiry = monthly_futures['expiry']
            self.futures_symbol = monthly_futures['symbol']

            if not self.index_key and self.futures_symbol:
                self.index_key = self.spot_key
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
        if isinstance(data, dict) and 'data' in data:
            quotes = data['data']
        else:
            quotes = data
        if not quotes:
            return None
        if isinstance(quotes, dict):
            if instrument_key in quotes:
                return quotes[instrument_key]
            alt_key = instrument_key.replace('|', ':')
            if alt_key in quotes:
                return quotes[alt_key]
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
        Returns dict(strike -> {ce_oi, pe_oi, ce_vol, pe_vol, ce_ltp, pe_ltp}) or None
        """
        if not instrument_key:
            logger.debug("get_option_chain: instrument_key empty")
            return None
        try:
            if isinstance(expiry_date, datetime):
                expiry = expiry_date.strftime('%Y-%m-%d')
            else:
                expiry = str(expiry_date)
        except Exception:
            expiry = str(expiry_date)

        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry}"
        data = await self._request(url)

        if not data or (isinstance(data, dict) and not data.get('data')):
            tried = [instrument_key]
            if self.futures_key and self.futures_key not in tried:
                tried.append(self.futures_key)
                encoded2 = quote(self.futures_key, safe='')
                url2 = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded2}&expiry_date={expiry}"
                data = await self._request(url2)
                if data and data.get('data'):
                    logger.info("✅ Option chain fetched using futures_key fallback")
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

        try:
            payload = data.get('data') if isinstance(data, dict) and 'data' in data else data
            if not payload:
                return None
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
            elif isinstance(payload, dict):
                for key, item in payload.items():
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
                return None
            return strike_data
        except Exception as e:
            logger.error(f"❌ Option chain parse error: {e}")
            return None


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


# ==================== DataFetcher ====================
class DataFetcher:
    """
    DataFetcher: contains all fetch methods used by main bot loop.
    """

    def __init__(self, client):
        self.client = client

        # Candle tracking
        self.last_candle_timestamp = None
        self.candle_repeat_count = 0
        self.candle_frozen = False
        self.candle_freeze_start = None

        # Volume tracking
        self.previous_cumulative_volume = 0
        self.previous_volume_time = None
        self.volume_history = []

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
            ltp = data.get('last_price') or data.get('ltp') or data.get('last')
            if ltp is None:
                return None
            return float(ltp)
        except Exception as e:
            logger.error(f"❌ Spot error: {e}")
            return None

    async def fetch_futures_candles(self):
        """Fetch futures 1-minute candles with freeze detection"""
        try:
            if not self.client.futures_key:
                return None
            data = await self.client.get_candles(self.client.futures_key, '1minute')
            if not data:
                return None
            candles = data.get('candles') if isinstance(data, dict) and 'candles' in data else data
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
            self.live_price_history = self.live_price_history[-50:]
            return price
        except Exception as e:
            logger.error(f"❌ Futures LTP error: {e}")
            return None

    async def fetch_futures_live_volume(self):
        """Fetch live cumulative volume and compute delta"""
        try:
            if not self.client.futures_key:
                return None, None, None
            data = await self.client.get_quote(self.client.futures_key)
            if not data:
                return None, None, None
            cumulative_volume = data.get('volume') or data.get('cum_volume') or data.get('total_volume') or 0
            if not cumulative_volume:
                logger.warning("⚠️ Live volume = 0")
                return None, None, None
            current_time = datetime.now(IST)
            cumulative_volume = float(cumulative_volume)
            delta_volume = None
            if self.previous_cumulative_volume > 0:
                delta_volume = cumulative_volume - self.previous_cumulative_volume
                self.volume_history.append({'delta': delta_volume, 'time': current_time})
                self.volume_history = self.volume_history[-30:]
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
        """Update VWAP incrementally using live trade delta"""
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
        """Calculate synthetic ATR from live price moves"""
        if len(self.live_price_history) < periods:
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
        """Fetch option chain (returns (atm, strike_data) or None)"""
        try:
            if not self.client.index_key and not self.client.futures_key and not self.client.spot_key:
                logger.error("❌ No instrument keys available to fetch option chain")
                return None
            expiry = get_next_weekly_expiry()
            atm = calculate_atm_strike(reference_price)
            min_strike, max_strike = get_strike_range_fetch(atm)
            option_result = None
            tried_keys = []
            keys_to_try = []
            if getattr(self.client, "index_key", None):
                keys_to_try.append(self.client.index_key)
            if getattr(self.client, "futures_key", None) and self.client.futures_key not in keys_to_try:
                keys_to_try.append(self.client.futures_key)
            if getattr(self.client, "spot_key", None) and self.client.spot_key not in keys_to_try:
                keys_to_try.append(self.client.spot_key)
            if getattr(self.client, "futures_symbol", None):
                keys_to_try.append(self.client.futures_symbol)
            for key in keys_to_try:
                tried_keys.append(key)
                logger.info(f"🔎 Trying option chain for key: {key} expiry: {expiry}")
                res = await self.client.get_option_chain(key, expiry)
                if res:
                    option_result = res
                    logger.info(f"✅ Option chain returned for key: {key}")
                    break
            if not option_result:
                logger.error(f"❌ Option chain returned None for tried keys: {tried_keys}")
                return None
            # option_result is dict strike->data
            strike_data = {}
            for strike, dat in option_result.items():
                try:
                    strike_f = float(strike)
                except:
                    continue
                if strike_f < min_strike or strike_f > max_strike:
                    continue
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
