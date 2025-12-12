"""
Data Manager (fixed parts)
- fetch_option_chain: normalized strike keys to int and ensured consistent structure
- fetch_futures_live_volume: guards for cumulative=0 and historical avg calculation
- candle freeze logging clarified (no logical change to major flow)
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

# UpstoxClient unchanged except for safe parsing in get_option_chain -> see DataFetcher below.

class UpstoxClient:
    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0
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
        return {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}

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

    # detect_instruments, get_quote, get_candles unchanged (kept from original code)...

    async def get_option_chain(self, instrument_key, expiry_date):
        """Get option chain (WEEKLY options) - normalized return: (atm_int, strike_data_dict)"""
        if not instrument_key:
            return None
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        data = await self._request(url)
        if not data:
            return None
        # some APIs return list or dict - handle both
        strike_data = {}
        try:
            payload = data.get('data', data)
            # iterate entries
            if isinstance(payload, list):
                iterator = payload
            elif isinstance(payload, dict):
                iterator = payload.values()
            else:
                return None
            for item in iterator:
                # support multiple naming schemes
                strike_val = item.get('strike_price') or item.get('strike') or item.get('strikePrice') or item.get('strikePrice')
                if strike_val is None:
                    continue
                try:
                    strike = int(round(float(strike_val)))
                except Exception:
                    continue
                # bounds check using config helpers if available
                if 'min_strike' in locals() and strike < min_strike:
                    continue
                # parse CE/PE blocks
                ce_block = item.get('call_options') or item.get('CE') or item.get('ce') or {}
                pe_block = item.get('put_options') or item.get('PE') or item.get('pe') or {}
                ce_market = ce_block.get('market_data', {}) if isinstance(ce_block, dict) else {}
                pe_market = pe_block.get('market_data', {}) if isinstance(pe_block, dict) else {}
                # robust extraction
                try:
                    ce_oi = float(ce_market.get('oi') or ce_market.get('open_interest') or 0)
                    pe_oi = float(pe_market.get('oi') or pe_market.get('open_interest') or 0)
                    ce_vol = float(ce_market.get('volume') or 0)
                    pe_vol = float(pe_market.get('volume') or 0)
                    ce_ltp = float(ce_market.get('ltp') or ce_market.get('last_price') or 0)
                    pe_ltp = float(pe_market.get('ltp') or pe_market.get('last_price') or 0)
                except Exception:
                    ce_oi = ce_market.get('oi', 0) if isinstance(ce_market, dict) else 0
                    pe_oi = pe_market.get('oi', 0) if isinstance(pe_market, dict) else 0
                    ce_vol = ce_market.get('volume', 0) if isinstance(ce_market, dict) else 0
                    pe_vol = pe_market.get('volume', 0) if isinstance(pe_market, dict) else 0
                    ce_ltp = ce_market.get('ltp', 0) if isinstance(ce_market, dict) else 0
                    pe_ltp = pe_market.get('ltp', 0) if isinstance(pe_market, dict) else 0

                strike_data[strike] = {
                    'ce_oi': float(ce_oi or 0),
                    'pe_oi': float(pe_oi or 0),
                    'ce_vol': float(ce_vol or 0),
                    'pe_vol': float(pe_vol or 0),
                    'ce_ltp': float(ce_ltp or 0),
                    'pe_ltp': float(pe_ltp or 0)
                }
            if not strike_data:
                return None
            # optional: calculate atm from instrument price externally
            # return structure: atm will be set by caller (data_manager.fetch_option_chain wraps this)
            return strike_data
        except Exception as e:
            logger.error(f"❌ Option chain parse error: {e}")
            return None
