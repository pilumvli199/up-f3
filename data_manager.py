# ═══════════════════════════════════════════════════════════════════════════════
# FILE 3: data_manager.py (400 lines)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Data Manager: Upstox API + Redis Memory
Handles all data fetching and OI storage
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


# ==================== Upstox Client ====================
class UpstoxClient:
    """Upstox API V3 Client"""
    
    def __init__(self):
        self.session = None
        self._rate_limit_delay = 0.1
        self._last_request = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
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
                async with self.session.get(url, headers=self._get_headers(), params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"API error: {resp.status}")
                        return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return None
        return None
    
    async def get_quote(self, instrument_key):
        """Get market quote"""
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_QUOTE_URL_V3}?symbol={encoded}"
        data = await self._request(url)
        return data['data'].get(instrument_key) if data and 'data' in data else None
    
    async def get_candles(self, instrument_key, interval='1minute'):
        """Get historical candles"""
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_HISTORICAL_URL_V3}/intraday/{encoded}/{interval}"
        data = await self._request(url)
        return data['data'] if data and 'data' in data else None
    
    async def get_option_chain(self, instrument_key, expiry_date):
        """Get option chain"""
        encoded = quote(instrument_key, safe='')
        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={encoded}&expiry_date={expiry_date}"
        data = await self._request(url)
        return data['data'] if data and 'data' in data else None


# ==================== Redis Memory Manager ====================
class RedisBrain:
    """Memory manager for OI snapshots"""
    
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
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis failed: {e}. Using RAM.")
                self.client = None
        else:
            logger.info("💾 Using RAM-only mode")
    
    def save_total_oi(self, ce, pe):
        """Save total OI snapshot"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:total:{now.strftime('%Y%m%d_%H%M')}"
        value = json.dumps({'ce': ce, 'pe': pe})
        
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
        self._cleanup()
    
    def get_total_oi_change(self, current_ce, current_pe, minutes_ago=15):
        """Get OI change from X minutes ago"""
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
            return 0.0, 0.0, False
        
        try:
            past = json.loads(past_str)
            ce_chg = ((current_ce - past['ce']) / past['ce'] * 100) if past['ce'] > 0 else 0
            pe_chg = ((current_pe - past['pe']) / past['pe'] * 100) if past['pe'] > 0 else 0
            return ce_chg, pe_chg, True
        except:
            return 0.0, 0.0, False
    
    def save_strike(self, strike, data):
        """Save strike OI snapshot"""
        now = datetime.now(IST).replace(second=0, microsecond=0)
        key = f"nifty:strike:{strike}:{now.strftime('%Y%m%d_%H%M')}"
        value = json.dumps(data)
        
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
        
        # Try ±3 min tolerance if exact not found
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
            ce_chg = ((current_data.get('ce_oi', 0) - past.get('ce_oi', 0)) / 
                     past.get('ce_oi', 1) * 100) if past.get('ce_oi', 0) > 0 else 0
            pe_chg = ((current_data.get('pe_oi', 0) - past.get('pe_oi', 0)) / 
                     past.get('pe_oi', 1) * 100) if past.get('pe_oi', 0) > 0 else 0
            return ce_chg, pe_chg, True
        except:
            return 0.0, 0.0, False
    
    def is_warmed_up(self, minutes=10):
        """Check if enough data collected"""
        elapsed = (datetime.now(IST) - self.startup_time).total_seconds() / 60
        return elapsed >= minutes
    
    def get_stats(self):
        """Get memory statistics"""
        elapsed = (datetime.now(IST) - self.startup_time).total_seconds() / 60
        return {
            'snapshot_count': self.snapshot_count,
            'elapsed_minutes': elapsed,
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
    
    async def load_previous_day_data(self):
        """Load previous day data during premarket"""
        if self.premarket_loaded:
            return
        logger.info("📚 Loading previous day data...")
        self.premarket_loaded = True
        logger.info("✅ Previous day data loaded")


# ==================== Data Fetcher ====================
class DataFetcher:
    """High-level data fetching"""
    
    def __init__(self, client):
        self.client = client
    
    async def fetch_spot(self):
        """Fetch NIFTY spot price"""
        try:
            data = await self.client.get_quote(NIFTY_SPOT_KEY)
            return float(data.get('last_price')) if data else None
        except Exception as e:
            logger.error(f"Spot fetch error: {e}")
            return None
    
    async def fetch_futures(self):
        """Fetch futures candles"""
        try:
            key = get_nifty_futures_key()
            data = await self.client.get_candles(key, '1minute')
            
            if not data or 'candles' not in data:
                return None
            
            candles = data['candles']
            if not candles:
                return None
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            logger.error(f"Futures fetch error: {e}")
            return None
    
    async def fetch_option_chain(self, spot_price):
        """Fetch option chain"""
        try:
            expiry = get_next_tuesday_expiry()
            atm = calculate_atm_strike(spot_price)
            min_strike, max_strike = get_strike_range(atm)
            
            data = await self.client.get_option_chain(NIFTY_INDEX_KEY, expiry)
            
            if not data:
                return None
            
            strike_data = {}
            
            if isinstance(data, list):
                for item in data:
                    strike = item.get('strike_price')
                    if not strike or strike < min_strike or strike > max_strike:
                        continue
                    strike_data[strike] = {
                        'ce_oi': item.get('call_options', {}).get('open_interest', 0),
                        'pe_oi': item.get('put_options', {}).get('open_interest', 0),
                        'ce_vol': item.get('call_options', {}).get('volume', 0),
                        'pe_vol': item.get('put_options', {}).get('volume', 0),
                        'ce_ltp': item.get('call_options', {}).get('last_price', 0),
                        'pe_ltp': item.get('put_options', {}).get('last_price', 0)
                    }
            
            elif isinstance(data, dict):
                for key, item in data.items():
                    strike = item.get('strike_price')
                    if not strike or strike < min_strike or strike > max_strike:
                        continue
                    strike_data[strike] = {
                        'ce_oi': item.get('call_options', {}).get('open_interest', 0),
                        'pe_oi': item.get('put_options', {}).get('open_interest', 0),
                        'ce_vol': item.get('call_options', {}).get('volume', 0),
                        'pe_vol': item.get('put_options', {}).get('volume', 0),
                        'ce_ltp': item.get('call_options', {}).get('last_price', 0),
                        'pe_ltp': item.get('put_options', {}).get('last_price', 0)
                    }
            
            return atm, strike_data
        
        except Exception as e:
            logger.error(f"Option chain fetch error: {e}")
            return None
