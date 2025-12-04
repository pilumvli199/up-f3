"""
Configuration & Settings
All constants, thresholds, and instrument logic
FIXED: API URLs updated to December 2025 specs
"""
import os
from datetime import datetime, timedelta, time

# ==================== API Configuration ====================
API_VERSION = 'v3'
UPSTOX_BASE_URL = 'https://api.upstox.com'

# FIXED: Updated endpoint URLs for V3
UPSTOX_QUOTE_URL_V3 = f'{UPSTOX_BASE_URL}/v3/market-quote/ltp'  # Changed from /v3/quote
UPSTOX_HISTORICAL_URL_V3 = f'{UPSTOX_BASE_URL}/v3/historical-candle/intraday'  # Changed format
UPSTOX_OPTION_CHAIN_URL = f'{UPSTOX_BASE_URL}/v2/option/chain'  # V2 still active

UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', '')

# ==================== Memory & Storage ====================
REDIS_URL = os.getenv('REDIS_URL', None)
MEMORY_TTL_SECONDS = 14400  # 4 hours
SCAN_INTERVAL = 60  # seconds

# ==================== Market Timings ====================
PREMARKET_START = time(9, 10)
PREMARKET_END = time(9, 20)
SIGNAL_START = time(9, 25)
MARKET_CLOSE = time(15, 30)
WARMUP_MINUTES = 10

# ==================== Trading Thresholds ====================
OI_THRESHOLD_STRONG = 3.0
OI_THRESHOLD_MEDIUM = 1.5
ATM_OI_THRESHOLD = 2.0
OI_5M_THRESHOLD = 1.5

VOL_SPIKE_MULTIPLIER = 1.5
PCR_BULLISH = 1.2
PCR_BEARISH = 0.8

ATR_PERIOD = 14
ATR_TARGET_MULTIPLIER = 2.5
ATR_SL_MULTIPLIER = 1.5
ATR_SL_GAMMA_MULTIPLIER = 2.0

VWAP_BUFFER = 3
MIN_CANDLE_SIZE = 5

# ==================== Risk Management ====================
USE_PREMIUM_SL = True
PREMIUM_SL_PERCENT = 30

ENABLE_TRAILING_SL = True
TRAILING_SL_TRIGGER = 0.6
TRAILING_SL_DISTANCE = 0.4

SIGNAL_COOLDOWN_SECONDS = 180
MIN_PRIMARY_CHECKS = 2
MIN_CONFIDENCE = 70

# ==================== Exit Logic Thresholds ====================
EXIT_OI_REVERSAL_THRESHOLD = 1.0  # If OI reverses by +1%
EXIT_VOLUME_DRY_THRESHOLD = 0.8  # Volume drops below 80% avg
EXIT_PREMIUM_DROP_PERCENT = 10  # Exit if premium drops 10% from peak
EXIT_CANDLE_REJECTION_MULTIPLIER = 2  # Wick > 2x body

# ==================== Telegram ====================
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ==================== Logging ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ==================== NIFTY Instrument Config ====================
# FIXED: Correct format for Upstox API
NIFTY_SPOT_KEY = "NSE_INDEX|Nifty 50"  # With space - correct format
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"  # With space - correct format

STRIKE_GAP = 50
LOT_SIZE = 50
ATR_FALLBACK = 30


def get_next_tuesday_expiry():
    """Get next Tuesday expiry (weekly)"""
    today = datetime.now()
    days_ahead = 1 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_tuesday = today + timedelta(days=days_ahead)
    return next_tuesday.strftime('%Y-%m-%d')


def get_futures_contract_name():
    """Generate NIFTY futures contract name"""
    expiry = datetime.strptime(get_next_tuesday_expiry(), '%Y-%m-%d')
    year = expiry.strftime('%y')
    month = expiry.strftime('%b').upper()
    return f"NIFTY{year}{month}FUT"


def get_nifty_futures_key():
    """Get NIFTY futures instrument key"""
    return f"NSE_FO|{get_futures_contract_name()}"


def calculate_atm_strike(spot_price):
    """Calculate ATM strike"""
    return round(spot_price / STRIKE_GAP) * STRIKE_GAP


def get_strike_range(atm_strike, num_strikes=2):
    """Get min/max strike range"""
    min_strike = atm_strike - (num_strikes * STRIKE_GAP)
    max_strike = atm_strike + (num_strikes * STRIKE_GAP)
    return min_strike, max_strike
