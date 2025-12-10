"""
Configuration & Settings
FIXED: Monthly futures, 11 strikes fetch + 5 deep analysis, strict thresholds
Upstox API V2 endpoints verified: https://upstox.com/developer/api-documentation/
"""
import os
from datetime import datetime, timedelta, time

# ==================== API CONFIGURATION ====================
# Upstox API V2 - Verified endpoints (Dec 2024)
API_VERSION = 'v2'
UPSTOX_BASE_URL = 'https://api.upstox.com'
UPSTOX_QUOTE_URL = f'{UPSTOX_BASE_URL}/v2/market-quote/quotes'
UPSTOX_HISTORICAL_URL = f'{UPSTOX_BASE_URL}/v2/historical-candle'
UPSTOX_OPTION_CHAIN_URL = f'{UPSTOX_BASE_URL}/v2/option/chain'
UPSTOX_INSTRUMENTS_URL = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'

UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', '')

# ==================== MEMORY & STORAGE ====================
REDIS_URL = os.getenv('REDIS_URL', None)
MEMORY_TTL_HOURS = 24
MEMORY_TTL_SECONDS = MEMORY_TTL_HOURS * 3600  # 86400 seconds
SCAN_INTERVAL = 60  # seconds

# ==================== MARKET TIMINGS ====================
PREMARKET_START = time(9, 10)
PREMARKET_END = time(9, 15)
FIRST_DATA_TIME = time(9, 16)  # Start collecting at 9:16 AM
SIGNAL_START = time(9, 21)     # Allow early signals from 9:21
MARKET_CLOSE = time(15, 30)
WARMUP_MINUTES = 15
EARLY_SIGNAL_CONFIDENCE = 85   # Higher threshold before full warmup

# ==================== STRIKE CONFIGURATION ====================
# Strike Range Logic:
# - FETCH 11 strikes (ATM ± 5) for safety buffer
# - USE 5 strikes (ATM ± 2) for deep OI analysis
# - Remaining 6 strikes for total OI calculation only

STRIKE_GAP = 50
STRIKES_TO_FETCH = 5           # ATM ± 5 = 11 total strikes
STRIKES_FOR_ANALYSIS = 2       # ATM ± 2 = 5 strikes for deep analysis

# ==================== OI THRESHOLDS ====================
# Stricter thresholds for quality signals
OI_THRESHOLD_STRONG = 5.0      # Very strong unwinding
OI_THRESHOLD_MEDIUM = 2.5      # Medium unwinding
ATM_OI_THRESHOLD = 3.0         # ATM needs stronger signal
OI_5M_THRESHOLD = 2.0          # 5m minimum

# Multi-timeframe Requirements (AND logic - both must show unwinding)
MIN_OI_5M_FOR_ENTRY = 2.0      # Both timeframes minimum
MIN_OI_15M_FOR_ENTRY = 2.5     # 15m should be stronger
STRONG_OI_5M_THRESHOLD = 3.5   # Strong signal threshold
STRONG_OI_15M_THRESHOLD = 5.0  # Very strong threshold

# ==================== VOLUME THRESHOLDS ====================
VOL_SPIKE_MULTIPLIER = 1.8     # Reduced from 2.0 - allow 1.8x+ (market realistic)
VOL_SPIKE_STRONG = 3.0         # Very strong volume

# ==================== PCR THRESHOLDS ====================
PCR_BULLISH = 1.2
PCR_BEARISH = 0.8

# ==================== TECHNICAL INDICATORS ====================
ATR_PERIOD = 14
ATR_TARGET_MULTIPLIER = 2.5
ATR_SL_MULTIPLIER = 1.5
ATR_SL_GAMMA_MULTIPLIER = 2.0

# VWAP Settings
VWAP_BUFFER = 10               # Points buffer for VWAP distance
VWAP_DISTANCE_MAX_ATR_MULTIPLE = 3.0  # Max 3.0x ATR distance (was 0.5x - TOO STRICT!)
VWAP_STRICT_MODE = True        # Reject signals far from VWAP

# Candle Settings
MIN_CANDLE_SIZE = 5

# ==================== EXIT LOGIC ====================
# Exit Thresholds - Much stricter to avoid premature exits
EXIT_OI_REVERSAL_THRESHOLD = 3.0      # Sustained building (not 1.0%)
EXIT_OI_CONFIRMATION_CANDLES = 2      # Need 2 candles confirmation
EXIT_OI_SPIKE_THRESHOLD = 8.0         # Single spike threshold

EXIT_VOLUME_DRY_THRESHOLD = 0.5       # Stricter (was 0.8)
EXIT_PREMIUM_DROP_PERCENT = 15        # More lenient (was 10)
EXIT_CANDLE_REJECTION_MULTIPLIER = 2

# Minimum Hold Time - Prevent panic exits
MIN_HOLD_TIME_MINUTES = 10            # Don't exit too early
MIN_HOLD_BEFORE_OI_EXIT = 8           # Give OI time to develop

# ==================== RE-ENTRY PROTECTION ====================
SAME_STRIKE_COOLDOWN_MINUTES = 10     # No immediate re-entry same strike
OPPOSITE_SIGNAL_COOLDOWN_MINUTES = 5  # Wait after opposite signal
SAME_DIRECTION_COOLDOWN_MINUTES = 3   # Same direction minimum gap

# ==================== RISK MANAGEMENT ====================
USE_PREMIUM_SL = True
PREMIUM_SL_PERCENT = 30

ENABLE_TRAILING_SL = True
TRAILING_SL_TRIGGER = 0.6
TRAILING_SL_DISTANCE = 0.4
TRAILING_SL_UPDATE_THRESHOLD = 5      # Only notify if 5%+ move

SIGNAL_COOLDOWN_SECONDS = 180
MIN_PRIMARY_CHECKS = 2
MIN_CONFIDENCE = 70

# ==================== TELEGRAM ====================
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ==================== NIFTY CONFIG ====================
NIFTY_SPOT_KEY = None  # Auto-detected
NIFTY_INDEX_KEY = None
NIFTY_FUTURES_KEY = None

LOT_SIZE = 50
ATR_FALLBACK = 30


# ==================== HELPER FUNCTIONS ====================

def get_next_weekly_expiry():
    """
    Get next Tuesday (weekly options expiry)
    Options are WEEKLY contracts - expires every Tuesday
    """
    today = datetime.now()
    days_ahead = 1 - today.weekday()  # Tuesday = 1
    if days_ahead <= 0:
        days_ahead += 7
    next_tuesday = today + timedelta(days=days_ahead)
    return next_tuesday.strftime('%Y-%m-%d')


def get_futures_contract_name():
    """
    Generate display name for futures contract
    NOTE: Actual detection happens automatically in data_manager.py
    This is just for display purposes
    """
    # Placeholder - will be overwritten by auto-detection
    return "NIFTY_FUTURES_AUTO"


def calculate_atm_strike(spot_price):
    """Calculate ATM strike (rounded to nearest 50)"""
    return round(spot_price / STRIKE_GAP) * STRIKE_GAP


def get_strike_range_fetch(atm_strike):
    """
    Get strike range for FETCHING (11 strikes total)
    ATM ± 5 = 11 strikes covering ±250 points
    """
    min_strike = atm_strike - (STRIKES_TO_FETCH * STRIKE_GAP)
    max_strike = atm_strike + (STRIKES_TO_FETCH * STRIKE_GAP)
    return min_strike, max_strike


def get_deep_analysis_strikes(atm_strike):
    """
    Get strikes for DEEP ANALYSIS (5 strikes only)
    ATM ± 2 = 5 strikes for OI unwinding analysis
    Returns list of strikes: [ATM-100, ATM-50, ATM, ATM+50, ATM+100]
    """
    strikes = []
    for i in range(-STRIKES_FOR_ANALYSIS, STRIKES_FOR_ANALYSIS + 1):
        strikes.append(atm_strike + (i * STRIKE_GAP))
    return strikes


def is_deep_analysis_strike(strike, atm_strike):
    """Check if strike is in deep analysis range"""
    diff = abs(strike - atm_strike)
    return diff <= (STRIKES_FOR_ANALYSIS * STRIKE_GAP)
