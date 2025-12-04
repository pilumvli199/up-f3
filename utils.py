FILE 2: utils.py (200 lines)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Utilities: Logging, Time Helpers, Validators
"""

import logging
import sys
from datetime import datetime
import pytz

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Import from config
from config import (
    LOG_LEVEL, PREMARKET_START, PREMARKET_END, 
    SIGNAL_START, MARKET_CLOSE
)

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')


# ==================== Logger Setup ====================
def setup_logger(name="nifty_bot"):
    """Setup colored logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    logger.handlers.clear()
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    if COLORLOG_AVAILABLE:
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    
    return logger


# ==================== Time Utilities ====================
def get_ist_time():
    """Get current IST time"""
    return datetime.now(IST)


def is_premarket():
    """Check if premarket time"""
    t = get_ist_time().time()
    return PREMARKET_START <= t < PREMARKET_END


def is_signal_time():
    """Check if signal generation time"""
    t = get_ist_time().time()
    return SIGNAL_START <= t < MARKET_CLOSE


def is_market_open():
    """Check if market is open"""
    from datetime import time as dt_time
    t = get_ist_time().time()
    return dt_time(9, 15) <= t < MARKET_CLOSE


def is_market_closed():
    """Check if market closed"""
    return not is_market_open()


def get_market_status():
    """Get market status tuple (status, description)"""
    t = get_ist_time().time()
    
    if PREMARKET_START <= t < PREMARKET_END:
        return 'PREMARKET', 'Loading previous data'
    elif PREMARKET_END <= t < SIGNAL_START:
        return 'WARMUP', 'Collecting data'
    elif SIGNAL_START <= t < MARKET_CLOSE:
        return 'OPEN', 'Monitoring signals'
    else:
        return 'CLOSED', 'Market closed'


def format_time_ist(dt):
    """Format datetime in IST"""
    return dt.astimezone(IST).strftime('%H:%M:%S IST')


# ==================== Validators ====================
def validate_price(price, name="Price"):
    """Validate price value"""
    if price is None or not isinstance(price, (int, float)):
        return False
    if price <= 0:
        return False
    if price < 10000 or price > 50000:
        logging.warning(f"⚠️ {name} outside normal range: {price}")
    return True


def validate_strike_data(strike_data, min_strikes=3):
    """Validate option chain data"""
    if not strike_data or not isinstance(strike_data, dict):
        return False
    if len(strike_data) < min_strikes:
        return False
    
    for strike, data in strike_data.items():
        if not isinstance(strike, int) or not isinstance(data, dict):
            return False
        required = ['ce_oi', 'pe_oi', 'ce_vol', 'pe_vol']
        if not all(f in data for f in required):
            return False
    
    return True


def validate_candle_data(df, min_candles=10):
    """Validate futures candle DataFrame"""
    if df is None or len(df) < min_candles:
        return False
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check OHLC relationship
    if (df['high'] < df['low']).any():
        return False
    
    return True
