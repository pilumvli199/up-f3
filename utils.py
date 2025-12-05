"""
Utilities: Time, Logging, Validators
Helper functions for bot operations
"""

import pytz
from datetime import datetime, time
import logging
import colorlog

# Timezone
IST = pytz.timezone('Asia/Kolkata')


def setup_logger(name):
    """Setup colored logger"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    ))
    
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger


def get_ist_time():
    """Get current IST time"""
    return datetime.now(IST)


def format_time_ist(dt):
    """Format datetime as IST string"""
    return dt.strftime('%I:%M:%S %p IST')


def is_premarket():
    """Check if premarket time (9:10-9:20 AM)"""
    from config import PREMARKET_START, PREMARKET_END
    now = get_ist_time().time()
    return PREMARKET_START <= now < PREMARKET_END


def is_signal_time():
    """Check if signal generation time (9:25 AM - 3:15 PM)"""
    from config import SIGNAL_START, MARKET_CLOSE
    now = get_ist_time().time()
    cutoff = time(15, 15)
    return SIGNAL_START <= now < cutoff


def is_market_open():
    """Check if market is open (9:15 AM - 3:30 PM)"""
    now = get_ist_time().time()
    return time(9, 15) <= now < time(15, 30)


def is_market_closed():
    """Check if market is closed"""
    now = get_ist_time().time()
    return now >= time(15, 30) or now < time(9, 10)


def get_market_status():
    """Get current market status"""
    now = get_ist_time()
    current_time = now.time()
    
    if current_time < time(9, 10):
        return "CLOSED", "Market opens at 9:10 AM"
    elif current_time < time(9, 20):
        return "PREMARKET", "Warmup period"
    elif current_time < time(9, 25):
        return "WARMUP", "Signal gen starts at 9:25 AM"
    elif current_time < time(15, 30):
        return "OPEN", "Market active"
    else:
        return "CLOSED", "Market closed"


def validate_price(price):
    """Validate spot/futures price"""
    if not price:
        return False
    if price <= 0 or price > 100000:
        return False
    return True


def validate_strike_data(strike_data, min_strikes=3):
    """Validate option chain data - FIXED to accept float strikes"""
    if not strike_data or not isinstance(strike_data, dict):
        return False
    if len(strike_data) < min_strikes:
        return False
    
    for strike, data in strike_data.items():
        # Accept both int and float strikes
        if not isinstance(strike, (int, float)) or not isinstance(data, dict):
            return False
        required = ['ce_oi', 'pe_oi', 'ce_vol', 'pe_vol']
        if not all(f in data for f in required):
            return False
    
    return True


def validate_candle_data(df, min_candles=10):
    """Validate futures candle data"""
    if df is None or df.empty:
        return False
    if len(df) < min_candles:
        return False
    required = ['close', 'high', 'low', 'volume']
    if not all(col in df.columns for col in required):
        return False
    return True
