"""
Utilities: Time, Logging, Validators
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
    """Check if premarket time (9:10-9:15 AM)"""
    from config import PREMARKET_START, PREMARKET_END
    now = get_ist_time().time()
    return PREMARKET_START <= now < PREMARKET_END

def is_first_data_time():
    """Check if it's time for first data collection (9:16 AM)"""
    from config import FIRST_DATA_TIME
    now = get_ist_time().time()
    return now >= FIRST_DATA_TIME and now < time(9, 17)

def is_signal_time(warmup_complete=False):
    """Check if signal generation time WITH warmup validation"""
    from config import SIGNAL_START
    now = get_ist_time().time()
    cutoff = time(15, 15)
    
    # Time window check
    if not (SIGNAL_START <= now < cutoff):
        return False, "Outside signal window (9:21-3:15 PM)"
    
    # Warmup check
    if not warmup_complete:
        return False, "Warmup in progress"
    
    return True, "Ready for signals"

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
    elif current_time < time(9, 15):
        return "PREMARKET", "Warmup period"
    elif current_time < time(9, 16):
        return "MARKET_OPEN", "Waiting for 9:16 first snapshot"
    elif current_time < time(9, 21):
        return "WARMUP", "Collecting baseline data"
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

def validate_strike_data(strike_data, min_strikes=7):
    """
    Validate option chain data - need at least 7 strikes for safety
    """
    if not strike_data or not isinstance(strike_data, dict):
        return False
    
    if len(strike_data) < min_strikes:
        return False
    
    # Check structure
    for strike, data in strike_data.items():
        # Accept int or float strikes
        if not isinstance(strike, (int, float)):
            return False
        
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        required = ['ce_oi', 'pe_oi', 'ce_vol', 'pe_vol']
        if not all(f in data for f in required):
            return False
        
        # Check if values are numeric
        for field in required:
            if not isinstance(data[field], (int, float)):
                return False
    
    # Check if at least some OI exists
    total_oi = sum(d['ce_oi'] + d['pe_oi'] for d in strike_data.values())
    if total_oi == 0:
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
