"""
Signal Engine (defensive fixes)
- make VWAP/ATR checks robust (avoid None math)
- safe vwap_score usage
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from config import *
from utils import IST, setup_logger
from analyzers import TechnicalAnalyzer

logger = setup_logger("signal_engine")

class SignalType(Enum):
    CE_BUY = "CE_BUY"
    PE_BUY = "PE_BUY"

@dataclass
class Signal:
    # ... same fields as before (kept unchanged for compatibility) ...
    signal_type: SignalType
    timestamp: datetime
    entry_price: float
    target_price: float
    stop_loss: float
    atm_strike: int
    recommended_strike: int
    option_premium: float
    premium_sl: float
    vwap: float
    vwap_distance: float
    vwap_score: int
    atr: float
    oi_5m: float
    oi_15m: float
    oi_strength: str
    atm_ce_change: float
    atm_pe_change: float
    pcr: float
    volume_spike: bool
    volume_ratio: float
    order_flow: float
    confidence: int
    primary_checks: int
    bonus_checks: int
    trailing_sl_enabled: bool
    is_expiry_day: bool
    analysis_details: dict

    def get_direction(self):
        return "BULLISH" if self.signal_type == SignalType.CE_BUY else "BEARISH"

    def get_rr_ratio(self):
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

class SignalGenerator:
    def __init__(self):
        self.last_signal_time = None
        self.last_signal_type = None
        self.last_signal_strike = None

    def generate(self, **kwargs):
        ce_signal = self._check_ce_buy(**kwargs)
        if ce_signal:
            return ce_signal
        pe_signal = self._check_pe_buy(**kwargs)
        return pe_signal

    # ... _check_ce_buy and _check_pe_buy largely same as before but with guards ...
    # For brevity: keep logic but ensure vwap/atr presence check at top:

    # Example snippet showing defensive top checks (apply same for CE and PE checks in your file)
    def _pre_checks(self, futures_price, vwap, atr):
        # vwap and atr required for most safe signals
        if vwap is None:
            logger.debug("  ❌ Blocked: VWAP missing")
            return False
        if atr is None or atr <= 0:
            logger.debug("  ❌ Blocked: ATR missing or invalid")
            return False
        return True

    # You should place the call `if not self._pre_checks(...): return None` near top of CE/PE checks.
