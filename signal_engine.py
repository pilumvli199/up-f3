

"""
Signal Engine: Entry Signal Generation & Validation
OI-weighted signal logic
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from config import *
from utils import IST, setup_logger

logger = setup_logger("signal_engine")


# ==================== Signal Models ====================
class SignalType(Enum):
    CE_BUY = "CE_BUY"
    PE_BUY = "PE_BUY"


@dataclass
class Signal:
    """Trading signal data structure"""
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
    atr: float
    oi_5m: float
    oi_15m: float
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


# ==================== Signal Generator ====================
class SignalGenerator:
    """Generate entry signals"""
    
    def __init__(self):
        self.last_signal_time = None
    
    def generate(self, **kwargs):
        """Generate CE_BUY or PE_BUY signal"""
        
        # Try CE_BUY
        ce_signal = self._check_ce_buy(**kwargs)
        if ce_signal:
            return ce_signal
        
        # Try PE_BUY
        pe_signal = self._check_pe_buy(**kwargs)
        return pe_signal
    
    def _check_ce_buy(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                      atm_strike, atm_data, ce_total_5m, pe_total_5m, ce_total_15m, pe_total_15m,
                      atm_ce_5m, atm_pe_5m, atm_ce_15m, atm_pe_15m,
                      has_5m_total, has_15m_total, has_5m_atm, has_15m_atm,
                      volume_spike, volume_ratio, order_flow, candle_data, 
                      gamma_zone, momentum, multi_tf):
        """Check CE_BUY setup"""
        
        # Primary checks (80% weight)
        primary_ce = ce_total_15m < -OI_THRESHOLD_MEDIUM and has_15m_total
        primary_atm = atm_ce_15m < -ATM_OI_THRESHOLD and has_15m_atm
        primary_vol = volume_spike
        
        primary_passed = sum([primary_ce, primary_atm, primary_vol])
        
        if primary_passed < MIN_PRIMARY_CHECKS:
            return None
        
        # Secondary checks (20% weight)
        secondary_price = futures_price > vwap
        secondary_green = candle_data.get('color') == 'GREEN'
        
        # Bonus checks
        bonus_5m = ce_total_5m < -OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap = vwap_distance >= VWAP_BUFFER
        bonus_pcr = pcr > PCR_BULLISH
        bonus_momentum = momentum.get('consecutive_green', 0) >= 2
        bonus_flow = order_flow < 1.0
        
        bonus_passed = sum([bonus_5m, bonus_candle, bonus_vwap, bonus_pcr, 
                           bonus_momentum, bonus_flow, multi_tf, gamma_zone])
        
        # Calculate confidence
        confidence = 50
        if primary_ce: confidence += 20
        if primary_atm: confidence += 15
        if primary_vol: confidence += 10
        if secondary_green: confidence += 3
        if secondary_price: confidence += 2
        confidence += (bonus_passed * 2)
        confidence = min(confidence, 98)
        
        if confidence < MIN_CONFIDENCE:
            return None
        
        # Calculate levels
        sl_mult = ATR_SL_GAMMA_MULTIPLIER if gamma_zone else ATR_SL_MULTIPLIER
        entry = futures_price
        target = entry + int(atr * ATR_TARGET_MULTIPLIER)
        sl = entry - int(atr * sl_mult)
        
        premium = atm_data.get('ce_ltp', 150.0)
        premium_sl = premium * (1 - PREMIUM_SL_PERCENT / 100) if USE_PREMIUM_SL else 0
        
        signal = Signal(
            signal_type=SignalType.CE_BUY,
            timestamp=datetime.now(IST),
            entry_price=entry,
            target_price=target,
            stop_loss=sl,
            atm_strike=atm_strike,
            recommended_strike=atm_strike,
            option_premium=premium,
            premium_sl=premium_sl,
            vwap=vwap,
            atr=atr,
            oi_5m=ce_total_5m,
            oi_15m=ce_total_15m,
            atm_ce_change=atm_ce_15m,
            atm_pe_change=atm_pe_15m,
            pcr=pcr,
            volume_spike=volume_spike,
            volume_ratio=volume_ratio,
            order_flow=order_flow,
            confidence=confidence,
            primary_checks=primary_passed,
            bonus_checks=bonus_passed,
            trailing_sl_enabled=ENABLE_TRAILING_SL,
            is_expiry_day=gamma_zone,
            analysis_details={
                'primary': {'ce_unwinding': primary_ce, 'atm_unwinding': primary_atm, 'volume': primary_vol},
                'bonus_count': bonus_passed
            }
        )
        
        self.last_signal_time = datetime.now(IST)
        return signal
    
    def _check_pe_buy(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                      atm_strike, atm_data, ce_total_5m, pe_total_5m, ce_total_15m, pe_total_15m,
                      atm_ce_5m, atm_pe_5m, atm_ce_15m, atm_pe_15m,
                      has_5m_total, has_15m_total, has_5m_atm, has_15m_atm,
                      volume_spike, volume_ratio, order_flow, candle_data, 
                      gamma_zone, momentum, multi_tf):
        """Check PE_BUY setup"""
        
        # Primary checks
        primary_pe = pe_total_15m < -OI_THRESHOLD_MEDIUM and has_15m_total
        primary_atm = atm_pe_15m < -ATM_OI_THRESHOLD and has_15m_atm
        primary_vol = volume_spike
        
        primary_passed = sum([primary_pe, primary_atm, primary_vol])
        
        if primary_passed < MIN_PRIMARY_CHECKS:
            return None
        
        # Secondary checks
        secondary_price = futures_price < vwap
        secondary_red = candle_data.get('color') == 'RED'
        
        # Bonus checks
        bonus_5m = pe_total_5m < -OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap = vwap_distance >= VWAP_BUFFER
        bonus_pcr = pcr < PCR_BEARISH
        bonus_momentum = momentum.get('consecutive_red', 0) >= 2
        bonus_flow = order_flow > 1.5
        
        bonus_passed = sum([bonus_5m, bonus_candle, bonus_vwap, bonus_pcr,
                           bonus_momentum, bonus_flow, multi_tf, gamma_zone])
        
        # Confidence
        confidence = 50
        if primary_pe: confidence += 20
        if primary_atm: confidence += 15
        if primary_vol: confidence += 10
        if secondary_red: confidence += 3
        if secondary_price: confidence += 2
        confidence += (bonus_passed * 2)
        confidence = min(confidence, 98)
        
        if confidence < MIN_CONFIDENCE:
            return None
        
        # Levels
        sl_mult = ATR_SL_GAMMA_MULTIPLIER if gamma_zone else ATR_SL_MULTIPLIER
        entry = futures_price
        target = entry - int(atr * ATR_TARGET_MULTIPLIER)
        sl = entry + int(atr * sl_mult)
        
        premium = atm_data.get('pe_ltp', 150.0)
        premium_sl = premium * (1 - PREMIUM_SL_PERCENT / 100) if USE_PREMIUM_SL else 0
        
        signal = Signal(
            signal_type=SignalType.PE_BUY,
            timestamp=datetime.now(IST),
            entry_price=entry,
            target_price=target,
            stop_loss=sl,
            atm_strike=atm_strike,
            recommended_strike=atm_strike,
            option_premium=premium,
            premium_sl=premium_sl,
            vwap=vwap,
            atr=atr,
            oi_5m=pe_total_5m,
            oi_15m=pe_total_15m,
            atm_ce_change=atm_ce_15m,
            atm_pe_change=atm_pe_15m,
            pcr=pcr,
            volume_spike=volume_spike,
            volume_ratio=volume_ratio,
            order_flow=order_flow,
            confidence=confidence,
            primary_checks=primary_passed,
            bonus_checks=bonus_passed,
            trailing_sl_enabled=ENABLE_TRAILING_SL,
            is_expiry_day=gamma_zone,
            analysis_details={
                'primary': {'pe_unwinding': primary_pe, 'atm_unwinding': primary_atm, 'volume': primary_vol},
                'bonus_count': bonus_passed
            }
        )
        
        self.last_signal_time = datetime.now(IST)
        return signal


# ==================== Signal Validator ====================
class SignalValidator:
    """Validate and manage signal cooldown"""
    
    def __init__(self):
        self.last_signal_time = None
        self.signal_count = 0
    
    def validate(self, signal):
        """Validate signal"""
        if signal is None:
            return None
        
        # Check cooldown
        if not self._check_cooldown():
            logger.info("⏸️ Signal in cooldown")
            return None
        
        # Check R:R
        rr = signal.get_rr_ratio()
        if rr < 1.0:
            logger.warning(f"⚠️ Poor R:R: {rr:.2f}")
            return None
        
        # Check confidence
        if signal.confidence < MIN_CONFIDENCE:
            logger.warning(f"⚠️ Low confidence: {signal.confidence}%")
            return None
        
        self.last_signal_time = datetime.now(IST)
        self.signal_count += 1
        
        return signal
    
    def _check_cooldown(self):
        """Check cooldown period"""
        if self.last_signal_time is None:
            return True
        
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return elapsed >= SIGNAL_COOLDOWN_SECONDS
    
    def get_cooldown_remaining(self):
        """Get seconds until next signal"""
        if self.last_signal_time is None:
            return 0
        
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return max(0, int(SIGNAL_COOLDOWN_SECONDS - elapsed))

