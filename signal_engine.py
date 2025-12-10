"""
Signal Engine: Entry Signal Generation & Validation
FIXED: VWAP validation, better confidence, re-entry protection
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from config import *
from utils import IST, setup_logger
from analyzers import TechnicalAnalyzer

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


# ==================== Signal Generator ====================
class SignalGenerator:
    """Generate entry signals with FIXED logic"""
    
    def __init__(self):
        self.last_signal_time = None
        self.last_signal_type = None
        self.last_signal_strike = None
    
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
                      gamma_zone, momentum, multi_tf, oi_strength='weak'):
        """Check CE_BUY setup with VWAP validation"""
        
        # STEP 1: VWAP Validation (BLOCKING CHECK)
        vwap_valid, vwap_reason, vwap_score = TechnicalAnalyzer.validate_signal_with_vwap(
            "CE_BUY", futures_price, vwap, atr
        )
        
        if not vwap_valid:
            logger.debug(f"  ❌ CE_BUY rejected: {vwap_reason}")
            return None
        
        logger.debug(f"  ✅ VWAP check passed: {vwap_reason} (Score: {vwap_score})")
        
        # STEP 2: Primary checks (STRICTER - both TF required)
        primary_ce = ce_total_15m < -MIN_OI_15M_FOR_ENTRY and ce_total_5m < -MIN_OI_5M_FOR_ENTRY and has_15m_total and has_5m_total
        
        # ATM check - ONLY if we have 15m data (not blocking on ATM shifts)
        primary_atm = False
        atm_skipped = False
        
        if has_15m_atm and atm_ce_15m < -ATM_OI_THRESHOLD:
            primary_atm = True
        elif not has_15m_atm:
            logger.debug(f"  ⚠️ ATM check skipped (no 15m data after ATM shift)")
            atm_skipped = True  # Track that we skipped it
            # Don't count as passed OR failed - neutral
        
        primary_vol = volume_spike
        
        # STRICT: Always require 3 checks, no relaxation
        # If ATM skipped, only CE+Vol can pass = 2/3 (BLOCKED!)
        checks_to_evaluate = [primary_ce, primary_atm, primary_vol]
        primary_passed = sum(checks_to_evaluate)
        
        if primary_passed < MIN_PRIMARY_CHECKS:
            logger.debug(f"  ❌ CE_BUY: Only {primary_passed}/{MIN_PRIMARY_CHECKS} primary checks")
            return None
        
        # STEP 3: Secondary checks
        secondary_price = futures_price > vwap
        secondary_green = candle_data.get('color') == 'GREEN'
        
        # STEP 4: Bonus checks
        bonus_5m_strong = ce_total_5m < -STRONG_OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap_above = vwap_distance > 0
        bonus_pcr = pcr > PCR_BULLISH
        bonus_momentum = momentum.get('consecutive_green', 0) >= 2
        bonus_flow = order_flow < 1.0
        bonus_vol_strong = volume_ratio >= VOL_SPIKE_STRONG
        
        bonus_passed = sum([bonus_5m_strong, bonus_candle, bonus_vwap_above, bonus_pcr, 
                           bonus_momentum, bonus_flow, multi_tf, gamma_zone, bonus_vol_strong])
        
        # STEP 5: Calculate confidence (IMPROVED)
        confidence = 40  # Base
        
        # Primary checks (60 points max)
        if primary_ce: 
            if oi_strength == 'strong':
                confidence += 25
            else:
                confidence += 20
        # ATM: Only add points if actually checked AND passed
        if primary_atm and not atm_skipped:
            confidence += 20
        if primary_vol: confidence += 15
        
        # VWAP score (20 points max)
        confidence += int(vwap_score / 5)
        
        # Secondary checks (10 points)
        if secondary_green: confidence += 5
        if secondary_price: confidence += 5
        
        # Bonus checks (each 1-2 points)
        confidence += min(bonus_passed * 2, 15)
        
        confidence = min(confidence, 98)
        
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"  ❌ CE_BUY: Confidence {confidence}% < {MIN_CONFIDENCE}%")
            return None
        
        # STEP 6: Calculate levels
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
            vwap_distance=vwap_distance,
            vwap_score=vwap_score,
            atr=atr,
            oi_5m=ce_total_5m,
            oi_15m=ce_total_15m,
            oi_strength=oi_strength,
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
                'vwap_reason': vwap_reason,
                'bonus_count': bonus_passed
            }
        )
        
        self.last_signal_time = datetime.now(IST)
        self.last_signal_type = SignalType.CE_BUY
        self.last_signal_strike = atm_strike
        
        return signal
    
    def _check_pe_buy(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                      atm_strike, atm_data, ce_total_5m, pe_total_5m, ce_total_15m, pe_total_15m,
                      atm_ce_5m, atm_pe_5m, atm_ce_15m, atm_pe_15m,
                      has_5m_total, has_15m_total, has_5m_atm, has_15m_atm,
                      volume_spike, volume_ratio, order_flow, candle_data, 
                      gamma_zone, momentum, multi_tf, oi_strength='weak'):
        """Check PE_BUY setup with VWAP validation"""
        
        # STEP 1: VWAP Validation (BLOCKING CHECK)
        vwap_valid, vwap_reason, vwap_score = TechnicalAnalyzer.validate_signal_with_vwap(
            "PE_BUY", futures_price, vwap, atr
        )
        
        if not vwap_valid:
            logger.debug(f"  ❌ PE_BUY rejected: {vwap_reason}")
            return None
        
        logger.debug(f"  ✅ VWAP check passed: {vwap_reason} (Score: {vwap_score})")
        
        # STEP 2: Primary checks (STRICTER - both TF required)
        primary_pe = pe_total_15m < -MIN_OI_15M_FOR_ENTRY and pe_total_5m < -MIN_OI_5M_FOR_ENTRY and has_15m_total and has_5m_total
        
        # ATM check - ONLY if we have 15m data (not blocking on ATM shifts)
        primary_atm = False
        atm_skipped = False
        
        if has_15m_atm and atm_pe_15m < -ATM_OI_THRESHOLD:
            primary_atm = True
        elif not has_15m_atm:
            logger.debug(f"  ⚠️ ATM check skipped (no 15m data after ATM shift)")
            atm_skipped = True  # Track that we skipped it
            # Don't count as passed OR failed - neutral
        
        primary_vol = volume_spike
        
        # STRICT: Always require 3 checks, no relaxation
        # If ATM skipped, only PE+Vol can pass = 2/3 (BLOCKED!)
        checks_to_evaluate = [primary_pe, primary_atm, primary_vol]
        primary_passed = sum(checks_to_evaluate)
        
        if primary_passed < MIN_PRIMARY_CHECKS:
            logger.debug(f"  ❌ PE_BUY: Only {primary_passed}/{MIN_PRIMARY_CHECKS} primary checks")
            return None
        
        # STEP 3: Secondary checks
        secondary_price = futures_price < vwap
        secondary_red = candle_data.get('color') == 'RED'
        
        # STEP 4: Bonus checks
        bonus_5m_strong = pe_total_5m < -STRONG_OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap_below = vwap_distance < 0
        bonus_pcr = pcr < PCR_BEARISH
        bonus_momentum = momentum.get('consecutive_red', 0) >= 2
        bonus_flow = order_flow > 1.5
        bonus_vol_strong = volume_ratio >= VOL_SPIKE_STRONG
        
        bonus_passed = sum([bonus_5m_strong, bonus_candle, bonus_vwap_below, bonus_pcr,
                           bonus_momentum, bonus_flow, multi_tf, gamma_zone, bonus_vol_strong])
        
        # STEP 5: Calculate confidence (IMPROVED)
        confidence = 40  # Base
        
        # Primary checks (60 points max)
        if primary_pe:
            if oi_strength == 'strong':
                confidence += 25
            else:
                confidence += 20
        # ATM: Only add points if actually checked AND passed
        if primary_atm and not atm_skipped:
            confidence += 20
        if primary_vol: confidence += 15
        
        # VWAP score (20 points max)
        confidence += int(vwap_score / 5)
        
        # Secondary checks (10 points)
        if secondary_red: confidence += 5
        if secondary_price: confidence += 5
        
        # Bonus checks
        confidence += min(bonus_passed * 2, 15)
        
        confidence = min(confidence, 98)
        
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"  ❌ PE_BUY: Confidence {confidence}% < {MIN_CONFIDENCE}%")
            return None
        
        # STEP 6: Calculate levels
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
            vwap_distance=vwap_distance,
            vwap_score=vwap_score,
            atr=atr,
            oi_5m=pe_total_5m,
            oi_15m=pe_total_15m,
            oi_strength=oi_strength,
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
                'vwap_reason': vwap_reason,
                'bonus_count': bonus_passed
            }
        )
        
        self.last_signal_time = datetime.now(IST)
        self.last_signal_type = SignalType.PE_BUY
        self.last_signal_strike = atm_strike
        
        return signal


# ==================== Signal Validator ====================
class SignalValidator:
    """Validate and manage signal cooldown with re-entry protection"""
    
    def __init__(self):
        self.last_signal_time = None
        self.signal_count = 0
        self.recent_signals = []
        self.last_exit_time = None
        self.last_exit_type = None
        self.last_exit_strike = None
    
    def validate(self, signal):
        """Validate signal with enhanced duplicate/re-entry checks"""
        if signal is None:
            return None
        
        # Check 1: Basic cooldown
        if not self._check_cooldown():
            logger.info("⏸️ Signal in cooldown")
            return None
        
        # Check 2: Duplicate signal (same direction + strike in 10 min)
        if self._is_duplicate_signal(signal):
            logger.info("⚠️ Duplicate signal ignored (same direction+strike in last 10min)")
            return None
        
        # Check 3: Same strike re-entry protection
        if self._is_same_strike_too_soon(signal):
            logger.info(f"⚠️ Same strike {signal.atm_strike} re-entry blocked (need {SAME_STRIKE_COOLDOWN_MINUTES}min gap)")
            return None
        
        # Check 4: Opposite signal after exit protection
        if self._is_opposite_too_soon(signal):
            logger.info(f"⚠️ Opposite signal too soon after exit (need {OPPOSITE_SIGNAL_COOLDOWN_MINUTES}min gap)")
            return None
        
        # Check 5: R:R validation
        rr = signal.get_rr_ratio()
        if rr < 1.0:
            logger.warning(f"⚠️ Poor R:R: {rr:.2f}")
            return None
        
        # Check 6: Confidence validation
        if signal.confidence < MIN_CONFIDENCE:
            logger.warning(f"⚠️ Low confidence: {signal.confidence}%")
            return None
        
        # Track signal
        self.recent_signals.append({
            'type': signal.signal_type,
            'strike': signal.atm_strike,
            'time': signal.timestamp,
            'confidence': signal.confidence
        })
        
        # Keep only last 10
        self.recent_signals = self.recent_signals[-10:]
        
        self.last_signal_time = datetime.now(IST)
        self.signal_count += 1
        
        logger.info(f"✅ Signal validated: {signal.signal_type.value} @ {signal.atm_strike} ({signal.confidence}%)")
        
        return signal
    
    def record_exit(self, signal_type, strike):
        """Record exit for re-entry protection"""
        self.last_exit_time = datetime.now(IST)
        self.last_exit_type = signal_type
        self.last_exit_strike = strike
        logger.debug(f"📝 Exit recorded: {signal_type.value} @ {strike}")
    
    def _check_cooldown(self):
        """Check basic cooldown period"""
        if self.last_signal_time is None:
            return True
        
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return elapsed >= SIGNAL_COOLDOWN_SECONDS
    
    def _is_duplicate_signal(self, signal):
        """Check if same signal in last 10 minutes"""
        cutoff = datetime.now(IST) - timedelta(minutes=10)
        
        for old in self.recent_signals:
            if (old['type'] == signal.signal_type and 
                old['strike'] == signal.atm_strike and
                old['time'] > cutoff):
                return True
        
        return False
    
    def _is_same_strike_too_soon(self, signal):
        """Check if same strike re-entry too soon"""
        if not self.last_exit_time or not self.last_exit_strike:
            return False
        
        elapsed_minutes = (datetime.now(IST) - self.last_exit_time).total_seconds() / 60
        
        if (signal.atm_strike == self.last_exit_strike and 
            elapsed_minutes < SAME_STRIKE_COOLDOWN_MINUTES):
            return True
        
        return False
    
    def _is_opposite_too_soon(self, signal):
        """Check if opposite signal too soon after exit"""
        if not self.last_exit_time or not self.last_exit_type:
            return False
        
        elapsed_minutes = (datetime.now(IST) - self.last_exit_time).total_seconds() / 60
        
        # Check if opposite direction
        opposite = (
            (self.last_exit_type == SignalType.CE_BUY and signal.signal_type == SignalType.PE_BUY) or
            (self.last_exit_type == SignalType.PE_BUY and signal.signal_type == SignalType.CE_BUY)
        )
        
        if opposite and elapsed_minutes < OPPOSITE_SIGNAL_COOLDOWN_MINUTES:
            return True
        
        return False
    
    def get_cooldown_remaining(self):
        """Get seconds until next signal"""
        if self.last_signal_time is None:
            return 0
        
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return max(0, int(SIGNAL_COOLDOWN_SECONDS - elapsed))
