"""
Signal Engine: Entry Signal Generation & Validation
FIXED: VWAP validation, better confidence, re-entry protection, symmetric CE/PE logic
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
    """Generate CE_BUY or PE_BUY signals with improved checks"""

    def __init__(self):
        self.last_signal_time = None
        self.last_signal_type = None
        self.last_signal_strike = None

    def generate(self, **kwargs):
        """Try CE_BUY then PE_BUY"""
        ce_signal = self._check_ce_buy(**kwargs)
        if ce_signal:
            return ce_signal
        pe_signal = self._check_pe_buy(**kwargs)
        return pe_signal

    # Shared helpers
    def _basic_exhaustion_block(self, futures_price, vwap, atr):
        """Block entries that are far from VWAP (exhaustion)"""
        if vwap is None or atr is None:
            return False, None
        price_from_vwap = abs(futures_price - vwap)
        if price_from_vwap > (atr * 2.5):
            reason = f"EXHAUSTION (Price {price_from_vwap:.0f} pts from VWAP, limit {atr*2.5:.0f})"
            return True, reason
        return False, None

    def _compose_signal(self, signal_type: SignalType, futures_price: float, vwap: float, vwap_distance: float,
                        vwap_score: int, atr: float, atm_strike: int, atm_data: dict,
                        pcr: float, oi_5m: float, oi_15m: float, oi_strength: str,
                        atm_ce_15m: float, atm_pe_15m: float,
                        volume_spike: bool, volume_ratio: float, order_flow: float,
                        confidence: int, primary_passed: int, bonus_passed: int,
                        gamma_zone: bool, momentum: dict):
        """Create Signal dataclass instance"""
        # Calculate levels
        sl_mult = ATR_SL_GAMMA_MULTIPLIER if gamma_zone else ATR_SL_MULTIPLIER
        entry = futures_price
        if signal_type == SignalType.CE_BUY:
            target = entry + int(atr * ATR_TARGET_MULTIPLIER)
            sl = entry - int(atr * sl_mult)
            premium = float(atm_data.get('ce_ltp', 150.0))
        else:
            target = entry - int(atr * ATR_TARGET_MULTIPLIER)
            sl = entry + int(atr * sl_mult)
            premium = float(atm_data.get('pe_ltp', 150.0))

        premium_sl = premium * (1 - PREMIUM_SL_PERCENT / 100) if USE_PREMIUM_SL else 0

        signal = Signal(
            signal_type=signal_type,
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
            oi_5m=oi_5m,
            oi_15m=oi_15m,
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
                'primary_checks': primary_passed,
                'bonus_checks': bonus_passed,
                'vwap_reason_score': vwap_score
            }
        )
        # Update last signal info
        self.last_signal_time = datetime.now(IST)
        self.last_signal_type = signal_type
        self.last_signal_strike = atm_strike
        return signal

    def _check_common_blockers(self, ce_total_5m, pe_total_5m, has_5m_total, has_15m_total):
        """Return True if conflicting OI or other absolute blockers exist"""
        # Conflicting OI: both building heavily -> block signals (choppy)
        if ce_total_5m > 0 and pe_total_5m > 0:
            if abs(ce_total_5m) > 10 and abs(pe_total_5m) > 10:
                return True, f"CONFLICTING OI (Both CE +{ce_total_5m:.1f}% and PE +{pe_total_5m:.1f}% building)"
        # Require some baseline data
        if not has_5m_total and not has_15m_total:
            return True, "Insufficient OI history"
        return False, None

    def _check_ce_buy(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                      atm_strike, atm_data, ce_total_5m, pe_total_5m, ce_total_15m, pe_total_15m,
                      atm_ce_5m, atm_pe_5m, atm_ce_15m, atm_pe_15m,
                      has_5m_total, has_15m_total, has_5m_atm, has_15m_atm,
                      volume_spike, volume_ratio, order_flow, candle_data,
                      gamma_zone, momentum, multi_tf, oi_strength='weak'):
        """Check CE_BUY setup with VWAP validation and strict ATM requirements"""

        # Exhaustion block
        blocked, reason = self._basic_exhaustion_block(futures_price, vwap, atr)
        if blocked:
            logger.debug(f"  ❌ CE_BUY blocked: {reason}")
            return None

        # VWAP validation
        vwap_valid, vwap_reason, vwap_score = TechnicalAnalyzer.validate_signal_with_vwap(
            "CE_BUY", futures_price, vwap, atr
        )
        if not vwap_valid:
            logger.debug(f"  ❌ CE_BUY rejected: {vwap_reason}")
            return None

        # Common blockers
        blocked, blocker_reason = self._check_common_blockers(ce_total_5m, pe_total_5m, has_5m_total, has_15m_total)
        if blocked:
            logger.debug(f"  ❌ CE_BUY blocked: {blocker_reason}")
            return None

        # Primary checks: for CE_BUY we want PE unwinding (puts sold)
        primary_pe_unwind = (pe_total_15m < -MIN_OI_15M_FOR_ENTRY and pe_total_5m < -MIN_OI_5M_FOR_ENTRY and has_15m_total and has_5m_total)
        # Strict ATM: require ATM 15m unwinding for safety
        if has_15m_atm:
            primary_atm = atm_pe_15m < -ATM_OI_THRESHOLD
        else:
            logger.debug("  ❌ CE_BUY blocked: No ATM 15m data")
            return None

        primary_vol = bool(volume_spike)

        checks_to_evaluate = [primary_pe_unwind, primary_atm, primary_vol]
        primary_passed = sum(bool(x) for x in checks_to_evaluate)

        if primary_passed < MIN_PRIMARY_CHECKS:
            logger.debug(f"  ❌ CE_BUY: Only {primary_passed}/{MIN_PRIMARY_CHECKS} primary checks")
            return None

        # Secondary checks
        secondary_price = futures_price > vwap if (vwap is not None) else False
        secondary_green = candle_data.get('color') == 'GREEN'

        # Bonus checks
        bonus_5m_strong = ce_total_5m < -STRONG_OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap_above = vwap_distance > 0
        bonus_pcr = pcr > PCR_BULLISH
        bonus_momentum = momentum.get('consecutive_green', 0) >= 2
        bonus_flow = order_flow < 1.0
        bonus_vol_strong = volume_ratio >= VOL_SPIKE_STRONG

        bonus_passed = sum(bool(x) for x in [
            bonus_5m_strong, bonus_candle, bonus_vwap_above, bonus_pcr,
            bonus_momentum, bonus_flow, multi_tf, gamma_zone, bonus_vol_strong
        ])

        # Confidence calc (bounded)
        confidence = 40
        if primary_pe_unwind:
            confidence += 25 if oi_strength == 'strong' else 20
        if primary_atm:
            confidence += 20
        if primary_vol:
            confidence += 15
        confidence += int(vwap_score / 5)
        if secondary_green:
            confidence += 5
        if secondary_price:
            confidence += 5
        confidence += min(bonus_passed * 2, 15)
        confidence = min(confidence, 98)

        if confidence < MIN_CONFIDENCE:
            logger.debug(f"  ❌ CE_BUY: Confidence {confidence}% < {MIN_CONFIDENCE}%")
            return None

        # Build signal
        return self._compose_signal(
            SignalType.CE_BUY, futures_price, vwap, vwap_distance, vwap_score, atr,
            atm_strike, atm_data, pcr, ce_total_5m, ce_total_15m, oi_strength,
            atm_ce_15m, atm_pe_15m, volume_spike, volume_ratio, order_flow,
            confidence, primary_passed, bonus_passed, gamma_zone, momentum
        )

    def _check_pe_buy(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                      atm_strike, atm_data, ce_total_5m, pe_total_5m, ce_total_15m, pe_total_15m,
                      atm_ce_5m, atm_pe_5m, atm_ce_15m, atm_pe_15m,
                      has_5m_total, has_15m_total, has_5m_atm, has_15m_atm,
                      volume_spike, volume_ratio, order_flow, candle_data,
                      gamma_zone, momentum, multi_tf, oi_strength='weak'):
        """Mirror of CE_BUY but for PE_BUY"""

        # Exhaustion block
        blocked, reason = self._basic_exhaustion_block(futures_price, vwap, atr)
        if blocked:
            logger.debug(f"  ❌ PE_BUY blocked: {reason}")
            return None

        # VWAP validation
        vwap_valid, vwap_reason, vwap_score = TechnicalAnalyzer.validate_signal_with_vwap(
            "PE_BUY", futures_price, vwap, atr
        )
        if not vwap_valid:
            logger.debug(f"  ❌ PE_BUY rejected: {vwap_reason}")
            return None

        # Common blockers
        blocked, blocker_reason = self._check_common_blockers(ce_total_5m, pe_total_5m, has_5m_total, has_15m_total)
        if blocked:
            logger.debug(f"  ❌ PE_BUY blocked: {blocker_reason}")
            return None

        # Primary checks for PE_BUY: CE unwinding (calls being sold)
        primary_ce_unwind = (ce_total_15m < -MIN_OI_15M_FOR_ENTRY and ce_total_5m < -MIN_OI_5M_FOR_ENTRY and has_15m_total and has_5m_total)
        # ATM strict
        if has_15m_atm:
            primary_atm = atm_ce_15m < -ATM_OI_THRESHOLD
        else:
            logger.debug("  ❌ PE_BUY blocked: No ATM 15m data")
            return None

        primary_vol = bool(volume_spike)

        checks_to_evaluate = [primary_ce_unwind, primary_atm, primary_vol]
        primary_passed = sum(bool(x) for x in checks_to_evaluate)

        if primary_passed < MIN_PRIMARY_CHECKS:
            logger.debug(f"  ❌ PE_BUY: Only {primary_passed}/{MIN_PRIMARY_CHECKS} primary checks")
            return None

        # Secondary checks
        secondary_price = futures_price < vwap if (vwap is not None) else False
        secondary_red = candle_data.get('color') == 'RED'

        # Bonus checks
        bonus_5m_strong = pe_total_5m < -STRONG_OI_5M_THRESHOLD and has_5m_total
        bonus_candle = candle_data.get('size', 0) >= MIN_CANDLE_SIZE
        bonus_vwap_below = vwap_distance < 0
        bonus_pcr = pcr < PCR_BEARISH
        bonus_momentum = momentum.get('consecutive_red', 0) >= 2
        bonus_flow = order_flow > 1.5
        bonus_vol_strong = volume_ratio >= VOL_SPIKE_STRONG

        bonus_passed = sum(bool(x) for x in [
            bonus_5m_strong, bonus_candle, bonus_vwap_below, bonus_pcr,
            bonus_momentum, bonus_flow, multi_tf, gamma_zone, bonus_vol_strong
        ])

        # Confidence calc
        confidence = 40
        if primary_ce_unwind:
            confidence += 25 if oi_strength == 'strong' else 20
        if primary_atm:
            confidence += 20
        if primary_vol:
            confidence += 15
        confidence += int(vwap_score / 5)
        if secondary_red:
            confidence += 5
        if secondary_price:
            confidence += 5
        confidence += min(bonus_passed * 2, 15)
        confidence = min(confidence, 98)

        if confidence < MIN_CONFIDENCE:
            logger.debug(f"  ❌ PE_BUY: Confidence {confidence}% < {MIN_CONFIDENCE}%")
            return None

        # Build signal
        return self._compose_signal(
            SignalType.PE_BUY, futures_price, vwap, vwap_distance, vwap_score, atr,
            atm_strike, atm_data, pcr, pe_total_5m, pe_total_15m, oi_strength,
            atm_ce_15m, atm_pe_15m, volume_spike, volume_ratio, order_flow,
            confidence, primary_passed, bonus_passed, gamma_zone, momentum
        )


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

    def validate(self, signal: Optional[Signal]) -> Optional[Signal]:
        """Validate signal with duplicate/re-entry/cooldown checks"""
        if signal is None:
            return None

        # Basic cooldown
        if not self._check_cooldown():
            logger.info("⏸️ Signal in cooldown")
            return None

        # Duplicate signal (same dir+strike within 10 minutes)
        if self._is_duplicate_signal(signal):
            logger.info("⚠️ Duplicate signal ignored (same direction+strike in last 10min)")
            return None

        # Same strike re-entry protection
        if self._is_same_strike_too_soon(signal):
            logger.info(f"⚠️ Same strike {signal.atm_strike} re-entry blocked (need {SAME_STRIKE_COOLDOWN_MINUTES}min gap)")
            return None

        # Opposite signal too soon after exit
        if self._is_opposite_too_soon(signal):
            logger.info(f"⚠️ Opposite signal too soon after exit (need {OPPOSITE_SIGNAL_COOLDOWN_MINUTES}min gap)")
            return None

        # R:R validation
        rr = signal.get_rr_ratio()
        if rr < 1.0:
            logger.warning(f"⚠️ Poor R:R: {rr:.2f}")
            return None

        # Confidence validation (extra safety)
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
        self.recent_signals = self.recent_signals[-10:]

        self.last_signal_time = datetime.now(IST)
        self.signal_count += 1

        logger.info(f"✅ Signal validated: {signal.signal_type.value} @ {signal.atm_strike} ({signal.confidence}%)")
        return signal

    def record_exit(self, signal_type: SignalType, strike: int):
        """Record exit for re-entry protections"""
        self.last_exit_time = datetime.now(IST)
        self.last_exit_type = signal_type
        self.last_exit_strike = strike
        logger.debug(f"📝 Exit recorded: {signal_type.value} @ {strike}")

    def _check_cooldown(self) -> bool:
        if self.last_signal_time is None:
            return True
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return elapsed >= SIGNAL_COOLDOWN_SECONDS

    def _is_duplicate_signal(self, signal: Signal) -> bool:
        cutoff = datetime.now(IST) - timedelta(minutes=10)
        for old in self.recent_signals:
            if (old['type'] == signal.signal_type and
                    old['strike'] == signal.atm_strike and
                    old['time'] > cutoff):
                return True
        return False

    def _is_same_strike_too_soon(self, signal: Signal) -> bool:
        if not self.last_exit_time or not self.last_exit_strike:
            return False
        elapsed_minutes = (datetime.now(IST) - self.last_exit_time).total_seconds() / 60
        if (signal.atm_strike == self.last_exit_strike and
                elapsed_minutes < SAME_STRIKE_COOLDOWN_MINUTES):
            return True
        return False

    def _is_opposite_too_soon(self, signal: Signal) -> bool:
        if not self.last_exit_time or not self.last_exit_type:
            return False
        elapsed_minutes = (datetime.now(IST) - self.last_exit_time).total_seconds() / 60
        opposite = (
            (self.last_exit_type == SignalType.CE_BUY and signal.signal_type == SignalType.PE_BUY) or
            (self.last_exit_type == SignalType.PE_BUY and signal.signal_type == SignalType.CE_BUY)
        )
        if opposite and elapsed_minutes < OPPOSITE_SIGNAL_COOLDOWN_MINUTES:
            return True
        return False

    def get_cooldown_remaining(self) -> int:
        if self.last_signal_time is None:
            return 0
        elapsed = (datetime.now(IST) - self.last_signal_time).total_seconds()
        return max(0, int(SIGNAL_COOLDOWN_SECONDS - elapsed))
