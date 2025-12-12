"""
Market Analyzers V2 (FIXED)
- OIAnalyzer, VolumeAnalyzer, TechnicalAnalyzer, MarketAnalyzer
- Robust handling of edge cases and clearer thresholds / scoring
"""

import pandas as pd
from datetime import datetime
from config import *
from utils import IST, setup_logger

logger = setup_logger("analyzers")


# ==================== OI Analyzer ====================
class OIAnalyzer:
    """Open Interest analysis with deep focus"""

    @staticmethod
    def calculate_total_oi(strike_data):
        if not strike_data:
            return 0, 0
        total_ce = sum(d.get('ce_oi', 0) for d in strike_data.values())
        total_pe = sum(d.get('pe_oi', 0) for d in strike_data.values())
        return total_ce, total_pe

    @staticmethod
    def calculate_deep_analysis_oi(strike_data, atm_strike):
        deep_strikes = get_deep_analysis_strikes(atm_strike)
        deep_ce = 0
        deep_pe = 0
        for strike in deep_strikes:
            if strike in strike_data:
                deep_ce += strike_data[strike].get('ce_oi', 0)
                deep_pe += strike_data[strike].get('pe_oi', 0)
        return deep_ce, deep_pe, deep_strikes

    @staticmethod
    def calculate_pcr(total_pe, total_ce):
        # Handle divide-by-zero safely and cap PCR to reasonable range
        if total_ce == 0:
            return round(min(10.0, float(total_pe or 10.0)), 2)
        pcr = total_pe / total_ce
        return round(min(max(pcr, 0.0), 10.0), 2)

    @staticmethod
    def detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m):
        """
        Decide unwinding/building strength.
        Inputs are percent changes (can be negative for unwinding).
        Returns dict with booleans and strength strings.
        """

        def strength_from_vals(short_tf, long_tf):
            # prefer long_tf magnitude for strength
            if long_tf <= -STRONG_OI_15M_THRESHOLD and short_tf <= -STRONG_OI_5M_THRESHOLD:
                return 'strong'
            if long_tf <= -MIN_OI_15M_FOR_ENTRY and short_tf <= -MIN_OI_5M_FOR_ENTRY:
                return 'medium'
            return 'weak'

        ce_strength = strength_from_vals(ce_5m, ce_15m)
        pe_strength = strength_from_vals(pe_5m, pe_15m)

        ce_unwinding = (ce_5m < -MIN_OI_5M_FOR_ENTRY and ce_15m < -MIN_OI_15M_FOR_ENTRY)
        pe_unwinding = (pe_5m < -MIN_OI_5M_FOR_ENTRY and pe_15m < -MIN_OI_15M_FOR_ENTRY)

        # multi-timeframe signal if both timeframes show consistent move
        multi_tf = (abs(ce_5m) >= MIN_OI_5M_FOR_ENTRY and abs(ce_15m) >= MIN_OI_15M_FOR_ENTRY) or \
                   (abs(pe_5m) >= MIN_OI_5M_FOR_ENTRY and abs(pe_15m) >= MIN_OI_15M_FOR_ENTRY)

        return {
            'ce_unwinding': ce_unwinding,
            'pe_unwinding': pe_unwinding,
            'ce_strength': ce_strength,
            'pe_strength': pe_strength,
            'multi_timeframe': multi_tf
        }

    @staticmethod
    def detect_conflicting_oi(ce_5m, ce_15m, pe_5m, pe_15m):
        """
        Detect conflicting OI where both CE & PE building strongly (choppy market).
        Returns (bool, message)
        """
        BUILDING_THRESHOLD = 3.0  # percent building considered meaningful

        ce_building = (ce_5m > BUILDING_THRESHOLD) or (ce_15m > BUILDING_THRESHOLD)
        pe_building = (pe_5m > BUILDING_THRESHOLD) or (pe_15m > BUILDING_THRESHOLD)

        # both building strongly -> conflict
        if ce_building and pe_building:
            msg = f"BOTH CE (+{ce_5m:.1f}% / {ce_15m:.1f}%) & PE (+{pe_5m:.1f}% / {pe_15m:.1f}%) building = CHOPPY"
            return True, msg

        # cross-signals: short-term building while long-term unwinding -> confusion
        if ce_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m > BUILDING_THRESHOLD:
            return True, f"CE unwinding 15m ({ce_15m:.1f}%) BUT PE building 5m (+{pe_5m:.1f}%) = CONFUSED"
        if pe_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m > BUILDING_THRESHOLD:
            return True, f"PE unwinding 15m ({pe_15m:.1f}%) BUT CE building 5m (+{ce_5m:.1f}%) = CONFUSED"

        return False, "OI aligned"

    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        return strike_data.get(atm_strike, {
            'ce_oi': 0, 'pe_oi': 0, 'ce_vol': 0, 'pe_vol': 0, 'ce_ltp': 0, 'pe_ltp': 0
        })

    @staticmethod
    def get_atm_oi_changes(strike_data, atm_strike, previous_strike_data=None):
        current = strike_data.get(atm_strike, {
            'ce_oi': 0, 'pe_oi': 0, 'ce_vol': 0, 'pe_vol': 0, 'ce_ltp': 0, 'pe_ltp': 0
        })
        ce_change_pct = 0.0
        pe_change_pct = 0.0
        has_prev = previous_strike_data is not None and atm_strike in previous_strike_data

        if previous_strike_data and atm_strike in previous_strike_data:
            prev = previous_strike_data[atm_strike]
            try:
                prev_ce = prev.get('ce_oi', 0)
                prev_pe = prev.get('pe_oi', 0)
                if prev_ce == 0:
                    ce_change_pct = 100.0 if current.get('ce_oi', 0) > 0 else 0.0
                else:
                    ce_change_pct = ((current.get('ce_oi', 0) - prev_ce) / prev_ce) * 100

                if prev_pe == 0:
                    pe_change_pct = 100.0 if current.get('pe_oi', 0) > 0 else 0.0
                else:
                    pe_change_pct = ((current.get('pe_oi', 0) - prev_pe) / prev_pe) * 100
            except Exception:
                ce_change_pct = 0.0
                pe_change_pct = 0.0

        return {
            'ce_oi': current.get('ce_oi', 0),
            'pe_oi': current.get('pe_oi', 0),
            'ce_vol': current.get('ce_vol', 0),
            'pe_vol': current.get('pe_vol', 0),
            'ce_ltp': current.get('ce_ltp', 0),
            'pe_ltp': current.get('pe_ltp', 0),
            'ce_change_pct': round(ce_change_pct, 1),
            'pe_change_pct': round(pe_change_pct, 1),
            'has_previous_data': has_prev,
            'atm_strike': atm_strike
        }

    @staticmethod
    def validate_atm_data(atm_data):
        ce_ltp = atm_data.get('ce_ltp', 0)
        pe_ltp = atm_data.get('pe_ltp', 0)
        ce_oi = atm_data.get('ce_oi', 0)
        pe_oi = atm_data.get('pe_oi', 0)

        if ce_ltp == 0 and pe_ltp == 0 and ce_oi == 0 and pe_oi == 0:
            return False, "ATM data ALL ZERO (API not updating or wrong strike)"

        # If premiums exist but tiny (< min threshold) they might be deep OTM — warn
        if ce_ltp > 0 and ce_ltp < 2:
            return False, f"CE premium {ce_ltp:.2f} too low (deep OTM?)"
        if pe_ltp > 0 and pe_ltp < 2:
            return False, f"PE premium {pe_ltp:.2f} too low (deep OTM?)"

        if ce_oi == 0 and pe_oi == 0:
            return False, "ATM has NO open interest (illiquid strike)"

        return True, "ATM data valid"

    @staticmethod
    def validate_atm_strike(new_atm, previous_atm, futures_price):
        """Validate ATM strike makes sense"""
        if previous_atm is None:
            return True, "First ATM"

        atm_diff = abs(new_atm - previous_atm)
        if atm_diff > 200:
            # large jump; ensure futures proximity
            futures_to_atm = abs(futures_price - new_atm)
            if futures_to_atm > 100:
                return False, f"ATM {new_atm} too far from futures {futures_price:.0f} ({futures_to_atm:.0f} pts)"
        if atm_diff > 0:
            logger.info(f"📊 ATM SHIFT: {previous_atm} → {new_atm} ({atm_diff:+.0f} pts)")
        return True, "ATM valid"

    @staticmethod
    def check_oi_reversal(signal_type, oi_changes_history, threshold=EXIT_OI_REVERSAL_THRESHOLD):
        """
        Check for sustained building (reversal) in OI history list of percent changes.
        Returns: (is_reversal(bool), strength(str), avg(float), message(str))
        """
        if not oi_changes_history or len(oi_changes_history) < EXIT_OI_CONFIRMATION_CANDLES:
            return False, 'none', 0.0, "Insufficient data"

        recent = oi_changes_history[-EXIT_OI_CONFIRMATION_CANDLES:]
        # Count candles above threshold
        building_count = sum(1 for oi in recent if oi > threshold)
        avg_building = sum(recent) / len(recent)

        if building_count >= EXIT_OI_CONFIRMATION_CANDLES:
            strength = 'strong' if avg_building > (threshold * 1.5) else 'medium'
            return True, strength, avg_building, f"{signal_type} sustained building: {building_count}/{len(recent)} candles"

        # Single big spike
        current = recent[-1]
        if current > EXIT_OI_SPIKE_THRESHOLD:
            return True, 'spike', current, f"{signal_type} spike: {current:.1f}%"

        return False, 'none', current, f"{signal_type} OI change: {current:.1f}% (not sustained)"


# ==================== Volume Analyzer ====================
class VolumeAnalyzer:
    """ V2: Delta volume comparison with adaptive thresholds """

    @staticmethod
    def calculate_total_volume(strike_data):
        if not strike_data:
            return 0, 0
        ce_vol = sum(d.get('ce_vol', 0) for d in strike_data.values())
        pe_vol = sum(d.get('pe_vol', 0) for d in strike_data.values())
        return ce_vol, pe_vol

    @staticmethod
    def detect_volume_spike(current, avg, adaptive_threshold=None):
        """ ADAPTIVE threshold handling with edge-case guards """
        if current is None or avg is None:
            return False, 0.0

        try:
            if avg <= 0:
                # if avg missing, be conservative — require a higher absolute current
                is_spike = current >= (VOL_SPIKE_STRONG * 1000)  # arbitrary safety guard
                ratio = float('inf') if avg == 0 and current > 0 else 0.0
                threshold = adaptive_threshold or VOL_SPIKE_MULTIPLIER
                logger.info(f"📊 VOL SPIKE CHECK (no historical avg): current={current}, threshold={threshold}")
                return is_spike, round(ratio if ratio != float('inf') else current / (1 if avg == 0 else avg), 2)
            ratio = current / avg
            threshold = adaptive_threshold if adaptive_threshold is not None else VOL_SPIKE_MULTIPLIER
            is_spike = ratio >= threshold
            logger.info(f"📊 VOL SPIKE CHECK: {current:,.0f} / {avg:,.0f} = {ratio:.2f}x")
            logger.info(f" Threshold: {threshold}x, Result: {'🔥 SPIKE!' if is_spike else 'Normal'}")
            return is_spike, round(ratio, 2)
        except Exception as e:
            logger.error(f"❌ Volume spike check error: {e}")
            return False, 0.0

    @staticmethod
    def calculate_order_flow(strike_data):
        ce_vol, pe_vol = VolumeAnalyzer.calculate_total_volume(strike_data)
        # Guard division by zero and clamp ratio
        try:
            if ce_vol == 0 and pe_vol == 0:
                return 1.0
            if pe_vol == 0:
                return 5.0
            if ce_vol == 0:
                return 0.2
            ratio = ce_vol / pe_vol
            return round(max(0.2, min(ratio, 5.0)), 2)
        except Exception:
            return 1.0

    @staticmethod
    def analyze_volume_trend(df, periods=5, live_volume_delta=None, candle_frozen=False):
        """
        Uses DELTA mode when available, falls back to candle volumes.
        Returns dictionary with keys: trend, avg_volume, current_volume, ratio, source
        """
        # MODE 1: DELTA
        if live_volume_delta is not None:
            logger.info("📊 VOL CALC (DELTA MODE - CORRECT!):")
            if df is not None and len(df) >= periods:
                recent = df['volume'].tail(periods)
                avg = recent.mean() if len(recent) > 0 else 0
            else:
                avg = 50000  # fallback reasonable
                logger.warning(f"⚠️ No candle history, using fallback avg: {avg:,.0f}")
            current = float(live_volume_delta)
            if avg <= 0:
                ratio = 0.0
            else:
                ratio = current / avg
            trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
            return {
                'trend': trend,
                'avg_volume': round(float(avg), 2),
                'current_volume': round(float(current), 2),
                'ratio': round(float(ratio), 2),
                'source': 'DELTA',
                'adaptive_threshold': VOL_SPIKE_MULTIPLIER
            }

        # MODE 2: CANDLE fallback
        elif df is not None and len(df) >= periods:
            logger.info("📊 VOL CALC (CANDLE MODE - Fallback):")
            recent = df['volume'].tail(periods + 1)
            if not pd.api.types.is_numeric_dtype(recent):
                logger.error("❌ Volume not numeric")
                return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'ERROR'}
            avg = recent.iloc[:-1].mean()
            current = recent.iloc[-1]
            if pd.isna(avg) or pd.isna(current) or avg <= 0:
                logger.error("❌ Invalid avg/current values")
                return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'ERROR'}
            ratio = current / avg
            trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
            if candle_frozen:
                logger.warning("⚠️ CANDLE FROZEN - Volume data may be stale!")
                source = 'CANDLE_FROZEN'
            else:
                source = 'CANDLE'
            return {
                'trend': trend,
                'avg_volume': round(float(avg), 2),
                'current_volume': round(float(current), 2),
                'ratio': round(float(ratio), 2),
                'source': source,
                'adaptive_threshold': VOL_SPIKE_MULTIPLIER
            }

        else:
            logger.warning("⚠️ VOL CALC: Insufficient data")
            return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'NO_DATA'}


# ==================== Technical Analyzer ====================
class TechnicalAnalyzer:
    """Technical indicators with fallback support"""

    @staticmethod
    def calculate_vwap(df, live_vwap_fallback=None):
        """VWAP calculation from candles with fallback to live incremental VWAP"""
        if df is None or len(df) == 0:
            if live_vwap_fallback is not None:
                logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                return round(float(live_vwap_fallback), 2)
            return None
        try:
            df_copy = df.copy()
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
            df_copy['vol_price'] = df_copy['typical_price'] * df_copy['volume']
            df_copy['cum_vol_price'] = df_copy['vol_price'].cumsum()
            df_copy['cum_volume'] = df_copy['volume'].cumsum()
            # guard division by zero
            if df_copy['cum_volume'].iloc[-1] == 0:
                if live_vwap_fallback is not None:
                    return round(float(live_vwap_fallback), 2)
                return None
            df_copy['vwap'] = df_copy['cum_vol_price'] / df_copy['cum_volume']
            vwap = round(float(df_copy['vwap'].iloc[-1]), 2)
            # basic sanity check: if vwap is wildly different from last close
            last_close = float(df_copy['close'].iloc[-1])
            vwap_diff = abs(vwap - last_close)
            if vwap_diff > 1000:  # very large difference -> suspicious
                logger.warning(f"⚠️ VWAP {vwap:.2f} too far from close {last_close:.2f}")
                if live_vwap_fallback is not None:
                    logger.warning(f" Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                    return round(float(live_vwap_fallback), 2)
            return vwap
        except Exception as e:
            logger.error(f"❌ VWAP error: {e}")
            if live_vwap_fallback is not None:
                logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                return round(float(live_vwap_fallback), 2)
            return None

    @staticmethod
    def calculate_vwap_distance(price, vwap):
        if vwap is None or price is None:
            return 0.0
        return round(price - vwap, 2)

    @staticmethod
    def validate_signal_with_vwap(signal_type, spot, vwap, atr):
        """
        Validate signals against VWAP with scoring.
        Returns tuple (bool allowed, reason_str, score_int)
        """
        if vwap is None or spot is None or atr is None or atr <= 0:
            return False, "Missing VWAP/Price/ATR data", 0

        distance = spot - vwap
        # buffer: prefer ATR-based buffer when strict mode set
        if VWAP_STRICT_MODE:
            buffer = max(1.0, atr * VWAP_DISTANCE_MAX_ATR_MULTIPLE)
        else:
            buffer = VWAP_BUFFER or max(1.0, atr * 0.5)

        # signal_type "CE_BUY": want price near or above VWAP but not overly extended below/above
        if signal_type == "CE_BUY":
            if distance < -buffer:
                return False, f"Price {abs(distance):.0f} pts below VWAP (too far below)", 0
            if distance > buffer * 6:
                return False, f"Price {distance:.0f} pts above VWAP (overextended)", 0
            # scoring: closer above VWAP increases score, slightly below reduces a bit
            if distance >= 0:
                score = min(100, int(80 + (distance / (buffer + 1) * 20)))
            else:
                score = max(40, int(80 - (abs(distance) / (buffer + 1) * 20)))
            return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)

        elif signal_type == "PE_BUY":
            if distance > buffer:
                return False, f"Price {distance:.0f} pts above VWAP (too far)", 0
            if distance < -buffer * 6:
                return False, f"Price {abs(distance):.0f} pts below VWAP (overextended)", 0
            if distance <= 0:
                score = min(100, int(80 + (abs(distance) / (buffer + 1) * 20)))
            else:
                score = max(40, int(80 - (distance / (buffer + 1) * 20)))
            return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)

        return False, "Unknown signal type", 0

    @staticmethod
    def calculate_atr(df, period=ATR_PERIOD, synthetic_atr_fallback=None):
        """ATR calculation with fallback to synthetic"""
        if df is None or len(df) < period:
            if synthetic_atr_fallback is not None:
                logger.warning(f"⚠️ Using SYNTHETIC ATR fallback: {synthetic_atr_fallback:.1f}")
                return synthetic_atr_fallback
            return ATR_FALLBACK
        try:
            df_copy = df.copy()
            df_copy['h_l'] = df_copy['high'] - df_copy['low']
            df_copy['h_cp'] = (df_copy['high'] - df_copy['close'].shift(1)).abs()
            df_copy['l_cp'] = (df_copy['low'] - df_copy['close'].shift(1)).abs()
            df_copy['tr'] = df_copy[['h_l', 'h_cp', 'l_cp']].max(axis=1)
            atr = df_copy['tr'].rolling(window=period, min_periods=1).mean().iloc[-1]
            return round(float(atr), 2)
        except Exception as e:
            logger.error(f"❌ ATR error: {e}")
            if synthetic_atr_fallback is not None:
                return synthetic_atr_fallback
            return ATR_FALLBACK

    @staticmethod
    def analyze_candle(df):
        if df is None or len(df) == 0:
            return TechnicalAnalyzer._empty_candle()
        try:
            candle = df.iloc[-1]
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            total_size = h - l
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            color = 'GREEN' if c > o else 'RED' if c < o else 'DOJI'
            rejection = False
            rejection_type = None
            if body > 0 and upper_wick > body * EXIT_CANDLE_REJECTION_MULTIPLIER:
                rejection = True
                rejection_type = 'upper'
            elif body > 0 and lower_wick > body * EXIT_CANDLE_REJECTION_MULTIPLIER:
                rejection = True
                rejection_type = 'lower'
            return {
                'color': color,
                'size': round(total_size, 2),
                'body_size': round(body, 2),
                'upper_wick': round(upper_wick, 2),
                'lower_wick': round(lower_wick, 2),
                'rejection': rejection,
                'rejection_type': rejection_type,
                'open': o, 'high': h, 'low': l, 'close': c
            }
        except Exception as e:
            logger.error(f"❌ Candle error: {e}")
            return TechnicalAnalyzer._empty_candle()

    @staticmethod
    def detect_momentum(df, periods=3):
        if df is None or len(df) < periods:
            return {'direction': 'unknown', 'strength': 0, 'consecutive_green': 0, 'consecutive_red': 0}
        recent = df.tail(periods)
        min_body_size = MIN_CANDLE_SIZE
        green_valid = 0
        red_valid = 0
        for idx, row in recent.iterrows():
            body_size = abs(row['close'] - row['open'])
            if body_size >= min_body_size:
                if row['close'] > row['open']:
                    green_valid += 1
                elif row['close'] < row['open']:
                    red_valid += 1
        direction = 'bullish' if green_valid >= 2 else 'bearish' if red_valid >= 2 else 'sideways'
        strength = green_valid if green_valid >= 2 else red_valid if red_valid >= 2 else 0
        return {'direction': direction, 'strength': strength, 'consecutive_green': green_valid, 'consecutive_red': red_valid}

    @staticmethod
    def _empty_candle():
        return {'color': 'UNKNOWN', 'size': 0, 'body_size': 0, 'upper_wick': 0, 'lower_wick': 0, 'rejection': False, 'rejection_type': None, 'open': 0, 'high': 0, 'low': 0, 'close': 0}


# ==================== Market Analyzer ====================
class MarketAnalyzer:
    """Market structure helper functions"""

    @staticmethod
    def calculate_max_pain(strike_data, spot_price):
        if not strike_data:
            return 0, 0.0
        strikes = sorted(strike_data.keys())
        if not strikes:
            return 0, 0.0
        max_pain_strike = strikes[0]
        min_pain = float('inf')
        for test_strike in strikes:
            total_pain = 0.0
            for strike, data in strike_data.items():
                ce_oi = data.get('ce_oi', 0)
                pe_oi = data.get('pe_oi', 0)
                if test_strike > strike:
                    total_pain += ce_oi * (test_strike - strike)
                if test_strike < strike:
                    total_pain += pe_oi * (strike - test_strike)
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike
        return max_pain_strike, round(min_pain, 2)

    @staticmethod
    def detect_gamma_zone():
        try:
            from config import get_next_weekly_expiry
            today = datetime.now(IST).date()
            expiry = datetime.strptime(get_next_weekly_expiry(), '%Y-%m-%d').date()
            return today == expiry
        except Exception:
            return False

    @staticmethod
    def calculate_sentiment(pcr, order_flow, ce_change, pe_change):
        bullish = 0
        bearish = 0
        if pcr > PCR_BULLISH:
            bullish += 1
        elif pcr < PCR_BEARISH:
            bearish += 1
        if order_flow < 1.0:
            bullish += 1
        elif order_flow > 1.5:
            bearish += 1
        if ce_change < -2.0:
            bullish += 1
        if pe_change < -2.0:
            bearish += 1
        if bullish > bearish:
            return "BULLISH"
        elif bearish > bullish:
            return "BEARISH"
        else:
            return "NEUTRAL"
