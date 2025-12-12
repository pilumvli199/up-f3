"""
Analyzers V2 (fixed): OI, Volume, Technical (VWAP/ATR), Market
- defensive guards for missing data
- consistent return types
"""

import logging
import pandas as pd
from datetime import datetime
from config import *
from utils import setup_logger, IST

logger = setup_logger("analyzers")

# ==================== OI Analyzer ====================
class OIAnalyzer:
    """Open Interest analysis with 5 strikes deep focus"""

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
            # key normalization: strikes may be float or int in data
            sd = strike_data.get(strike) or strike_data.get(int(strike)) or strike_data.get(float(strike))
            if sd:
                deep_ce += sd.get('ce_oi', 0)
                deep_pe += sd.get('pe_oi', 0)
        return deep_ce, deep_pe, deep_strikes

    @staticmethod
    def calculate_pcr(total_pe, total_ce):
        if total_ce == 0:
            return 10.0 if total_pe > 0 else 1.0
        pcr = total_pe / total_ce
        return round(min(pcr, 10.0), 2)

    @staticmethod
    def detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m):
        # returns dict with booleans and strength labels
        def strength(perc_5m, perc_15m):
            if perc_15m < -STRONG_OI_15M_THRESHOLD and perc_5m < -STRONG_OI_5M_THRESHOLD:
                return 'strong'
            if perc_15m < -MIN_OI_15M_FOR_ENTRY and perc_5m < -MIN_OI_5M_FOR_ENTRY:
                return 'medium'
            return 'weak'

        ce_strength = strength(ce_5m, ce_15m)
        pe_strength = strength(pe_5m, pe_15m)

        ce_unwinding = ce_15m < -MIN_OI_15M_FOR_ENTRY and ce_5m < -MIN_OI_5M_FOR_ENTRY
        pe_unwinding = pe_15m < -MIN_OI_15M_FOR_ENTRY and pe_5m < -MIN_OI_5M_FOR_ENTRY

        multi_tf = (ce_5m < -2.0 and ce_15m < -3.0) or (pe_5m < -2.0 and pe_15m < -3.0)

        return {
            'ce_unwinding': ce_unwinding,
            'pe_unwinding': pe_unwinding,
            'ce_strength': ce_strength,
            'pe_strength': pe_strength,
            'multi_timeframe': multi_tf
        }

    @staticmethod
    def detect_conflicting_oi(ce_5m, ce_15m, pe_5m, pe_15m):
        BUILDING_THRESHOLD = 3.0
        ce_building_5m = ce_5m > BUILDING_THRESHOLD
        pe_building_5m = pe_5m > BUILDING_THRESHOLD
        if ce_building_5m and pe_building_5m:
            return True, f"BOTH CE (+{ce_5m:.1f}%) & PE (+{pe_5m:.1f}%) building on 5m = CHOPPY"

        ce_unwinding_15m = ce_15m < -MIN_OI_15M_FOR_ENTRY
        pe_building_5m_strong = pe_5m > BUILDING_THRESHOLD
        if ce_unwinding_15m and pe_building_5m_strong:
            return True, f"CE unwinding 15m ({ce_15m:.1f}%) BUT PE building 5m (+{pe_5m:.1f}%) = CONFUSED"

        pe_unwinding_15m = pe_15m < -MIN_OI_15M_FOR_ENTRY
        ce_building_5m_strong = ce_5m > BUILDING_THRESHOLD
        if pe_unwinding_15m and ce_building_5m_strong:
            return True, f"PE unwinding 15m ({pe_15m:.1f}%) BUT CE building 5m (+{ce_5m:.1f}%) = CONFUSED"

        return False, "OI aligned"

    @staticmethod
    def get_atm_data(strike_data, atm_strike):
        # normalize key access: return dict or empty mapped fields
        sd = strike_data.get(atm_strike) or strike_data.get(int(atm_strike)) or strike_data.get(float(atm_strike))
        if not sd:
            return {
                'ce_oi': 0, 'pe_oi': 0, 'ce_vol': 0, 'pe_vol': 0, 'ce_ltp': 0, 'pe_ltp': 0
            }
        return sd

    @staticmethod
    def get_atm_oi_changes(strike_data, atm_strike, previous_strike_data=None):
        current = OIAnalyzer.get_atm_data(strike_data, atm_strike)
        ce_change_pct = 0.0
        pe_change_pct = 0.0
        if previous_strike_data:
            prev = previous_strike_data.get(atm_strike) or previous_strike_data.get(int(atm_strike)) or {}
            prev_ce = prev.get('ce_oi', 0)
            curr_ce = current.get('ce_oi', 0)
            if prev_ce > 0:
                ce_change_pct = ((curr_ce - prev_ce) / prev_ce) * 100
            elif curr_ce > 0:
                ce_change_pct = 100.0

            prev_pe = prev.get('pe_oi', 0)
            curr_pe = current.get('pe_oi', 0)
            if prev_pe > 0:
                pe_change_pct = ((curr_pe - prev_pe) / prev_pe) * 100
            elif curr_pe > 0:
                pe_change_pct = 100.0

        return {
            'ce_oi': current.get('ce_oi', 0),
            'pe_oi': current.get('pe_oi', 0),
            'ce_vol': current.get('ce_vol', 0),
            'pe_vol': current.get('pe_vol', 0),
            'ce_ltp': current.get('ce_ltp', 0),
            'pe_ltp': current.get('pe_ltp', 0),
            'ce_change_pct': round(ce_change_pct, 1),
            'pe_change_pct': round(pe_change_pct, 1),
            'has_previous_data': previous_strike_data is not None,
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
        if ce_ltp > 0 and ce_ltp < 2.5:
            return False, f"CE premium {ce_ltp:.2f} too low (deep OTM?)"
        if pe_ltp > 0 and pe_ltp < 2.5:
            return False, f"PE premium {pe_ltp:.2f} too low (deep OTM?)"
        if ce_oi == 0 and pe_oi == 0:
            return False, "ATM has NO open interest (illiquid strike)"
        return True, "ATM data valid"

    @staticmethod
    def validate_atm_strike(new_atm, previous_atm, futures_price):
        if previous_atm is None:
            return True, "First ATM"
        try:
            atm_diff = abs(new_atm - previous_atm)
            if atm_diff > 200:
                futures_to_atm = abs(futures_price - new_atm)
                if futures_to_atm > 100:
                    return False, f"ATM {new_atm} too far from futures {futures_price:.0f} ({futures_to_atm:.0f} pts)"
            if atm_diff > 0:
                logger.info(f"📊 ATM SHIFT: {previous_atm} → {new_atm} ({atm_diff:+.0f} pts)")
            return True, "ATM valid"
        except Exception as e:
            logger.error(f"❌ ATM validate error: {e}")
            return False, "ATM validation error"

    @staticmethod
    def check_oi_reversal(signal_type, oi_changes_history, threshold=EXIT_OI_REVERSAL_THRESHOLD):
        if not oi_changes_history or len(oi_changes_history) < EXIT_OI_CONFIRMATION_CANDLES:
            return False, 'none', 0.0, "Insufficient data"
        recent = oi_changes_history[-EXIT_OI_CONFIRMATION_CANDLES:]
        current = recent[-1]
        building_count = sum(1 for oi in recent if oi > threshold)
        if building_count >= EXIT_OI_CONFIRMATION_CANDLES:
            avg_building = sum(recent) / len(recent)
            strength = 'strong' if avg_building > 5.0 else 'medium'
            return True, strength, avg_building, f"{signal_type} sustained building: {building_count}/{len(recent)} candles"
        if current > EXIT_OI_SPIKE_THRESHOLD:
            return True, 'spike', current, f"{signal_type} spike: {current:.1f}%"
        return False, 'none', current, f"{signal_type} OI change: {current:.1f}% (not sustained)"

# ==================== Volume Analyzer V2 ====================
class VolumeAnalyzer:
    """V2: Delta volume comparison with adaptive thresholds"""

    @staticmethod
    def calculate_total_volume(strike_data):
        if not strike_data:
            return 0, 0
        ce_vol = sum(d.get('ce_vol', 0) for d in strike_data.values())
        pe_vol = sum(d.get('pe_vol', 0) for d in strike_data.values())
        return ce_vol, pe_vol

    @staticmethod
    def detect_volume_spike(current, avg, adaptive_threshold=None):
        """Adaptive threshold safe handling"""
        try:
            if current is None or avg is None:
                return False, 0.0
            current = float(current)
            avg = float(avg)
            if avg <= 0:
                return False, 0.0
            ratio = current / avg
            threshold = adaptive_threshold if (adaptive_threshold is not None and adaptive_threshold > 0) else VOL_SPIKE_MULTIPLIER
            is_spike = ratio >= threshold
            logger.info(f"📊 VOL SPIKE CHECK: {current:,.0f} / {avg:,.0f} = {ratio:.2f}x")
            logger.info(f" Threshold: {threshold}x, Result: {'🔥 SPIKE!' if is_spike else 'Normal'}")
            return is_spike, round(ratio, 2)
        except Exception as e:
            logger.error(f"❌ detect_volume_spike error: {e}")
            return False, 0.0

    @staticmethod
    def calculate_order_flow(strike_data):
        ce_vol, pe_vol = VolumeAnalyzer.calculate_total_volume(strike_data)
        if ce_vol == 0 and pe_vol == 0:
            return 1.0
        if pe_vol == 0:
            return 5.0
        if ce_vol == 0:
            return 0.2
        ratio = ce_vol / pe_vol
        return round(max(0.2, min(ratio, 5.0)), 2)

    @staticmethod
    def analyze_volume_trend(df, periods=5, live_volume_delta=None, candle_frozen=False):
        """Uses DELTA first, then candle fallback. Returns consistent dict"""
        try:
            # DELTA MODE
            if live_volume_delta is not None:
                logger.info(f"📊 VOL CALC (DELTA MODE - CORRECT!):")
                if df is not None and len(df) >= periods:
                    recent = df['volume'].tail(periods)
                    avg = recent.mean() if len(recent) > 0 else 0
                else:
                    avg = 50000  # conservative
                    logger.warning(f"⚠️ No candle history, using fallback avg: {avg:,.0f}")
                current = float(live_volume_delta)
                if avg <= 0:
                    logger.error("❌ Invalid avg volume in delta mode")
                    return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': current, 'ratio': 1.0, 'source': 'ERROR'}
                ratio = current / avg
                trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
                return {'trend': trend, 'avg_volume': round(float(avg), 2), 'current_volume': round(float(current), 2), 'ratio': round(float(ratio), 2), 'source': 'DELTA', 'adaptive_threshold': VOL_SPIKE_MULTIPLIER}
            # CANDLE MODE fallback
            elif df is not None and len(df) >= periods:
                logger.info("📊 VOL CALC (CANDLE MODE - Fallback):")
                recent = df['volume'].tail(periods + 1)
                if not pd.api.types.is_numeric_dtype(recent):
                    logger.error("❌ Volume not numeric")
                    return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'ERROR'}
                avg = recent.iloc[:-1].mean()
                current = recent.iloc[-1]
                if pd.isna(avg) or pd.isna(current) or avg <= 0:
                    logger.error("❌ Invalid values in candle mode")
                    return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'ERROR'}
                ratio = current / avg
                trend = 'increasing' if ratio > 1.3 else 'decreasing' if ratio < 0.7 else 'stable'
                if candle_frozen:
                    logger.warning("⚠️ CANDLE FROZEN - Volume data may be stale!")
                return {'trend': trend, 'avg_volume': round(float(avg), 2), 'current_volume': round(float(current), 2), 'ratio': round(float(ratio), 2), 'source': 'CANDLE' if not candle_frozen else 'CANDLE_FROZEN', 'adaptive_threshold': VOL_SPIKE_MULTIPLIER}
            else:
                logger.warning("⚠️ VOL CALC: Insufficient data")
                return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'NO_DATA'}
        except Exception as e:
            logger.error(f"❌ analyze_volume_trend error: {e}")
            return {'trend': 'unknown', 'avg_volume': 0, 'current_volume': 0, 'ratio': 1.0, 'source': 'ERROR'}

# ==================== Technical Analyzer V2 ====================
class TechnicalAnalyzer:
    """Technical indicators with fallback support"""

    @staticmethod
    def calculate_vwap(df, live_vwap_fallback=None):
        """VWAP with fallback for frozen candles"""
        try:
            if df is None or len(df) == 0:
                if live_vwap_fallback is not None:
                    logger.warning(f"⚠️ Using LIVE VWAP fallback: ₹{live_vwap_fallback:.2f}")
                    return float(live_vwap_fallback)
                return None
            df_copy = df.copy()
            if 'volume' not in df_copy.columns or df_copy['volume'].sum() <= 0:
                if live_vwap_fallback is not None:
                    logger.warning("⚠️ Candle volume invalid - using live VWAP fallback")
                    return float(live_vwap_fallback)
                return None
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3.0
            df_copy['vol_price'] = df_copy['typical_price'] * df_copy['volume']
            df_copy['cum_vol_price'] = df_copy['vol_price'].cumsum()
            df_copy['cum_volume'] = df_copy['volume'].cumsum()
            # avoid division by zero
            if df_copy['cum_volume'].iloc[-1] <= 0:
                if live_vwap_fallback is not None:
                    return float(live_vwap_fallback)
                return None
            vwap = round((df_copy['cum_vol_price'].iloc[-1] / df_copy['cum_volume'].iloc[-1]), 2)
            # Basic sanity check: vwap shouldn't be astronomically far from last close
            last_close = df_copy['close'].iloc[-1]
            vwap_diff = abs(vwap - last_close)
            if vwap_diff > 1000:  # too wide, likely bad data
                logger.warning(f"⚠️ VWAP {vwap:.2f} too far from close {last_close:.2f}")
                if live_vwap_fallback is not None:
                    return float(live_vwap_fallback)
            return vwap
        except Exception as e:
            logger.error(f"❌ VWAP error: {e}")
            if live_vwap_fallback is not None:
                return float(live_vwap_fallback)
            return None

    @staticmethod
    def calculate_vwap_distance(price, vwap):
        if vwap is None or price is None:
            return 0.0
        try:
            return round(price - vwap, 2)
        except Exception:
            return 0.0

    @staticmethod
    def validate_signal_with_vwap(signal_type, spot, vwap, atr):
        """Return (valid:bool, reason:str, score:int)"""
        if vwap is None or spot is None:
            return False, "Missing VWAP/Price data", 0
        if atr is None or atr <= 0:
            # fallback to buffer constant
            buffer = VWAP_BUFFER
        else:
            buffer = atr * VWAP_DISTANCE_MAX_ATR_MULTIPLE if VWAP_STRICT_MODE else VWAP_BUFFER
            # ensure buffer > 0
            if buffer <= 0:
                buffer = VWAP_BUFFER

        distance = spot - vwap
        # blocking rules per type
        if signal_type == "CE_BUY":
            if distance < -buffer:
                return False, f"Price {abs(distance):.0f} pts below VWAP (too far)", 0
            if distance > buffer * 3:
                return False, f"Price {distance:.0f} pts above VWAP (overextended)", 0
            # scoring: closer to vwap and slightly above is good
            try:
                if distance >= 0:
                    score = min(100, 80 + int((distance / buffer) * 20)) if buffer > 0 else 80
                else:
                    score = max(60, 80 - int((abs(distance) / buffer) * 20)) if buffer > 0 else 80
            except Exception:
                score = 80
            return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)

        elif signal_type == "PE_BUY":
            if distance > buffer:
                return False, f"Price {distance:.0f} pts above VWAP (too far)", 0
            if distance < -buffer * 3:
                return False, f"Price {abs(distance):.0f} pts below VWAP (overextended)", 0
            try:
                if distance <= 0:
                    score = min(100, 80 + int((abs(distance) / buffer) * 20)) if buffer > 0 else 80
                else:
                    score = max(60, 80 - int((distance / buffer) * 20)) if buffer > 0 else 80
            except Exception:
                score = 80
            return True, f"VWAP distance OK: {distance:+.0f} pts", int(score)

        return False, "Unknown signal type", 0

    @staticmethod
    def calculate_atr(df, period=ATR_PERIOD, synthetic_atr_fallback=None):
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
            atr = df_copy['tr'].rolling(window=period).mean().iloc[-1]
            if pd.isna(atr) or atr <= 0:
                return synthetic_atr_fallback if synthetic_atr_fallback is not None else ATR_FALLBACK
            return round(atr, 2)
        except Exception as e:
            logger.error(f"❌ ATR error: {e}")
            return synthetic_atr_fallback if synthetic_atr_fallback is not None else ATR_FALLBACK

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
    @staticmethod
    def calculate_max_pain(strike_data, spot_price):
        if not strike_data:
            return 0, 0.0
        strikes = sorted(list(strike_data.keys()))
        max_pain_strike = strikes[len(strikes) // 2]
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
