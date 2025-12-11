"""
NIFTY Trading Bot - Main Orchestrator
FIXED: Conflicting OI detection, ATM validation, volume debug
"""

import asyncio
from datetime import datetime

from config import *
from utils import *
from data_manager import UpstoxClient, RedisBrain, DataFetcher
from analyzers import OIAnalyzer, VolumeAnalyzer, TechnicalAnalyzer, MarketAnalyzer
from signal_engine import SignalGenerator, SignalValidator
from position_tracker import PositionTracker
from alerts import TelegramBot, MessageFormatter

BOT_VERSION = "4.1.0-FIXED"

logger = setup_logger("main")


class NiftyTradingBot:
    """Main bot orchestrator - PRODUCTION READY"""
    
    def __init__(self):
        self.memory = RedisBrain()
        self.upstox = None
        self.data_fetcher = None
        
        self.oi_analyzer = OIAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        
        self.signal_gen = SignalGenerator()
        self.signal_validator = SignalValidator()
        self.position_tracker = PositionTracker()
        
        self.telegram = TelegramBot()
        self.formatter = MessageFormatter()
        
        self.previous_strike_data = None
        self.exit_triggered_this_cycle = False
    
    async def initialize(self):
        """Initialize bot with comprehensive startup notification"""
        logger.info("=" * 60)
        logger.info(f"🚀 NIFTY Trading Bot v{BOT_VERSION}")
        logger.info("=" * 60)
        
        self.upstox = UpstoxClient()
        await self.upstox.__aenter__()
        
        self.data_fetcher = DataFetcher(self.upstox)
        
        # Get contract details from ACTUAL auto-detection
        weekly_expiry = get_next_weekly_expiry()
        
        # Get actual detected futures info
        monthly_expiry = self.upstox.futures_expiry.strftime('%Y-%m-%d') if self.upstox.futures_expiry else "AUTO"
        futures_contract = self.upstox.futures_symbol if self.upstox.futures_symbol else "NIFTY FUTURES"
        
        current_time = format_time_ist(get_ist_time())
        
        # Calculate deep analysis strikes for display
        example_atm = 24150
        deep_strikes = get_deep_analysis_strikes(example_atm)
        deep_range = f"{deep_strikes[0]}-{deep_strikes[-1]}"
        
        fetch_min, fetch_max = get_strike_range_fetch(example_atm)
        
        startup_msg = f"""
🚀 <b>NIFTY BOT v{BOT_VERSION} STARTED</b>

━━━━━━━━━━━━━━━━━━━━
🔧 <b>FIXES IN THIS VERSION</b>
━━━━━━━━━━━━━━━━━━━━

✅ Conflicting OI Detection
✅ ATM Data Validation
✅ Volume Calculation Debug
✅ Better Error Logging

━━━━━━━━━━━━━━━━━━━━
📅 <b>CONTRACT DETAILS</b>
━━━━━━━━━━━━━━━━━━━━

<b>Futures (MONTHLY):</b>
• Contract: {futures_contract}
• Expiry: {monthly_expiry}
• Usage: Technical analysis (VWAP, ATR, Volume)

<b>Options (WEEKLY):</b>
• Expiry: {weekly_expiry}
• Usage: Trading instrument + OI analysis

━━━━━━━━━━━━━━━━━━━━
📊 <b>DATA STRATEGY</b>
━━━━━━━━━━━━━━━━━━━━

<b>MONTHLY Futures Data:</b>
✅ Candles for VWAP/ATR/EMA
✅ LIVE price for Entry/Exit decisions
✅ Volume analysis (DEBUG MODE ON)

<b>Spot Price:</b>
✅ ATM strike calculation only

<b>WEEKLY Option Chain:</b>
✅ Fetch: 11 strikes (ATM ± 5)
   Range: {fetch_min} to {fetch_max}
✅ Deep Analysis: 5 strikes (ATM ± 2)
   Range: {deep_range}
✅ Total OI: All 11 strikes
✅ Unwinding Analysis: 5 deep strikes

━━━━━━━━━━━━━━━━━━━━
🔧 <b>TIMING &amp; WARMUP</b>
━━━━━━━━━━━━━━━━━━━━

• Market Opens: 9:15 AM (ignored - freak trades)
• First Data: 9:16 AM (base reference)
• Early Signals: 9:21 AM (confidence ≥ 85%)
• Full Signals: 9:31 AM (confidence ≥ 70%)
• Signal Window: 9:21 AM - 3:15 PM
• Warmup Period: {WARMUP_MINUTES} min from first snapshot
• Scan Interval: {SCAN_INTERVAL}s
• Memory TTL: {MEMORY_TTL_HOURS}h (auto-cleanup)

━━━━━━━━━━━━━━━━━━━━
⚙️ <b>OI THRESHOLDS (STRICT)</b>
━━━━━━━━━━━━━━━━━━━━

<b>Entry Requirements (AND Logic):</b>
• 5m OI Unwinding: &lt; -{MIN_OI_5M_FOR_ENTRY}%
• 15m OI Unwinding: &lt; -{MIN_OI_15M_FOR_ENTRY}%
• BOTH timeframes must show unwinding
• ATM OI Threshold: &lt; -{ATM_OI_THRESHOLD}%
• Volume Spike: ≥ {VOL_SPIKE_MULTIPLIER}x average
• 🆕 NO Conflicting OI (both building blocked)

<b>Strong Signal:</b>
• 5m OI: &lt; -{STRONG_OI_5M_THRESHOLD}%
• 15m OI: &lt; -{STRONG_OI_15M_THRESHOLD}%

━━━━━━━━━━━━━━━━━━━━
🎯 <b>RISK MANAGEMENT</b>
━━━━━━━━━━━━━━━━━━━━

• Premium SL: {PREMIUM_SL_PERCENT}%
• Trailing SL: {'Enabled' if ENABLE_TRAILING_SL else 'Disabled'}
• Trailing Distance: {int(TRAILING_SL_DISTANCE * 100)}%
• Signal Cooldown: {SIGNAL_COOLDOWN_SECONDS}s
• Min Confidence: {MIN_CONFIDENCE}%
• Min Primary Checks: {MIN_PRIMARY_CHECKS}/3

<b>Exit Protection:</b>
• Min Hold Time: {MIN_HOLD_TIME_MINUTES} min
• OI Exit Hold: {MIN_HOLD_BEFORE_OI_EXIT} min
• OI Reversal: {EXIT_OI_REVERSAL_THRESHOLD}% sustained
• Volume Dry: &lt; {EXIT_VOLUME_DRY_THRESHOLD}x
• Premium Drop: {EXIT_PREMIUM_DROP_PERCENT}% from peak

<b>Re-Entry Protection:</b>
• Same Strike Cooldown: {SAME_STRIKE_COOLDOWN_MINUTES} min
• Opposite Signal Gap: {OPPOSITE_SIGNAL_COOLDOWN_MINUTES} min
• Same Direction Gap: {SAME_DIRECTION_COOLDOWN_MINUTES} min

━━━━━━━━━━━━━━━━━━━━
📈 <b>TECHNICAL SETTINGS</b>
━━━━━━━━━━━━━━━━━━━━

• ATR Period: {ATR_PERIOD}
• ATR Target Multiple: {ATR_TARGET_MULTIPLIER}x
• ATR SL Multiple: {ATR_SL_MULTIPLIER}x
• VWAP Buffer: {VWAP_BUFFER} pts
• VWAP Strict Mode: {'ON' if VWAP_STRICT_MODE else 'OFF'}
• PCR Bullish: &gt; {PCR_BULLISH}
• PCR Bearish: &lt; {PCR_BEARISH}

━━━━━━━━━━━━━━━━━━━━
⏰ Bot started at {current_time}
"""
        
        if self.telegram.is_enabled():
            await self.telegram.send(startup_msg)
        
        logger.info("✅ Bot initialized successfully")
        logger.info(f"📅 Monthly Futures: {futures_contract} (Expiry: {monthly_expiry})")
        logger.info(f"📅 Weekly Options: {weekly_expiry}")
        logger.info(f"📊 Strike Strategy: Fetch 11, Analyze 5 deep")
        logger.info("=" * 60)
    
    async def shutdown(self):
        """Shutdown bot"""
        logger.info("🛑 Shutting down...")
        self.running = False
        
        if self.upstox:
            await self.upstox.__aexit__(None, None, None)
        
        logger.info("✅ Shutdown complete")
    
    async def run(self):
        """Main loop"""
        self.running = True
        
        try:
            await self.initialize()
            
            while self.running:
                try:
                    await self._cycle()
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}", exc_info=True)
                
                await asyncio.sleep(SCAN_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("⚠️ Keyboard interrupt")
        finally:
            await self.shutdown()
    
    async def _cycle(self):
        """Single scan cycle"""
        now = get_ist_time()
        status, _ = get_market_status()
        current_time = now.time()
        
        self.exit_triggered_this_cycle = False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"⏰ {format_time_ist(now)} | {status}")
        logger.info(f"{'='*60}")
        
        if is_market_closed():
            logger.info("🌙 Market closed")
            return
        
        if is_premarket():
            logger.info("🌅 Premarket - waiting for 9:16 AM")
            await self.memory.load_previous_day_data()
            return
        
        if current_time >= time(9, 15) and current_time < time(9, 16):
            logger.info("⏭️ Skipping 9:15 AM (freak trade prevention)")
            return
        
        logger.info("📥 Fetching market data...")
        
        # ========== STEP 1: FETCH ALL DATA ==========
        
        # Fetch spot (for ATM calculation)
        spot = await self.data_fetcher.fetch_spot()
        if not validate_price(spot):
            logger.error("❌ STOP: Spot validation failed")
            return
        logger.info(f"  ✅ Spot: ₹{spot:.2f}")
        
        # Fetch MONTHLY futures candles (for technical analysis)
        futures_df = await self.data_fetcher.fetch_futures_candles()
        if not validate_candle_data(futures_df):
            logger.error("❌ STOP: Futures candles validation failed")
            return
        logger.info(f"  ✅ Futures Candles: {len(futures_df)} bars (for VWAP/ATR)")
        
        # Fetch MONTHLY futures LIVE price (for entry/exit)
        futures_ltp = await self.data_fetcher.fetch_futures_ltp()
        if not validate_price(futures_ltp):
            logger.error("❌ STOP: Live Futures price validation failed")
            return
        logger.info(f"  ✅ Futures LIVE: ₹{futures_ltp:.2f} (REAL-TIME)")
        
        # Compare candle close vs live price
        candle_close = futures_df['close'].iloc[-1]
        price_diff = futures_ltp - candle_close
        logger.info(f"  📊 Price Check: Candle={candle_close:.2f}, Live={futures_ltp:.2f}, Diff={price_diff:+.2f}")
        
        # Fetch WEEKLY option chain (11 strikes)
        option_result = await self.data_fetcher.fetch_option_chain(spot)
        if not option_result:
            logger.error("❌ STOP: Option chain returned None")
            return
        
        atm, strike_data = option_result
        if not validate_strike_data(strike_data):
            logger.error(f"❌ STOP: Strike validation failed")
            return
        
        # Get deep analysis strikes
        deep_strikes = get_deep_analysis_strikes(atm)
        logger.info(f"  ✅ Strikes: {len(strike_data)} total (ATM {atm})")
        logger.info(f"  🔍 Deep Analysis: {len(deep_strikes)} strikes {deep_strikes[0]}-{deep_strikes[-1]}")
        
        # Use LIVE price for all decisions
        futures_price = futures_ltp
        logger.info(f"\n💹 Prices: Spot={spot:.2f}, Futures(LIVE)={futures_price:.2f}, ATM={atm}")
        
        # ========== STEP 2: SAVE OI SNAPSHOTS (ALL 11 STRIKES) ==========
        
        logger.info("🔄 Saving OI snapshots (11 strikes)...")
        total_ce, total_pe = self.oi_analyzer.calculate_total_oi(strike_data)
        deep_ce, deep_pe, _ = self.oi_analyzer.calculate_deep_analysis_oi(strike_data, atm)
        
        self.memory.save_total_oi(total_ce, total_pe)
        
        for strike, data in strike_data.items():
            self.memory.save_strike(strike, data)
        
        logger.info(f"  ✅ Total OI (11 strikes): CE={total_ce:,.0f}, PE={total_pe:,.0f}")
        logger.info(f"  🔍 Deep OI (5 strikes): CE={deep_ce:,.0f}, PE={deep_pe:,.0f}")
        
        # ========== STEP 3: CALCULATE OI CHANGES ==========
        
        logger.info("📊 Calculating OI changes...")
        
        ce_5m, pe_5m, has_5m = self.memory.get_total_oi_change(total_ce, total_pe, 5)
        ce_15m, pe_15m, has_15m = self.memory.get_total_oi_change(total_ce, total_pe, 15)
        
        atm_info = self.oi_analyzer.get_atm_oi_changes(
            strike_data, 
            atm, 
            self.previous_strike_data
        )
        
        atm_data = self.oi_analyzer.get_atm_data(strike_data, atm)
        atm_ce_5m, atm_pe_5m, has_atm_5m = self.memory.get_strike_oi_change(atm, atm_data, 5)
        atm_ce_15m, atm_pe_15m, has_atm_15m = self.memory.get_strike_oi_change(atm, atm_data, 15)
        
        if not atm_info['has_previous_data']:
            atm_info['ce_change_pct'] = atm_ce_15m
            atm_info['pe_change_pct'] = atm_pe_15m
        
        logger.info(f"  5m:  CE={ce_5m:+.1f}% PE={pe_5m:+.1f}% {'✅' if has_5m else '⏳'}")
        logger.info(f"  15m: CE={ce_15m:+.1f}% PE={pe_15m:+.1f}% {'✅' if has_15m else '⏳'}")
        logger.info(f"  ATM {atm}: CE={atm_info['ce_change_pct']:+.1f}% PE={atm_info['pe_change_pct']:+.1f}%")
        
        # 🆕 VALIDATE ATM DATA
        atm_valid, atm_reason = self.oi_analyzer.validate_atm_data(atm_data)
        if not atm_valid:
            logger.warning(f"⚠️ ATM DATA ISSUE: {atm_reason}")
            logger.warning(f"   CE LTP: ₹{atm_data.get('ce_ltp', 0):.2f}, PE LTP: ₹{atm_data.get('pe_ltp', 0):.2f}")
            logger.warning(f"   CE OI: {atm_data.get('ce_oi', 0):,.0f}, PE OI: {atm_data.get('pe_oi', 0):,.0f}")
        
        self.previous_strike_data = strike_data.copy()
        
        # ========== STEP 4: RUN ANALYSIS ==========
        
        logger.info("🔍 Running technical analysis...")
        
        pcr = self.oi_analyzer.calculate_pcr(total_pe, total_ce)
        vwap = self.technical_analyzer.calculate_vwap(futures_df)
        atr = self.technical_analyzer.calculate_atr(futures_df)
        vwap_dist = self.technical_analyzer.calculate_vwap_distance(futures_price, vwap) if vwap else 0
        candle = self.technical_analyzer.analyze_candle(futures_df)
        momentum = self.technical_analyzer.detect_momentum(futures_df)
        
        # 🆕 VOLUME ANALYSIS with detailed logging
        vol_trend = self.volume_analyzer.analyze_volume_trend(futures_df)
        logger.info(f"📊 VOLUME ANALYSIS RESULT:")
        logger.info(f"   Trend: {vol_trend['trend']}")
        logger.info(f"   Current: {vol_trend['current_volume']:,.0f}")
        logger.info(f"   Average: {vol_trend['avg_volume']:,.0f}")
        logger.info(f"   Ratio: {vol_trend['ratio']:.2f}x")
        
        vol_spike, vol_ratio = self.volume_analyzer.detect_volume_spike(
            vol_trend['current_volume'], vol_trend['avg_volume']
        )
        
        order_flow = self.volume_analyzer.calculate_order_flow(strike_data)
        
        gamma = self.market_analyzer.detect_gamma_zone()
        unwinding = self.oi_analyzer.detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m)
        
        # 🆕 CHECK CONFLICTING OI
        is_conflicting, conflict_reason = self.oi_analyzer.detect_conflicting_oi(
            ce_5m, ce_15m, pe_5m, pe_15m
        )
        
        if is_conflicting:
            logger.warning(f"⚠️ CONFLICTING OI DETECTED: {conflict_reason}")
        
        if ce_15m < -STRONG_OI_15M_THRESHOLD or pe_15m < -STRONG_OI_15M_THRESHOLD:
            oi_strength = 'strong'
        elif ce_15m < -MIN_OI_15M_FOR_ENTRY or pe_15m < -MIN_OI_15M_FOR_ENTRY:
            oi_strength = 'medium'
        else:
            oi_strength = 'weak'
        
        logger.info(f"\n📊 ANALYSIS COMPLETE:")
        logger.info(f"  📈 PCR: {pcr:.2f}, VWAP: ₹{vwap:.2f}, ATR: {atr:.1f}")
        logger.info(f"  📍 Price vs VWAP: {vwap_dist:+.1f} pts (LIVE price)")
        logger.info(f"  🔄 OI Changes (Total - 11 strikes):")
        logger.info(f"     5m:  CE {ce_5m:+.1f}% | PE {pe_5m:+.1f}%")
        logger.info(f"     15m: CE {ce_15m:+.1f}% | PE {pe_15m:+.1f}% (Strength: {oi_strength})")
        if is_conflicting:
            logger.info(f"  ⚠️ OI Status: {conflict_reason}")
        logger.info(f"  📊 Volume: {vol_ratio:.1f}x {'🔥SPIKE' if vol_spike else ''}")
        logger.info(f"  💨 Flow: {order_flow:.2f}, Momentum: {momentum['direction']}")
        logger.info(f"  🎯 Gamma Zone: {gamma}")
        
        # ========== STEP 5: CHECK WARMUP ==========
        
        stats = self.memory.get_stats()
        logger.info(f"\n⏱️  WARMUP STATUS:")
        if stats['first_snapshot_time']:
            logger.info(f"  Base Time: {stats['first_snapshot_time'].strftime('%H:%M')}")
        logger.info(f"  Elapsed: {stats['elapsed_minutes']:.1f} min")
        logger.info(f"  5m Ready: {'✅' if stats['warmed_up_5m'] else '⏳'}")
        logger.info(f"  10m Ready: {'✅' if stats['warmed_up_10m'] else '⏳'}")
        logger.info(f"  15m Ready: {'✅' if stats['warmed_up_15m'] else '⏳'}")
        
        full_warmup = stats['warmed_up_15m']
        early_warmup = stats['warmed_up_5m'] and stats['elapsed_minutes'] >= 5
        
        if not full_warmup and not early_warmup:
            remaining = WARMUP_MINUTES - stats['elapsed_minutes']
            logger.info(f"\n🚫 SIGNALS BLOCKED - Warmup: {remaining:.1f} min remaining")
            return
        
        if full_warmup:
            logger.info(f"\n✅ FULL WARMUP COMPLETE - All signals active!")
        else:
            logger.info(f"\n⚡ EARLY WARMUP READY - High confidence signals only!")
        
        # ========== STEP 6: CHECK EXIT CONDITIONS ==========
        
        if self.position_tracker.has_active_position():
            logger.info(f"📍 Active position exists - checking exit...")
            
            current_data = {
                'ce_oi_5m': ce_5m,
                'pe_oi_5m': pe_5m,
                'volume_ratio': vol_ratio,
                'candle_data': candle,
                'futures_price': futures_price,
                'atm_data': atm_data
            }
            
            exit_check = self.position_tracker.check_exit_conditions(current_data)
            
            if exit_check:
                should_exit, reason, details = exit_check
                
                if reason == "SL_UPDATED" and not should_exit:
                    if self.telegram.is_enabled():
                        msg = f"🔒 <b>TRAILING SL UPDATED</b>\n\n{details}"
                        await self.telegram.send_update(msg)
                    logger.info(f"📢 Trailing SL updated: {details}")
                
                elif should_exit:
                    exit_premium = self.position_tracker._estimate_premium(current_data, 
                        self.position_tracker.active_position.signal)
                    
                    self.signal_validator.record_exit(
                        self.position_tracker.active_position.signal.signal_type,
                        self.position_tracker.active_position.signal.atm_strike
                    )
                    
                    self.position_tracker.close_position(reason, details, exit_premium)
                    
                    if self.telegram.is_enabled():
                        msg = self.formatter.format_exit_signal(
                            self.position_tracker.closed_positions[-1],
                            reason, details
                        )
                        await self.telegram.send_exit(msg)
                    
                    logger.info(f"🚪 EXIT TRIGGERED: {reason} - {details}")
                    self.exit_triggered_this_cycle = True
            else:
                logger.info(f"✅ Position holding - no exit conditions met")
        
        # ========== STEP 7: GENERATE ENTRY SIGNAL ==========
        
        if self.exit_triggered_this_cycle:
            logger.info(f"\n⏸️ EXIT TRIGGERED THIS CYCLE - Skipping entry check")
            return
        
        # 🆕 BLOCK SIGNALS IF CONFLICTING OI
        if is_conflicting:
            logger.info(f"\n🚫 SIGNALS BLOCKED - {conflict_reason}")
            return
        
        # 🆕 BLOCK SIGNALS IF ATM DATA INVALID
        if not atm_valid:
            logger.info(f"\n🚫 SIGNALS BLOCKED - {atm_reason}")
            return
        
        signal_allowed, signal_msg = is_signal_time(warmup_complete=full_warmup or early_warmup)
        
        if not self.position_tracker.has_active_position() and signal_allowed:
            logger.info(f"\n🔎 SIGNAL GENERATION:")
            logger.info(f"  No active position - checking for entry...")
            
            signal = self.signal_gen.generate(
                spot_price=spot, 
                futures_price=futures_price,
                vwap=vwap,
                vwap_distance=vwap_dist, 
                pcr=pcr, 
                atr=atr, 
                atm_strike=atm,
                atm_data=atm_data, 
                ce_total_5m=ce_5m, 
                pe_total_5m=pe_5m,
                ce_total_15m=ce_15m, 
                pe_total_15m=pe_15m,
                atm_ce_5m=atm_info['ce_change_pct'], 
                atm_pe_5m=atm_info['pe_change_pct'],
                atm_ce_15m=atm_ce_15m, 
                atm_pe_15m=atm_pe_15m,
                has_5m_total=has_5m, 
                has_15m_total=has_15m,
                has_5m_atm=has_atm_5m or atm_info['has_previous_data'], 
                has_15m_atm=has_atm_15m,
                volume_spike=vol_spike, 
                volume_ratio=vol_ratio,
                order_flow=order_flow, 
                candle_data=candle,
                gamma_zone=gamma, 
                momentum=momentum,
                multi_tf=unwinding['multi_timeframe'],
                oi_strength=oi_strength
            )
            
            if not full_warmup and signal:
                if signal.confidence < EARLY_SIGNAL_CONFIDENCE:
                    logger.info(f"  ⚡ Early signal {signal.confidence}% < {EARLY_SIGNAL_CONFIDENCE}% threshold")
                    signal = None
            
            validated = self.signal_validator.validate(signal)
            
            if validated:
                logger.info(f"\n🔔 SIGNAL GENERATED!")
                logger.info(f"  Type: {validated.signal_type.value}")
                logger.info(f"  Entry: ₹{validated.entry_price:.2f} (LIVE PRICE)")
                logger.info(f"  Confidence: {validated.confidence}%")
                logger.info(f"  VWAP Score: {validated.vwap_score}/100")
                logger.info(f"  OI Strength: {validated.oi_strength}")
                if not full_warmup:
                    logger.info(f"  ⚡ EARLY SIGNAL (High Confidence)")
                
                self.position_tracker.open_position(validated)
                
                if self.telegram.is_enabled():
                    msg = self.formatter.format_entry_signal(validated)
                    if not full_warmup:
                        msg = f"⚡ <b>EARLY SIGNAL</b> (High Confidence)\n\n" + msg
                    await self.telegram.send_signal(msg)
            else:
                logger.info(f"  ✋ No valid setup found")
        elif not signal_allowed:
            logger.info(f"\n⏰ {signal_msg}")
        elif self.position_tracker.has_active_position():
            logger.info(f"\n📍 Position already active - not generating new signals")


async def main():
    bot = NiftyTradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
