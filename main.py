"""
NIFTY Trading Bot - Main Orchestrator
Complete autonomous trading signal generator with exit alerts
"""

import asyncio
from datetime import datetime

# Import all modules
from config import *
from utils import *
from data_manager import UpstoxClient, RedisBrain, DataFetcher
from analyzers import OIAnalyzer, VolumeAnalyzer, TechnicalAnalyzer, MarketAnalyzer
from signal_engine import SignalGenerator, SignalValidator
from position_tracker import PositionTracker
from alerts import TelegramBot, MessageFormatter

BOT_VERSION = "3.0.0"

logger = setup_logger("main")


# ==================== Main Bot ====================
class NiftyTradingBot:
    """Main bot orchestrator"""
    
    def __init__(self):
        # Core components
        self.memory = RedisBrain()
        self.upstox = None
        self.data_fetcher = None
        
        # Analyzers
        self.oi_analyzer = OIAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        
        # Signal & Position
        self.signal_gen = SignalGenerator()
        self.signal_validator = SignalValidator()
        self.position_tracker = PositionTracker()
        
        # Alerts
        self.telegram = TelegramBot()
        self.formatter = MessageFormatter()
        
        # State
        self.running = False
    
    async def initialize(self):
        """Initialize bot"""
        logger.info("=" * 60)
        logger.info(f"🚀 NIFTY Trading Bot v{BOT_VERSION}")
        logger.info("=" * 60)
        
        self.upstox = UpstoxClient()
        await self.upstox.__aenter__()
        
        self.data_fetcher = DataFetcher(self.upstox)
        
        if self.telegram.is_enabled():
            await self.telegram.send(f"🚀 Bot v{BOT_VERSION} Started")
        
        logger.info("✅ Bot initialized")
        logger.info(f"📅 Next Expiry: {get_next_tuesday_expiry()}")
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"⏰ {format_time_ist(now)} | {status}")
        logger.info(f"{'='*60}")
        
        # Market closed
        if is_market_closed():
            return
        
        # Premarket
        if is_premarket():
            await self.memory.load_previous_day_data()
            return
        
        # Fetch data
        spot = await self.data_fetcher.fetch_spot()
        if not validate_price(spot):
            return
        
        futures_df = await self.data_fetcher.fetch_futures()
        if not validate_candle_data(futures_df):
            return
        
        option_result = await self.data_fetcher.fetch_option_chain(spot)
        if not option_result:
            return
        
        atm, strike_data = option_result
        if not validate_strike_data(strike_data):
            return
        
        futures_price = futures_df['close'].iloc[-1]
        
        logger.info(f"✅ Data: Spot={spot:.2f}, Futures={futures_price:.2f}, ATM={atm}")
        
        # Save OI
        total_ce, total_pe = self.oi_analyzer.calculate_total_oi(strike_data)
        self.memory.save_total_oi(total_ce, total_pe)
        
        for strike, data in strike_data.items():
            self.memory.save_strike(strike, data)
        
        # Get OI changes
        ce_5m, pe_5m, has_5m = self.memory.get_total_oi_change(total_ce, total_pe, 5)
        ce_15m, pe_15m, has_15m = self.memory.get_total_oi_change(total_ce, total_pe, 15)
        
        atm_data = self.oi_analyzer.get_atm_data(strike_data, atm)
        atm_ce_5m, atm_pe_5m, has_atm_5m = self.memory.get_strike_oi_change(atm, atm_data, 5)
        atm_ce_15m, atm_pe_15m, has_atm_15m = self.memory.get_strike_oi_change(atm, atm_data, 15)
        
        # Analysis
        pcr = self.oi_analyzer.calculate_pcr(total_pe, total_ce)
        vwap = self.technical_analyzer.calculate_vwap(futures_df)
        atr = self.technical_analyzer.calculate_atr(futures_df)
        vwap_dist = self.technical_analyzer.calculate_vwap_distance(futures_price, vwap) if vwap else 0
        candle = self.technical_analyzer.analyze_candle(futures_df)
        momentum = self.technical_analyzer.detect_momentum(futures_df)
        
        vol_trend = self.volume_analyzer.analyze_volume_trend(futures_df)
        vol_spike, vol_ratio = self.volume_analyzer.detect_volume_spike(
            vol_trend['current_volume'], vol_trend['avg_volume']
        )
        order_flow = self.volume_analyzer.calculate_order_flow(strike_data)
        
        gamma = self.market_analyzer.detect_gamma_zone()
        unwinding = self.oi_analyzer.detect_unwinding(ce_5m, ce_15m, pe_5m, pe_15m)
        
        # Log analysis
        logger.info(f"\n📊 Analysis: PCR={pcr}, VWAP={vwap:.2f}, ATR={atr:.1f}")
        logger.info(f"   OI: 5m CE={ce_5m:+.1f}% PE={pe_5m:+.1f}% | 15m CE={ce_15m:+.1f}% PE={pe_15m:+.1f}%")
        logger.info(f"   Vol: {vol_ratio:.1f}x {'SPIKE' if vol_spike else ''}, Flow={order_flow:.2f}")
        
        # Check warmup
        stats = self.memory.get_stats()
        if not stats['warmed_up_10m']:
            logger.info(f"\n⏳ WARMUP: {stats['elapsed_minutes']:.1f}/{WARMUP_MINUTES} min")
            return  # BLOCK SIGNALS
        
        # Check exit conditions if position active
        if self.position_tracker.has_active_position():
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
                
                # Estimate exit premium
                exit_premium = self.position_tracker._estimate_premium(current_data, 
                    self.position_tracker.active_position.signal)
                
                self.position_tracker.close_position(reason, details, exit_premium)
                
                # Send exit alert
                if self.telegram.is_enabled():
                    msg = self.formatter.format_exit_signal(
                        self.position_tracker.closed_positions[-1],
                        reason, details
                    )
                    await self.telegram.send_exit(msg)
                
                logger.info(f"🚪 EXIT: {reason} - {details}")
        
        # Generate entry signal if no position
        if not self.position_tracker.has_active_position() and is_signal_time():
            signal = self.signal_gen.generate(
                spot_price=spot, futures_price=futures_price, vwap=vwap,
                vwap_distance=vwap_dist, pcr=pcr, atr=atr, atm_strike=atm,
                atm_data=atm_data, ce_total_5m=ce_5m, pe_total_5m=pe_5m,
                ce_total_15m=ce_15m, pe_total_15m=pe_15m,
                atm_ce_5m=atm_ce_5m, atm_pe_5m=atm_pe_5m,
                atm_ce_15m=atm_ce_15m, atm_pe_15m=atm_pe_15m,
                has_5m_total=has_5m, has_15m_total=has_15m,
                has_5m_atm=has_atm_5m, has_15m_atm=has_atm_15m,
                volume_spike=vol_spike, volume_ratio=vol_ratio,
                order_flow=order_flow, candle_data=candle,
                gamma_zone=gamma, momentum=momentum,
                multi_tf=unwinding['multi_timeframe']
            )
            
            validated = self.signal_validator.validate(signal)
            
            if validated:
                logger.info(f"\n🔔 SIGNAL: {validated.signal_type.value} @ ₹{validated.entry_price:.2f}")
                
                # Open position
                self.position_tracker.open_position(validated)
                
                # Send alert
                if self.telegram.is_enabled():
                    msg = self.formatter.format_entry_signal(validated)
                    await self.telegram.send_signal(msg)
            else:
                logger.info("\n✋ No setup")


# ==================== Entry Point ====================
async def main():
    bot = NiftyTradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())


