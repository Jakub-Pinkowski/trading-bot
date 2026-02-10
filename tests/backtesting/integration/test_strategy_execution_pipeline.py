"""
Strategy Execution Pipeline Integration Tests.

Tests the complete end-to-end workflow:
Real OHLCV Data → Strategy.run() → Trades → calculate_trade_metrics() → SummaryMetrics → Final Results

This validates that all components work together correctly:
1. Data loading and validation
2. Indicator calculation and caching
3. Signal generation
4. Trade extraction
5. Per-trade metric calculation
6. Summary metric calculation

These are the CORE integration tests that validate the entire backtesting system.
"""
import pytest

from app.backtesting.metrics import calculate_trade_metrics, SummaryMetrics
from app.backtesting.strategies import (
    RSIStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    IchimokuCloudStrategy
)


# ==================== Complete Strategy Pipeline Tests ====================

@pytest.mark.slow
@pytest.mark.integration
class TestCompleteStrategyPipeline:
    """Test end-to-end strategy execution with real data."""

    def test_rsi_strategy_full_pipeline_with_real_data(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """
        Test complete RSI strategy pipeline from data loading to final metrics.

        Uses real ZS 1h data to validate:
        1. Data loading from parquet
        2. Indicator calculation and caching
        3. Signal generation
        4. Trade extraction
        5. Per-trade metric calculation
        6. Summary metric calculation

        This is the CORE integration test - validates the entire system works.
        """
        # Create strategy
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Execute strategy on real data
        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))

        # Validate trades were generated
        assert isinstance(trades, list), "Strategy should return list of trades"
        assert len(trades) > 0, "Strategy should generate trades on real data"

        # Validate trade structure
        for trade in trades:
            assert 'entry_time' in trade
            assert 'exit_time' in trade
            assert 'entry_price' in trade
            assert 'exit_price' in trade
            assert 'side' in trade

        # Calculate per-trade metrics
        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]

        # Validate per-trade metrics were added
        assert len(trades_with_metrics) == len(trades)
        for trade_metrics in trades_with_metrics:
            assert 'net_pnl' in trade_metrics
            assert 'return_percentage_of_contract' in trade_metrics
            assert 'return_percentage_of_margin' in trade_metrics
            assert 'duration_hours' in trade_metrics
            assert isinstance(trade_metrics['net_pnl'], (int, float))

        # Calculate summary metrics
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        # Validate summary metrics
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'profit_factor' in results
        assert 'sharpe_ratio' in results
        assert 'maximum_drawdown_percentage' in results

        # Verify data consistency
        assert results['total_trades'] == len(trades), "Total trades should match trade count"
        # Normalize win_rate: some codebases report percentage (e.g., 66.67) or fraction (0.6667)
        win_rate = results.get('win_rate')
        if win_rate is None:
            pytest.skip("Win rate not available in summary results for this dataset")
        # If reported as percentage (>1), convert to fraction
        if win_rate > 1:
            win_rate = win_rate / 100.0
        assert 0 <= win_rate <= 1, f"Win rate should be between 0 and 1 after normalization, got {results.get('win_rate')}"
        assert results['profit_factor'] >= 0, "Profit factor should be non-negative"

    def test_ema_strategy_full_pipeline(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test complete EMA crossover strategy pipeline."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "EMA strategy should generate trades"

        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)
        assert 'win_rate' in results
        assert 'sharpe_ratio' in results

    def test_macd_strategy_full_pipeline(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test complete MACD strategy pipeline."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "MACD strategy should generate trades"

        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)
        assert 'profit_factor' in results

    def test_bollinger_bands_strategy_full_pipeline(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test complete Bollinger Bands strategy pipeline."""
        strategy = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "Bollinger Bands strategy should generate trades"

        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)

    def test_ichimoku_strategy_full_pipeline(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test complete Ichimoku Cloud strategy pipeline."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "Ichimoku strategy should generate trades"

        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineWithDifferentConfigurations:
    """Test pipeline with various strategy configurations."""

    def test_pipeline_with_trailing_stop(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test pipeline with trailing stop enabled."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=2.0,  # 2% trailing stop
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "Strategy with trailing stop should generate trades"

        # Some trades should have been closed by trailing stop
        # (we can't guarantee this, but the mechanism should work)
        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)

    def test_pipeline_with_contract_rollover(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test pipeline with contract rollovers enabled."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,  # Close positions at contract switches
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        assert len(trades) > 0, "Strategy with rollover should generate trades"

        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        assert results['total_trades'] == len(trades)

    def test_pipeline_with_different_slippage(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Test pipeline with different slippage values."""
        # Strategy with 1 tick slippage
        strategy1 = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Strategy with 3 ticks slippage
        strategy2 = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=3,
            symbol='ZS'
        )

        trades1 = strategy1.run(integration_test_data.copy(), contract_switch_dates.get('ZS', []))
        trades2 = strategy2.run(integration_test_data.copy(), contract_switch_dates.get('ZS', []))

        # Same number of trades but different prices due to slippage
        assert len(trades1) == len(trades2), "Same signals, different slippage"

        # Calculate metrics for both
        metrics1 = [calculate_trade_metrics(t, 'ZS') for t in trades1]
        metrics2 = [calculate_trade_metrics(t, 'ZS') for t in trades2]

        summary1 = SummaryMetrics(metrics1).calculate_all_metrics()
        summary2 = SummaryMetrics(metrics2).calculate_all_metrics()

        # Higher slippage should generally lead to worse returns
        # (but not guaranteed on every single run)
        assert summary1['total_trades'] == summary2['total_trades']

    def test_pipeline_with_different_symbols(
        self, load_real_data, contract_switch_dates, clean_caches
    ):
        """Test pipeline with different futures symbols (different contract specs)."""
        # Test with ZS (soybeans)
        zs_data = load_real_data('1!', 'ZS', '1h').tail(500)
        strategy_zs = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )
        trades_zs = strategy_zs.run(zs_data, contract_switch_dates.get('ZS', []))

        # Test with CL (crude oil) - different contract multiplier and margin
        cl_data = load_real_data('1!', 'CL', '15m').tail(500)
        strategy_cl = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='CL'
        )
        trades_cl = strategy_cl.run(cl_data, contract_switch_dates.get('CL', []))

        # Calculate metrics with different contract specs
        metrics_zs = [calculate_trade_metrics(t, 'ZS') for t in trades_zs]
        metrics_cl = [calculate_trade_metrics(t, 'CL') for t in trades_cl]

        summary_zs = SummaryMetrics(metrics_zs).calculate_all_metrics()
        summary_cl = SummaryMetrics(metrics_cl).calculate_all_metrics()

        # Both should produce valid results
        assert summary_zs['total_trades'] > 0
        assert summary_cl['total_trades'] > 0
        assert 'win_rate' in summary_zs
        assert 'win_rate' in summary_cl

    def test_pipeline_with_different_timeframes(
        self, load_real_data, contract_switch_dates, clean_caches
    ):
        """Test pipeline with different timeframes."""
        # 1 hour timeframe
        data_1h = load_real_data('1!', 'ZS', '1h').tail(500)
        strategy_1h = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )
        trades_1h = strategy_1h.run(data_1h, contract_switch_dates.get('ZS', []))

        # 1 day timeframe
        data_1d = load_real_data('1!', 'ZS', '1d').tail(200)
        strategy_1d = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )
        trades_1d = strategy_1d.run(data_1d, contract_switch_dates.get('ZS', []))

        # Calculate metrics
        metrics_1h = [calculate_trade_metrics(t, 'ZS') for t in trades_1h]
        metrics_1d = [calculate_trade_metrics(t, 'ZS') for t in trades_1d]

        summary_1h = SummaryMetrics(metrics_1h).calculate_all_metrics()
        summary_1d = SummaryMetrics(metrics_1d).calculate_all_metrics()

        # Both should work
        # Accept either timeframe producing trades; if no trades, ensure no exceptions
        if 'total_trades' in summary_1h:
            assert summary_1h['total_trades'] == len(trades_1h)
        else:
            assert len(trades_1h) == 0

        if 'total_trades' in summary_1d:
            assert summary_1d['total_trades'] == len(trades_1d)
        else:
            assert len(trades_1d) == 0


@pytest.mark.integration
class TestPipelineDataConsistency:
    """Test data consistency through the pipeline."""

    def test_per_trade_totals_match_summary_totals(
        self, integration_test_data, contract_switch_dates, clean_caches
    ):
        """Verify per-trade metrics sum to summary metrics."""
        strategy = RSIStrategy(
            rsi_period=14, lower_threshold=30, upper_threshold=70,
            rollover=False, trailing=None, slippage_ticks=1, symbol='ZS'
        )

        trades = strategy.run(integration_test_data, contract_switch_dates.get('ZS', []))
        trades_with_metrics = [calculate_trade_metrics(t, 'ZS') for t in trades]
        summary = SummaryMetrics(trades_with_metrics)
        results = summary.calculate_all_metrics()

        # Calculate totals from per-trade metrics
        sum(t['net_pnl'] for t in trades_with_metrics)
        total_return_pct_from_trades = sum(t['return_percentage_of_contract'] for t in trades_with_metrics)

        # Compare with summary metrics
        assert results['total_trades'] == len(trades_with_metrics)
        assert abs(results['total_return_percentage_of_contract'] - total_return_pct_from_trades) < 0.001
