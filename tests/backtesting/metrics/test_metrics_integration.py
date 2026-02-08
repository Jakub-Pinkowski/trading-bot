"""
Integration Tests for Metrics Modules.

Tests verify that per_trade_metrics and summary_metrics modules work correctly
together as a complete pipeline.

Coverage:
- Full pipeline from raw trades to summary metrics
- Data structure compatibility between modules
- Edge case handling across module boundaries
- Field presence and usage verification
- Performance tests (with 10,000 trades - runs in ~0.1s)
"""
import time
from datetime import datetime

from app.backtesting.metrics.per_trade_metrics import calculate_trade_metrics
from app.backtesting.metrics.summary_metrics import SummaryMetrics


# ==================== Integration Tests ====================

class TestMetricsIntegration:
    """Test integration between per-trade and summary metrics."""

    def test_per_trade_to_summary_pipeline(self):
        """Test complete pipeline from trade calculation to summary."""
        # Create raw trades (not yet calculated)
        raw_trades = [
            {
                'entry_time': datetime(2024, 1, 15, 10, 0),
                'exit_time': datetime(2024, 1, 15, 14, 0),
                'entry_price': 1200.0,
                'exit_price': 1210.0,
                'side': 'long'
            },
            {
                'entry_time': datetime(2024, 1, 16, 10, 0),
                'exit_time': datetime(2024, 1, 16, 14, 0),
                'entry_price': 1200.0,
                'exit_price': 1195.0,
                'side': 'long'
            },
        ]

        # Calculate per-trade metrics
        calculated_trades = [calculate_trade_metrics(t, 'ZS') for t in raw_trades]

        # Calculate summary metrics
        summary = SummaryMetrics(calculated_trades)
        result = summary.calculate_all_metrics()

        # Verify integration
        assert result['total_trades'] == 2
        assert result['win_rate'] == 50.0
        assert 'return_percentage_of_contract' in calculated_trades[0]

        # Verify total return is sum of individual returns
        expected_total = sum(t['return_percentage_of_contract'] for t in calculated_trades)
        assert result['total_return_percentage_of_contract'] == round(expected_total, 2)

    def test_summary_metrics_handles_all_per_trade_fields(self, trade_factory):
        """Verify SummaryMetrics correctly uses all per-trade metric fields."""
        trades = [
            trade_factory('ZS', 1200, 1210),
            trade_factory('ZS', 1210, 1205),
        ]

        # Verify all per-trade fields are present
        required_fields = [
            'net_pnl',
            'return_percentage_of_contract',
            'return_percentage_of_margin',
            'duration_hours'
        ]
        for trade in trades:
            for field in required_fields:
                assert field in trade, f"Missing field: {field}"

        # Verify summary metrics use these fields
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        assert result['average_trade_duration_hours'] == 4.0
        assert 'total_return_percentage_of_contract' in result
        assert 'average_trade_return_percentage_of_contract' in result

    def test_edge_case_trades_in_pipeline(self):
        """Test pipeline with edge case trades (zero duration, breakeven, etc.)."""
        edge_cases = [
            # Zero duration trade
            {
                'entry_time': datetime(2024, 1, 15, 10, 0),
                'exit_time': datetime(2024, 1, 15, 10, 0),
                'entry_price': 1200.0,
                'exit_price': 1205.0,
                'side': 'long'
            },
            # Breakeven trade
            {
                'entry_time': datetime(2024, 1, 16, 10, 0),
                'exit_time': datetime(2024, 1, 16, 14, 0),
                'entry_price': 1200.0,
                'exit_price': 1200.0,
                'side': 'long'
            },
        ]

        calculated = [calculate_trade_metrics(t, 'ZS') for t in edge_cases]
        summary = SummaryMetrics(calculated)
        result = summary.calculate_all_metrics()

        # Should handle edge cases without errors
        assert result['total_trades'] == 2
        assert isinstance(result['win_rate'], float)
        assert 0 <= result['win_rate'] <= 100

    def test_multiple_symbols_integration(self, trade_factory):
        """Test integration with trades from multiple symbols."""
        trades = [
            trade_factory('ZS', 1200, 1210),  # Grains
            trade_factory('CL', 75.0, 76.0),  # Energies
            trade_factory('ES', 5000, 5010),  # Indices
            trade_factory('GC', 2050, 2060),  # Metals
        ]

        # All trades should have consistent structure
        for trade in trades:
            assert 'return_percentage_of_contract' in trade
            assert 'net_pnl' in trade
            assert 'margin_requirement' in trade

        # Summary should aggregate correctly
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        assert result['total_trades'] == 4
        assert result['win_rate'] == 100.0

    def test_large_trade_sequence_integration(self, trades_factory):
        """Test integration with larger trade sequences."""
        # Create 100 trades
        trades = trades_factory.mixed(win_count=60, loss_count=40, symbol='ZS')

        # Verify all trades have required fields
        for trade in trades:
            assert 'return_percentage_of_contract' in trade
            assert 'duration_hours' in trade
            assert isinstance(trade['net_pnl'], (int, float))

        # Calculate summary
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        # Verify aggregation
        assert result['total_trades'] == 100
        assert result['win_rate'] == 60.0
        assert result['winning_trades'] == 60
        assert result['losing_trades'] == 40

    def test_data_type_consistency(self, trade_factory):
        """Test that data types remain consistent through the pipeline."""
        trades = [
            trade_factory('ZS', 1200.0, 1210.0),
            trade_factory('ZS', 1210.0, 1205.0),
        ]

        # Check per-trade metric types
        for trade in trades:
            assert isinstance(trade['net_pnl'], float)
            assert isinstance(trade['duration_hours'], float)
            assert isinstance(trade['return_percentage_of_contract'], float)
            assert isinstance(trade['return_percentage_of_margin'], float)

        # Check summary metric types
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        assert isinstance(result['win_rate'], float)
        assert isinstance(result['total_return_percentage_of_contract'], float)
        assert isinstance(result['profit_factor'], float)

    def test_negative_returns_integration(self, trades_factory):
        """Test integration with predominantly losing trades."""
        # Create mostly losing trades
        trades = trades_factory.mixed(win_count=2, loss_count=8, symbol='ZS')

        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        # Verify metrics handle negative returns correctly
        assert result['total_trades'] == 10
        assert result['win_rate'] == 20.0
        assert result['total_return_percentage_of_contract'] < 0
        assert result['profit_factor'] < 1.0
        assert result['maximum_drawdown_percentage'] > 0

    def test_extreme_price_movements_integration(self, trade_factory):
        """Test integration with extreme price movements."""
        trades = [
            trade_factory('CL', 50.0, 75.0),  # Large gain
            trade_factory('CL', 75.0, 45.0),  # Large loss
            trade_factory('CL', 45.0, 70.0),  # Large gain
        ]

        # All trades should calculate without errors
        for trade in trades:
            assert isinstance(trade['net_pnl'], float)
            assert 'return_percentage_of_contract' in trade

        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        # Should handle extreme volatility
        assert result['total_trades'] == 3
        assert 'sharpe_ratio' in result
        assert 'maximum_drawdown_percentage' in result

    def test_missing_field_handling(self):
        """Test handling of trades with missing fields."""
        # Create a trade with calculated metrics
        complete_trade = calculate_trade_metrics({
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 14, 0),
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'side': 'long'
        }, 'ZS')

        # Remove a field to simulate incomplete data
        incomplete_trade = complete_trade.copy()
        del incomplete_trade['duration_hours']

        # Summary metrics should handle this gracefully
        # Note: This tests defensive programming
        trades = [complete_trade, incomplete_trade]

        # This may raise an error or handle it - depends on implementation
        # The test documents the expected behavior
        try:
            summary = SummaryMetrics(trades)
            result = summary.calculate_all_metrics()
            # If it succeeds, verify basic metrics
            assert result['total_trades'] == 2
        except (KeyError, AttributeError) as e:
            # Expected if validation is strict
            assert 'duration' in str(e).lower() or 'hours' in str(e).lower()

    def test_roundtrip_field_preservation(self, trade_factory):
        """Test that fields are preserved through the entire pipeline."""
        # Create trade with factory
        original_trade = trade_factory('ZS', 1200, 1210)

        # Store original values
        original_entry = original_trade['entry_price']
        original_exit = original_trade['exit_price']
        original_side = original_trade['side']
        original_pnl = original_trade['net_pnl']

        # Pass through summary metrics
        summary = SummaryMetrics([original_trade])
        summary.calculate_all_metrics()

        # Verify original trade wasn't modified
        assert original_trade['entry_price'] == original_entry
        assert original_trade['exit_price'] == original_exit
        assert original_trade['side'] == original_side
        assert original_trade['net_pnl'] == original_pnl

    def test_short_trades_integration(self, trade_factory):
        """Test integration with short trades."""
        trades = [
            trade_factory('ZS', 1200, 1190, side='short'),  # Profitable short
            trade_factory('ZS', 1190, 1195, side='short'),  # Losing short
            trade_factory('ZS', 1195, 1185, side='short'),  # Profitable short
        ]

        # Verify short trades calculate correctly
        for trade in trades:
            assert trade['side'] == 'short'
            assert 'net_pnl' in trade

        # Summary should handle shorts correctly
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        assert result['total_trades'] == 3
        # Should have 2 wins, 1 loss
        assert result['winning_trades'] == 2
        assert result['losing_trades'] == 1

    def test_mixed_long_short_integration(self, trade_factory):
        """Test integration with mixed long and short positions."""
        trades = [
            trade_factory('ZS', 1200, 1210, side='long'),  # Long win
            trade_factory('ZS', 1210, 1205, side='short'),  # Short win
            trade_factory('ZS', 1205, 1200, side='long'),  # Long loss
            trade_factory('ZS', 1200, 1210, side='short'),  # Short loss
        ]

        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()

        assert result['total_trades'] == 4
        assert result['win_rate'] == 50.0
        assert result['winning_trades'] == 2
        assert result['losing_trades'] == 2

    def test_commission_impact_integration(self, trade_factory):
        """Test that commission is properly integrated through pipeline."""
        # Create a small profit trade
        trade = trade_factory('ZS', 1200.0, 1200.25)  # Small 0.25 cent move

        # Verify commission was applied
        assert trade['commission'] == 4.0
        assert trade['net_pnl'] > 0  # Should still be positive after commission

        # Verify summary handles commission correctly
        summary = SummaryMetrics([trade])
        result = summary.calculate_all_metrics()

        # Small profit should still register as winning trade
        assert result['winning_trades'] == 1

    def test_large_scale_performance(self, trades_factory):
        """
        Performance test with 10,000 trades.

        Verifies that metrics calculation scales efficiently with large datasets.
        Runs quickly (~0.1s) so included in default test suite.
        """
        # Create 10,000 trades
        trades = trades_factory.mixed(win_count=6000, loss_count=4000, symbol='ZS')

        # Measure time for summary metrics calculation
        start = time.perf_counter()
        summary = SummaryMetrics(trades)
        result = summary.calculate_all_metrics()
        end = time.perf_counter()

        # Check basic metrics
        assert result['total_trades'] == 10000
        assert result['win_rate'] == 60.0

        # Ensure performance is acceptable
        assert end - start < 2.0, "Performance test exceeded time limit"

    def test_memory_usage_performance(self, trades_factory):
        """
        Memory usage test with 10,000 trades.

        Verifies memory consumption stays within reasonable limits.
        Runs quickly so included in default test suite.
        """
        trades = trades_factory.mixed(win_count=6000, loss_count=4000, symbol='ZS')

        # Measure memory usage
        import tracemalloc
        tracemalloc.start()

        summary = SummaryMetrics(trades)
        summary.calculate_all_metrics()

        # Get current, peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

        # Check memory usage is within limits
        assert current_mb < 50.0, "Current memory usage exceeded limit"
        assert peak_mb < 100.0, "Peak memory usage exceeded limit"
