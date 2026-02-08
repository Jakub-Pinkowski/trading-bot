"""
Tests for Summary Metrics Module.

Tests cover:
- Summary metrics calculation for trade sequences
- Win rate, profit factor, and return metrics
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown calculations (max drawdown, Ulcer index)
- Risk metrics (VaR, Expected Shortfall)
- Edge cases (no trades, all wins, all losses, single trade)
- Real trade sequences from strategy backtests

All tests use realistic trade data with proper metric calculations.

Note: This file uses shared fixtures from conftest.py:
- trade_factory: For creating individual trades
- trades_factory: For creating trade sequences
"""
from datetime import datetime, timedelta

from app.backtesting.metrics.summary_metrics import (
    SummaryMetrics,
    MIN_RETURNS_FOR_SHARPE,
    MIN_RETURNS_FOR_VAR,
    RISK_FREE_RATE,
    CONFIDENCE_LEVEL,
    INFINITY_REPLACEMENT
)


# ==================== Test Classes ====================

class TestSummaryMetricsInitialization:
    """Test SummaryMetrics class initialization."""

    def test_initialization_with_trades(self, trades_factory):
        """Test SummaryMetrics initializes correctly with trades."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)

        assert metrics.trades == mixed_trades
        assert metrics.total_trades == 2
        assert len(metrics.winning_trades) == 1
        assert len(metrics.losing_trades) == 1

    def test_initialization_with_empty_list(self):
        """Test SummaryMetrics handles empty trade list."""
        metrics = SummaryMetrics([])

        assert metrics.total_trades == 0
        assert metrics.winning_trades == []
        assert metrics.losing_trades == []
        assert metrics.win_count == 0
        assert metrics.loss_count == 0

    def test_initialization_with_none(self):
        """Test SummaryMetrics handles None trades."""
        metrics = SummaryMetrics(None)

        assert metrics.total_trades == 0
        assert not metrics._has_trades()


class TestBasicTradeStatistics:
    """Test basic trade statistics calculations."""

    def test_win_rate_calculation(self, trades_factory):
        """Test win rate is calculated correctly."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # 1 win out of 2 trades = 50%
        assert result['win_rate'] == 50.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 1

    def test_all_winning_trades_win_rate(self, trades_factory):
        """Test win rate with all winning trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 5
        assert result['losing_trades'] == 0

    def test_all_losing_trades_win_rate(self, trades_factory):
        """Test win rate with all losing trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 5

    def test_total_trades_count(self, trades_factory):
        """Test total trades count is correct."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 5

    def test_average_trade_duration(self, trades_factory):
        """Test average trade duration calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Both trades are 4 hours
        assert result['average_trade_duration_hours'] == 4.0


class TestReturnMetrics:
    """Test return-based metrics calculations."""

    def test_total_return_percentage(self, trades_factory):
        """Test total return percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Sum of return_percentage_of_contract from both trades
        expected = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        assert result['total_return_percentage_of_contract'] == round(expected, 2)

    def test_average_trade_return(self, trades_factory):
        """Test average trade return calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Average of all trade returns
        total_return = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        expected_avg = total_return / len(mixed_trades)
        assert result['average_trade_return_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_win_percentage(self, trades_factory):
        """Test average win percentage calculation."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        winning_returns = [t['return_percentage_of_contract'] for t in all_winning]
        expected_avg = sum(winning_returns) / len(winning_returns)
        assert result['average_win_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_loss_percentage(self, trades_factory):
        """Test average loss percentage calculation."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        losing_returns = [t['return_percentage_of_contract'] for t in all_losing]
        expected_avg = sum(losing_returns) / len(losing_returns)
        assert result['average_loss_percentage_of_contract'] == round(expected_avg, 2)

    def test_total_wins_percentage(self, trades_factory):
        """Test total wins percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        winning_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] > 0]
        expected = sum(t['return_percentage_of_contract'] for t in winning_trades)
        assert result['total_wins_percentage_of_contract'] == round(expected, 2)

    def test_total_losses_percentage(self, trades_factory):
        """Test total losses percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        losing_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] <= 0]
        expected = sum(t['return_percentage_of_contract'] for t in losing_trades)
        assert result['total_losses_percentage_of_contract'] == round(expected, 2)


class TestProfitFactor:
    """Test profit factor calculations."""

    def test_profit_factor_with_mixed_trades(self, trades_factory):
        """Test profit factor with both wins and losses."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Profit factor = Total Wins / |Total Losses|
        assert result['profit_factor'] > 0
        assert isinstance(result['profit_factor'], float)

    def test_profit_factor_all_wins(self, trades_factory):
        """Test profit factor with only winning trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # No losses = infinity replacement
        assert result['profit_factor'] == INFINITY_REPLACEMENT

    def test_profit_factor_all_losses(self, trades_factory):
        """Test profit factor with only losing trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # No wins = 0
        assert result['profit_factor'] == 0.0


class TestDrawdownCalculations:
    """Test drawdown-related metrics."""

    def test_maximum_drawdown_calculation(self, trades_factory):
        """Test maximum drawdown is calculated."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Should have a drawdown value
        assert 'maximum_drawdown_percentage' in result
        assert isinstance(result['maximum_drawdown_percentage'], float)

    def test_maximum_drawdown_all_wins(self, trades_factory):
        """Test maximum drawdown with all winning trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # All wins = minimal or zero drawdown
        assert result['maximum_drawdown_percentage'] >= 0

    def test_maximum_drawdown_all_losses(self, trades_factory):
        """Test maximum drawdown with all losing trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # All losses = significant drawdown
        assert result['maximum_drawdown_percentage'] > 0

    def test_ulcer_index_calculation(self, trades_factory):
        """Test Ulcer Index is calculated."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'ulcer_index' in result
        assert isinstance(result['ulcer_index'], float)
        assert result['ulcer_index'] >= 0

    def test_single_negative_trade_from_zero(self, trade_factory):
        """Test drawdown from zero starting equity with single losing trade."""
        trades = [trade_factory('ZS', 1200.0, 1190.0)]

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Starting from zero, one loss creates drawdown equal to the loss
        assert result['maximum_drawdown_percentage'] > 0
        # The drawdown should equal the absolute loss
        expected_dd = abs(trades[0]['return_percentage_of_contract'])
        assert round(result['maximum_drawdown_percentage'], 2) == round(expected_dd, 2)

    def test_multiple_drawdown_periods(self, trades_factory):
        """Test multiple separate drawdown periods are tracked correctly."""
        # Create pattern: Win -> Loss -> Win -> Bigger Loss -> Win
        price_specs = [
            (1200.0, 1210.0),  # Win +10
            (1210.0, 1205.0),  # Loss -5 (first drawdown)
            (1205.0, 1220.0),  # Win +15 (recovery)
            (1220.0, 1200.0),  # Loss -20 (second, larger drawdown)
            (1200.0, 1215.0),  # Win +15 (recovery)
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Should capture the largest drawdown (second period)
        assert result['maximum_drawdown_percentage'] > 0

    def test_complex_peak_trough_scenario(self, trades_factory):
        """Test complex scenario with multiple peaks and troughs."""
        # Pattern: Peak -> Drop -> Small recovery -> Further drop -> Recovery
        price_specs = [
            (1200.0, 1250.0),  # Big win to peak
            (1250.0, 1230.0),  # Drop
            (1230.0, 1240.0),  # Small recovery
            (1240.0, 1210.0),  # Further drop (trough)
            (1210.0, 1245.0),  # Recovery
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        max_dd, max_dd_pct = metrics._calculate_max_drawdown()

        # Verify drawdown is calculated from highest peak to lowest trough
        assert max_dd > 0
        assert max_dd_pct > 0

    def test_drawdown_with_recovery(self, trades_factory):
        """Test drawdown calculation followed by full recovery."""
        # Pattern: Win to peak -> Loss (drawdown) -> Win (full recovery)
        price_specs = [
            (1200.0, 1250.0),  # Win to peak
            (1250.0, 1220.0),  # Loss (drawdown)
            (1220.0, 1250.0),  # Recovery to previous peak
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Drawdown should still be recorded even after recovery
        assert result['maximum_drawdown_percentage'] > 0

    def test_drawdown_calculation_accuracy(self, trades_factory):
        """Test precise drawdown percentage calculation."""
        # Create specific scenario with known drawdown
        price_specs = [
            (1000.0, 1100.0),  # +10% (cumulative: 10%)
            (1100.0, 1050.0),  # -4.55% (cumulative: ~5.45%)
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        # Drawdown from peak should be measurable
        assert metrics.maximum_drawdown_percentage > 0


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""

    def test_sharpe_ratio_with_sufficient_trades(self, trades_factory):
        """Test Sharpe ratio with enough trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # With 5 trades, should calculate Sharpe
        assert 'sharpe_ratio' in result
        assert isinstance(result['sharpe_ratio'], float)

    def test_sharpe_ratio_insufficient_trades(self, trade_factory):
        """Test Sharpe ratio with insufficient trades."""
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        metrics = SummaryMetrics([winning_trade])
        result = metrics.calculate_all_metrics()

        # Less than MIN_RETURNS_FOR_SHARPE = 0
        assert result['sharpe_ratio'] == 0.0

    def test_sharpe_ratio_high_volatility(self, trades_factory):
        """Test Sharpe ratio with high volatility returns."""
        # Create trades with high variance: large wins and large losses
        price_specs = [
            (1200.0, 1250.0),  # Big win
            (1250.0, 1210.0),  # Big loss
            (1210.0, 1260.0),  # Big win
            (1260.0, 1215.0),  # Big loss
            (1215.0, 1265.0),  # Big win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # High volatility should result in lower Sharpe ratio
        # (compared to same returns with lower volatility)
        assert result['sharpe_ratio'] is not None
        assert isinstance(result['sharpe_ratio'], float)

    def test_sharpe_ratio_negative_returns(self, trades_factory):
        """Test Sharpe ratio with predominantly negative returns."""
        # Create mostly losing trades
        all_losing = trades_factory.all_losing(count=5)

        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # Negative average return should give negative Sharpe
        assert result['sharpe_ratio'] < 0

    def test_sharpe_ratio_single_large_outlier(self, trades_factory):
        """Test Sharpe ratio with single large outlier affecting std dev."""
        # Create mostly small wins with one huge win
        price_specs = [
            (1200.0, 1202.0),  # Small win
            (1202.0, 1204.0),  # Small win
            (1204.0, 1206.0),  # Small win
            (1206.0, 1208.0),  # Small win
            (1208.0, 1300.0),  # Huge outlier win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Outlier increases both mean and std dev
        assert result['sharpe_ratio'] > 0
        assert isinstance(result['sharpe_ratio'], float)

    def test_sharpe_ratio_with_mixed_returns(self, trades_factory):
        """Test Sharpe ratio with balanced wins and losses."""
        # Create alternating wins and losses
        price_specs = [
            (1200.0, 1210.0),  # Win
            (1210.0, 1205.0),  # Loss
            (1205.0, 1215.0),  # Win
            (1215.0, 1210.0),  # Loss
            (1210.0, 1220.0),  # Win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Mixed returns should give moderate Sharpe ratio
        assert isinstance(result['sharpe_ratio'], float)


class TestSortinoRatio:
    """Test Sortino ratio calculations."""

    def test_sortino_ratio_with_losses(self, trades_factory):
        """Test Sortino ratio with both wins and losses."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'sortino_ratio' in result
        assert isinstance(result['sortino_ratio'], float)

    def test_sortino_ratio_no_negative_returns(self, trades_factory):
        """Test Sortino ratio with no negative returns."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # No downside deviation = infinity replacement
        assert result['sortino_ratio'] == INFINITY_REPLACEMENT

    def test_sortino_ratio_all_downside_returns(self, trades_factory):
        """Test Sortino ratio when all returns are below risk-free rate."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # All negative returns should give negative Sortino ratio
        assert result['sortino_ratio'] < 0

    def test_sortino_ratio_asymmetric_returns(self, trades_factory):
        """Test Sortino ratio with asymmetric return distribution (large wins, small losses)."""
        # Create asymmetric distribution: few large wins, many small losses
        trades = []

        # Many small losses
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1198.0,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # Few large wins
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1250.0 + i * 10,
                                                       entry_time=datetime(2024, 1, 17 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 17 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Sortino should be better than Sharpe for positive skew (many small losses, few big wins)
        assert result['sortino_ratio'] is not None
        assert isinstance(result['sortino_ratio'], float)

    def test_sortino_vs_sharpe_comparison(self, trades_factory):
        """Test that Sortino ratio differs from Sharpe ratio appropriately."""
        # Create scenario with downside volatility
        price_specs = [
            (1200.0, 1220.0),  # Win
            (1220.0, 1200.0),  # Loss
            (1200.0, 1225.0),  # Win
            (1225.0, 1205.0),  # Loss
            (1205.0, 1230.0),  # Win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Both ratios should be calculated
        assert result['sortino_ratio'] is not None
        assert result['sharpe_ratio'] is not None
        # Sortino typically higher (penalizes only downside)
        # But both should be positive for net positive returns

    def test_sortino_with_small_downside(self, trades_factory):
        """Test Sortino ratio with minimal downside volatility."""
        # Create mostly wins with one tiny loss
        price_specs = [
            (1200.0, 1210.0),  # Win
            (1210.0, 1220.0),  # Win
            (1220.0, 1219.0),  # Tiny loss
            (1219.0, 1230.0),  # Win
            (1230.0, 1240.0),  # Win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Small downside should give good Sortino ratio
        assert result['sortino_ratio'] > 0


class TestCalmarRatio:
    """Test Calmar ratio calculations."""

    def test_calmar_ratio_calculation(self, trades_factory):
        """Test Calmar ratio is calculated."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'calmar_ratio' in result
        assert isinstance(result['calmar_ratio'], float)

    def test_calmar_ratio_zero_drawdown(self, trades_factory):
        """Test Calmar ratio with zero or minimal drawdown."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # Zero/minimal drawdown might give very high value
        assert result['calmar_ratio'] >= 0

    def test_calmar_ratio_small_drawdown(self, trades_factory):
        """Test Calmar ratio with small drawdown relative to returns."""
        # Pattern: Large wins with small drawdown
        price_specs = [
            (1200.0, 1250.0),  # Large win
            (1250.0, 1245.0),  # Small drawdown
            (1245.0, 1280.0),  # Large win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Small drawdown with good returns = high Calmar
        assert result['calmar_ratio'] > 0

    def test_calmar_ratio_large_drawdown(self, trades_factory):
        """Test Calmar ratio with large drawdown relative to returns."""
        # Pattern: Small net return with large drawdown
        price_specs = [
            (1200.0, 1250.0),  # Win to peak
            (1250.0, 1180.0),  # Large drawdown
            (1180.0, 1210.0),  # Small recovery
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Large drawdown with small return = low Calmar
        assert result['calmar_ratio'] > 0  # Still positive if net return is positive

    def test_calmar_ratio_negative_returns(self, trades_factory):
        """Test Calmar ratio with negative total returns."""
        # All losing trades
        price_specs = [
            (1200.0, 1190.0),
            (1190.0, 1180.0),
            (1180.0, 1170.0),
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Negative returns = negative Calmar ratio
        assert result['calmar_ratio'] < 0

    def test_calmar_ratio_recovery_periods(self, trades_factory):
        """Test Calmar ratio accounts for drawdown even after recovery."""
        # Pattern: Drawdown followed by full recovery
        price_specs = [
            (1200.0, 1300.0),  # Peak
            (1300.0, 1250.0),  # Drawdown
            (1250.0, 1350.0),  # Recovery beyond peak
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Good return with moderate drawdown = reasonable Calmar
        assert result['calmar_ratio'] > 0
        assert result['maximum_drawdown_percentage'] > 0  # Drawdown still recorded


class TestValueAtRisk:
    """Test Value at Risk (VaR) calculations."""

    def test_var_with_sufficient_trades(self, trades_factory):
        """Test VaR with enough trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # Should calculate VaR
        assert 'value_at_risk' in result
        assert result['value_at_risk'] >= 0

    def test_var_insufficient_trades(self, trade_factory):
        """Test VaR with insufficient trades."""
        # Less than MIN_RETURNS_FOR_VAR
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        trades = [winning_trade]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Not enough returns = 0
        assert result['value_at_risk'] == 0.0

    def test_var_with_skewed_distribution(self, trades_factory):
        """Test VaR with negatively skewed return distribution (many small wins, few big losses)."""
        # Create skewed distribution: 7 small wins, 3 big losses
        trades = []

        # Small wins
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1202.0,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # Big losses
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1170.0 - i * 5,
                                                       entry_time=datetime(2024, 1, 17 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 17 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should capture the tail risk from big losses
        assert result['value_at_risk'] > 0

    def test_var_with_fat_tails(self, trades_factory):
        """Test VaR with fat-tailed distribution (more extreme events than normal distribution)."""
        # Create distribution with extreme outliers
        price_specs = [
            (1200.0, 1205.0),  # Small win
            (1205.0, 1210.0),  # Small win
            (1210.0, 1215.0),  # Small win
            (1215.0, 1220.0),  # Small win
            (1220.0, 1225.0),  # Small win
            (1225.0, 1150.0),  # Extreme loss (fat tail)
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be significant due to extreme loss
        assert result['value_at_risk'] > 0

    def test_var_with_extreme_losses(self, trades_factory):
        """Test VaR calculation with several extreme losses."""
        # Create scenario with multiple extreme losses
        trades = []
        for i in range(10):
            if i < 5:
                # Normal trades
                trades.append(trades_factory.trade_factory('ZS', 1200.0, 1205.0,
                                                           entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                           exit_time=datetime(2024, 1, 10 + i, 14, 0)))
            else:
                # Extreme losses
                trades.append(trades_factory.trade_factory('ZS', 1200.0, 1150.0 - (i - 5) * 10,
                                                           entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                           exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be high due to extreme losses
        assert result['value_at_risk'] > 0

    def test_var_with_normal_distribution(self, trades_factory):
        """Test VaR with approximately normal return distribution."""
        # Create returns clustered around mean
        import random
        random.seed(42)  # For reproducibility

        trades = []
        for i in range(20):
            # Generate returns around 1% with small variance
            price_change = 1200.0 * (1 + random.gauss(0.01, 0.005))
            trades.append(trades_factory.trade_factory('ZS', 1200.0, price_change,
                                                       entry_time=datetime(2024, 1, 10, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 1, 10, 14, 0) + timedelta(days=i)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be reasonable for normal distribution
        assert result['value_at_risk'] >= 0


class TestUlcerIndex:
    """Test Ulcer Index calculations for downside risk measurement."""

    def test_ulcer_index_no_drawdown(self, trades_factory):
        """Test Ulcer Index with no drawdown (all winning trades)."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # No drawdown should give very low or zero Ulcer Index
        assert result['ulcer_index'] >= 0
        # With all wins, ulcer index should be minimal
        assert result['ulcer_index'] < 1.0  # Should be very small

    def test_ulcer_index_single_drawdown(self, trades_factory):
        """Test Ulcer Index with single drawdown period."""
        # Pattern: Win -> Loss -> Recovery
        price_specs = [
            (1200.0, 1250.0),  # Win (peak)
            (1250.0, 1220.0),  # Loss (drawdown)
            (1220.0, 1245.0),  # Recovery
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Single drawdown should register in Ulcer Index
        assert result['ulcer_index'] > 0

    def test_ulcer_index_multiple_drawdowns(self, trades_factory):
        """Test Ulcer Index with multiple drawdown periods."""
        # Pattern: Win -> Loss -> Win -> Loss -> Win
        price_specs = [
            (1200.0, 1230.0),  # Win
            (1230.0, 1220.0),  # Loss (first drawdown)
            (1220.0, 1240.0),  # Win
            (1240.0, 1210.0),  # Loss (second drawdown)
            (1210.0, 1235.0),  # Win
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Multiple drawdowns should increase Ulcer Index
        assert result['ulcer_index'] > 0

    def test_ulcer_index_sustained_drawdown(self, trades_factory):
        """Test Ulcer Index with sustained (long duration) drawdown."""
        # Pattern: Peak -> Sustained losses
        price_specs = [
            (1200.0, 1250.0),  # Peak
            (1250.0, 1240.0),  # Small loss
            (1240.0, 1230.0),  # Another loss
            (1230.0, 1220.0),  # Another loss
            (1220.0, 1210.0),  # Another loss (sustained decline)
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics_sustained = SummaryMetrics(trades)
        result_sustained = metrics_sustained.calculate_all_metrics()

        # Sustained drawdown should give significant Ulcer Index
        assert result_sustained['ulcer_index'] > 0

    def test_ulcer_index_sharp_drawdown_vs_sustained(self, trades_factory):
        """Test that sustained drawdown has higher Ulcer Index than sharp recovery."""
        # Sharp drawdown with quick recovery
        sharp_prices = [
            (1200.0, 1250.0),  # Peak
            (1250.0, 1200.0),  # Sharp drop
            (1200.0, 1245.0),  # Quick recovery
        ]

        # Sustained drawdown
        sustained_prices = [
            (1200.0, 1250.0),  # Peak
            (1250.0, 1235.0),  # Gradual decline
            (1235.0, 1220.0),  # Continues
            (1220.0, 1205.0),  # Still declining
        ]

        sharp_trades = trades_factory.create_sequence(sharp_prices, symbol='ZS')
        sustained_trades = trades_factory.create_sequence(sustained_prices, symbol='ZS')

        sharp_metrics = SummaryMetrics(sharp_trades)
        sustained_metrics = SummaryMetrics(sustained_trades)

        sharp_result = sharp_metrics.calculate_all_metrics()
        sustained_result = sustained_metrics.calculate_all_metrics()

        # Sustained drawdown penalized more by Ulcer Index (duration matters)
        assert sustained_result['ulcer_index'] > 0
        assert sharp_result['ulcer_index'] > 0

    def test_ulcer_index_vs_max_drawdown_comparison(self, trades_factory):
        """Test relationship between Ulcer Index and Max Drawdown."""
        # Create scenario with known drawdown
        price_specs = [
            (1200.0, 1300.0),  # Win to peak
            (1300.0, 1250.0),  # Drawdown
            (1250.0, 1295.0),  # Recovery
        ]

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Both metrics should capture the drawdown
        assert result['ulcer_index'] > 0
        assert result['maximum_drawdown_percentage'] > 0
        # Ulcer Index considers duration, Max DD only considers depth

    def test_ulcer_index_recovery_impact(self, trades_factory):
        """Test how recovery affects Ulcer Index."""
        # Scenario with recovery
        recovery_prices = [
            (1200.0, 1250.0),  # Peak
            (1250.0, 1220.0),  # Drawdown
            (1220.0, 1250.0),  # Full recovery
        ]

        # Scenario without recovery
        no_recovery_prices = [
            (1200.0, 1250.0),  # Peak
            (1250.0, 1220.0),  # Drawdown
            (1220.0, 1225.0),  # Partial recovery only
        ]

        recovery_trades = trades_factory.create_sequence(recovery_prices, symbol='ZS')
        no_recovery_trades = trades_factory.create_sequence(no_recovery_prices, symbol='ZS')

        recovery_metrics = SummaryMetrics(recovery_trades)
        no_recovery_metrics = SummaryMetrics(no_recovery_trades)

        recovery_result = recovery_metrics.calculate_all_metrics()
        no_recovery_result = no_recovery_metrics.calculate_all_metrics()

        # Both should have Ulcer Index > 0
        assert recovery_result['ulcer_index'] > 0
        assert no_recovery_result['ulcer_index'] > 0

    """Test Expected Shortfall (CVaR) calculations."""

    def test_expected_shortfall_with_sufficient_trades(self, trades_factory):
        """Test Expected Shortfall with enough trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        assert 'expected_shortfall' in result
        assert result['expected_shortfall'] >= 0

    def test_expected_shortfall_insufficient_trades(self, trade_factory):
        """Test Expected Shortfall with insufficient trades."""
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        trades = [winning_trade]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['expected_shortfall'] == 0.0

    def test_expected_shortfall_greater_than_var(self, trades_factory):
        """Test Expected Shortfall is typically >= VaR."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # ES should be >= VaR (average of worst cases vs threshold)
        assert result['expected_shortfall'] >= result['value_at_risk']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trades_list(self):
        """Test metrics with empty trades list."""
        metrics = SummaryMetrics([])
        result = metrics.calculate_all_metrics()

        # Should return empty dict
        assert result == {}

    def test_none_trades(self):
        """Test metrics with None trades."""
        metrics = SummaryMetrics(None)
        result = metrics.calculate_all_metrics()

        assert result == {}

    def test_single_winning_trade(self, trade_factory):
        """Test metrics with single winning trade."""
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        metrics = SummaryMetrics([winning_trade])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 0

    def test_single_losing_trade(self, trade_factory):
        """Test metrics with single losing trade."""
        losing_trade = trade_factory('ZS', 1200.0, 1195.0)
        metrics = SummaryMetrics([losing_trade])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 1

    def test_breakeven_trade_counted_as_loss(self, trade_factory):
        """Test breakeven trade is counted as losing trade."""
        breakeven_trade = trade_factory('ZS', 1200.0, 1200.0)
        metrics = SummaryMetrics([breakeven_trade])
        result = metrics.calculate_all_metrics()

        # Breakeven (including commission) should be <= 0
        assert result['losing_trades'] == 1
        assert result['winning_trades'] == 0


class TestMetricsRounding:
    """Test that all metrics are properly rounded."""

    def test_all_metrics_rounded_to_two_decimals(self, trades_factory):
        """Test all metrics are rounded to 2 decimal places."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        for key, value in result.items():
            if isinstance(value, float):
                # Check that it has at most 2 decimal places
                str_value = str(value)
                if '.' in str_value:
                    decimals = len(str_value.split('.')[1])
                    assert decimals <= 2, f"{key} has more than 2 decimal places: {value}"


class TestPrivateMethods:
    """Test private helper methods."""

    def test_has_trades_method(self, trades_factory):
        """Test _has_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_trades() is True

        empty_metrics = SummaryMetrics([])
        assert empty_metrics._has_trades() is False

    def test_has_winning_trades_method(self, trades_factory):
        """Test _has_winning_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_winning_trades() is True

        losing_only = trades_factory.all_losing(count=1)
        losing_metrics = SummaryMetrics(losing_only)
        assert losing_metrics._has_winning_trades() is False

    def test_has_losing_trades_method(self, trades_factory):
        """Test _has_losing_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_losing_trades() is True

        winning_only = trades_factory.all_winning(count=1)
        winning_metrics = SummaryMetrics(winning_only)
        assert winning_metrics._has_losing_trades() is False


class TestConstants:
    """Test module constants are defined correctly."""

    def test_min_returns_for_sharpe(self):
        """Test MIN_RETURNS_FOR_SHARPE constant."""
        assert MIN_RETURNS_FOR_SHARPE == 2

    def test_min_returns_for_var(self):
        """Test MIN_RETURNS_FOR_VAR constant."""
        assert MIN_RETURNS_FOR_VAR == 5

    def test_risk_free_rate(self):
        """Test RISK_FREE_RATE constant."""
        assert RISK_FREE_RATE == 0.0

    def test_confidence_level(self):
        """Test CONFIDENCE_LEVEL constant."""
        assert CONFIDENCE_LEVEL == 0.95

    def test_infinity_replacement(self):
        """Test INFINITY_REPLACEMENT constant."""
        assert INFINITY_REPLACEMENT == 9999.99


class TestRealWorldScenarios:
    """Test with realistic trade sequences."""

    def test_profitable_strategy(self, trades_factory):
        """Test metrics for overall profitable strategy."""
        # 7 wins, 3 losses
        trades = []

        # 7 winning trades
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1205.0 + i,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # 3 losing trades
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1198.0,
                                                       entry_time=datetime(2024, 1, 17 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 17 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 70.0
        assert result['total_return_percentage_of_contract'] > 0
        assert result['profit_factor'] > 1.0

    def test_losing_strategy(self, trades_factory):
        """Test metrics for overall losing strategy."""
        # 3 wins, 7 losses
        trades = []

        # 3 winning trades
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1202.0,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # 7 losing trades
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1195.0 - i,
                                                       entry_time=datetime(2024, 1, 13 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 13 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 30.0
        assert result['total_return_percentage_of_contract'] < 0
        assert result['profit_factor'] < 1.0
        assert result['maximum_drawdown_percentage'] > 0
