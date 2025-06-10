import io
import sys
from datetime import datetime, timedelta

from app.backtesting.summary_metrics import calculate_max_drawdown, calculate_summary_metrics, print_summary_metrics


# Helper function to create a sample trade
def create_sample_trade(
    net_pnl=100.0,
    return_percentage=1.0,
    duration_hours=24,
    margin_requirement=10000.0,
    commission=4.0
):
    entry_time = datetime.now()
    exit_time = entry_time + timedelta(hours=duration_hours)

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'duration': timedelta(hours=duration_hours),
        'duration_hours': duration_hours,
        'net_pnl': net_pnl,
        'gross_pnl': net_pnl + commission,
        'return_percentage_of_margin': return_percentage,
        'margin_requirement': margin_requirement,
        'commission': commission
    }


class TestCalculateMaxDrawdown:
    """Tests for the calculate_max_drawdown function."""

    def test_empty_trades_list(self):
        """Test calculation of max drawdown with an empty trades list."""
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([])
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_single_trade_positive(self):
        """Test calculation of max drawdown with a single positive trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0)
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([trade])
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_single_trade_negative(self):
        """Test calculation of max drawdown with a single negative trade."""
        trade = create_sample_trade(net_pnl=-100.0, return_percentage=-1.0)
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([trade])
        # For a single negative trade, there's no peak to draw down from
        # The function initializes peak to the first value, so drawdown is 0
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_multiple_trades_no_drawdown(self):
        """Test calculation of max drawdown with multiple trades but no drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_multiple_trades_with_drawdown(self):
        """Test calculation of max drawdown with multiple trades with drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 150.0
        assert max_drawdown_pct == 1.5

    def test_complex_drawdown_scenario(self):
        """Test calculation of max drawdown with a complex sequence of trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 300, 3%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 250, 2.5%
            create_sample_trade(net_pnl=-300.0, return_percentage=-3.0),  # Cumulative: -50, -0.5%
            create_sample_trade(net_pnl=150.0, return_percentage=1.5)  # Cumulative: 100, 1%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 350.0  # From peak of 300 to low of -50
        assert max_drawdown_pct == 3.5  # From peak of 3% to low of -0.5%

    def test_drawdown_with_recovery(self):
        """Test calculation of max drawdown with a drawdown followed by recovery."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 50, 0.5%
            create_sample_trade(net_pnl=-60.0, return_percentage=-0.6),  # Cumulative: -10, -0.1%
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 190, 1.9%
            create_sample_trade(net_pnl=100.0, return_percentage=1.0)  # Cumulative: 290, 2.9%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 110.0  # From peak of 100 to low of -10
        assert max_drawdown_pct == 1.1  # From peak of 1% to low of -0.1%

    def test_multiple_drawdowns(self):
        """Test calculation of max drawdown with multiple drawdowns."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 50, 0.5% (Drawdown 1)
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 250, 2.5%
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),  # Cumulative: 100, 1% (Drawdown 2)
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)  # Cumulative: 400, 4%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 150.0  # The larger of the two drawdowns
        assert max_drawdown_pct == 1.5  # The larger of the two drawdowns in percentage


class TestCalculateSummaryMetrics:
    """Tests for the calculate_summary_metrics function."""

    def test_empty_trades_list(self):
        """Test calculation of summary metrics with an empty trades list."""
        summary = calculate_summary_metrics([])
        assert summary == {}

    def test_single_winning_trade(self):
        """Test calculation of summary metrics with a single winning trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0, margin_requirement=10000.0, commission=4.0)
        summary = calculate_summary_metrics([trade])

        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 100.0
        assert summary['total_net_pnl'] == 100.0
        assert summary['avg_trade_net_pnl'] == 100.0
        assert summary['total_return_percentage_of_margin'] == 1.0
        assert summary['profit_factor'] == float('inf')  # No losing trades

    def test_single_losing_trade(self):
        """Test calculation of summary metrics with a single losing trade."""
        trade = create_sample_trade(net_pnl=-100.0, return_percentage=-1.0, margin_requirement=10000.0, commission=4.0)
        summary = calculate_summary_metrics([trade])

        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 0
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == 0.0
        assert summary['total_net_pnl'] == -100.0
        assert summary['avg_trade_net_pnl'] == -100.0
        assert summary['total_return_percentage_of_margin'] == -1.0
        assert summary['profit_factor'] == 0.0  # No winning trades

    def test_multiple_trades(self):
        """Test calculation of summary metrics with multiple trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 2
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == round((2 / 3) * 100, 2)
        assert summary['total_net_pnl'] == 250.0
        assert summary['avg_trade_net_pnl'] == round(250.0 / 3, 2)
        assert summary['total_return_percentage_of_margin'] == 2.5
        assert summary['avg_win_net'] == 150.0  # (100 + 200) / 2
        assert summary['avg_loss_net'] == -50.0
        assert summary['profit_factor'] == round(300.0 / 50.0, 2)  # (100 + 200) / 50

    def test_all_winning_trades(self):
        """Test calculation of summary metrics with all winning trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 3
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 100.0
        assert summary['total_net_pnl'] == 600.0
        assert summary['profit_factor'] == float('inf')  # No losing trades

    def test_all_losing_trades(self):
        """Test calculation of summary metrics with all losing trades."""
        trades = [
            create_sample_trade(net_pnl=-100.0, return_percentage=-1.0),
            create_sample_trade(net_pnl=-200.0, return_percentage=-2.0),
            create_sample_trade(net_pnl=-300.0, return_percentage=-3.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 0
        assert summary['losing_trades'] == 3
        assert summary['win_rate'] == 0.0
        assert summary['total_net_pnl'] == -600.0
        assert summary['profit_factor'] == 0.0  # No winning trades

    def test_commission_metrics(self):
        """Test calculation of commission metrics."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0, commission=5.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, commission=5.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0, commission=5.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_commission_paid'] == 15.0
        assert summary['commission_percentage_of_margin'] == round((15.0 / 30000.0) * 100, 2)  # 3 trades * 10000 margin

    def test_duration_metrics(self):
        """Test calculation of duration metrics."""
        trades = [
            create_sample_trade(duration_hours=12),
            create_sample_trade(duration_hours=24),
            create_sample_trade(duration_hours=36)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['avg_trade_duration_hours'] == 24.0  # (12 + 24 + 36) / 3

    def test_risk_metrics(self):
        """Test calculation of risk metrics."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        # Max drawdown should be calculated correctly
        assert summary['max_drawdown'] > 0
        assert summary['maximum_drawdown_percentage'] > 0

        # Return to drawdown ratio should be calculated correctly
        assert summary['return_to_drawdown_ratio'] == round(2.5 / summary['maximum_drawdown_percentage'], 2)

    def test_zero_drawdown(self):
        """Test calculation of summary metrics with zero drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['max_drawdown'] == 0
        assert summary['maximum_drawdown_percentage'] == 0
        assert summary['return_to_drawdown_ratio'] == float('inf')  # Division by zero

    def test_margin_metrics(self):
        """Test calculation of margin metrics."""
        trades = [
            create_sample_trade(margin_requirement=10000.0),
            create_sample_trade(margin_requirement=20000.0),
            create_sample_trade(margin_requirement=30000.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_margin_used'] == 60000.0
        assert summary['avg_margin_requirement'] == 20000.0


class TestPrintSummaryMetrics:
    """Tests for the print_summary_metrics function."""

    def test_print_positive_summary(self):
        """Test printing of a positive summary."""
        summary = {
            'total_trades': 10,
            'winning_trades': 7,
            'losing_trades': 3,
            'win_rate': 70.0,
            'avg_trade_duration_hours': 24.0,
            'total_margin_used': 100000.0,
            'avg_margin_requirement': 10000.0,
            'total_net_pnl': 5000.0,
            'avg_trade_net_pnl': 500.0,
            'avg_win_net': 1000.0,
            'avg_loss_net': -500.0,
            'total_return_percentage_of_margin': 5.0,
            'average_trade_return_percentage_of_margin': 0.5,
            'average_win_percentage_of_margin': 1.0,
            'average_loss_percentage_of_margin': -0.5,
            'total_commission_paid': 40.0,
            'commission_percentage_of_margin': 0.04,
            'profit_factor': 3.5,
            'max_drawdown': 1000.0,
            'maximum_drawdown_percentage': 1.0,
            'return_to_drawdown_ratio': 5.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the summary metrics
        print_summary_metrics(summary)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "SUMMARY METRICS" in output
        assert "Total Trades: 10" in output
        assert "Winning Trades: 7 (70.0%)" in output
        assert "Losing Trades: 3" in output
        assert "Total Net PnL: $5,000.00" in output
        assert "Profit Factor: 3.5" in output
        assert "Max Drawdown: $1,000.00" in output
        assert "Maximum Drawdown Percentage: 1.0%" in output
        assert "Return to Drawdown Ratio: 5.0" in output

    def test_print_negative_summary(self):
        """Test printing of a negative summary."""
        summary = {
            'total_trades': 10,
            'winning_trades': 3,
            'losing_trades': 7,
            'win_rate': 30.0,
            'avg_trade_duration_hours': 24.0,
            'total_margin_used': 100000.0,
            'avg_margin_requirement': 10000.0,
            'total_net_pnl': -5000.0,
            'avg_trade_net_pnl': -500.0,
            'avg_win_net': 500.0,
            'avg_loss_net': -1000.0,
            'total_return_percentage_of_margin': -5.0,
            'average_trade_return_percentage_of_margin': -0.5,
            'average_win_percentage_of_margin': 0.5,
            'average_loss_percentage_of_margin': -1.0,
            'total_commission_paid': 40.0,
            'commission_percentage_of_margin': 0.04,
            'profit_factor': 0.3,
            'max_drawdown': 5000.0,
            'maximum_drawdown_percentage': 5.0,
            'return_to_drawdown_ratio': -1.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the summary metrics
        print_summary_metrics(summary)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "SUMMARY METRICS" in output
        assert "Total Trades: 10" in output
        assert "Winning Trades: 3 (30.0%)" in output
        assert "Losing Trades: 7" in output
        assert "Total Net PnL: $-5,000.00" in output
        assert "Profit Factor: 0.3" in output
        assert "Max Drawdown: $5,000.00" in output
        assert "Maximum Drawdown Percentage: 5.0%" in output
        assert "Return to Drawdown Ratio: -1.0" in output
