import numpy as np

from app.utils.logger import get_logger
from app.utils.math_utils import safe_average, safe_divide

logger = get_logger('backtesting/summary_metrics')

# Metrics Calculation Constants
MIN_RETURNS_FOR_SHARPE = 2  # Minimum returns needed to calculate the Sharpe ratio (need at least 2 for std dev)
MIN_RETURNS_FOR_VAR = 5  # Minimum returns needed for Value at Risk and Expected Shortfall calculations


class SummaryMetrics:
    """Class for calculating summary metrics for a list of trades."""

    def __init__(self, trades):
        """Initialize with a list of trades and perform initial calculations."""
        self.trades = trades
        self.winning_trades = []
        self.losing_trades = []
        self.win_count = 0
        self.loss_count = 0
        self.win_rate = 0
        self.total_trades = len(trades) if trades else 0

        # Initialize common calculations if trades exist
        if self._has_trades():
            self._initialize_calculations()

    def calculate_all_metrics(self):
        """Calculate all summary metrics and return as a dictionary."""
        if not self._has_trades():
            logger.error('No trades provided to calculate_all_metrics')
            return {}

        # ===== BASIC TRADE STATISTICS =====
        win_rate = self.win_rate
        win_count = self.win_count
        loss_count = self.loss_count
        avg_duration_hours = safe_average(self.durations, self.total_trades)

        # ===== NORMALIZED METRICS (PERCENTAGES) =====
        total_return_percentage_of_margin = self.total_return
        average_trade_return_percentage_of_margin = safe_average([self.total_return], self.total_trades)
        average_win_percentage_of_margin = self._calculate_average_win_percentage_of_margin()
        average_loss_percentage_of_margin = self._calculate_average_loss_percentage_of_margin()
        commission_percentage_of_margin = self._calculate_commission_percentage_of_margin()

        # Contract-based metrics
        total_return_percentage_of_contract = self.total_return_contract
        average_trade_return_percentage_of_contract = safe_average([self.total_return_contract], self.total_trades)

        # Calculate total wins and losses for aggregation
        total_wins_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in self.winning_trades)
        total_losses_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in self.losing_trades)

        # ===== RISK METRICS =====
        profit_factor = self._calculate_profit_factor()
        max_drawdown, maximum_drawdown_percentage = self.max_drawdown, self.maximum_drawdown_percentage
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio()
        value_at_risk = self._calculate_value_at_risk()
        expected_shortfall = self._calculate_expected_shortfall()
        ulcer_index = self._calculate_ulcer_index()

        return {
            # Basic info
            'total_trades': self.total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': round(win_rate, 2),
            'avg_trade_duration_hours': round(avg_duration_hours, 2),

            # Percentage-based metrics
            'total_return_percentage_of_margin': round(total_return_percentage_of_margin, 2),
            'average_trade_return_percentage_of_margin': round(average_trade_return_percentage_of_margin, 2),
            'average_win_percentage_of_margin': round(average_win_percentage_of_margin, 2),
            'average_loss_percentage_of_margin': round(average_loss_percentage_of_margin, 2),
            'commission_percentage_of_margin': round(commission_percentage_of_margin, 2),
            'total_wins_percentage_of_margin': round(total_wins_percentage_of_margin, 2),
            'total_losses_percentage_of_margin': round(total_losses_percentage_of_margin, 2),

            # Contract-based metrics
            'total_return_percentage_of_contract': round(total_return_percentage_of_contract, 2),
            'average_trade_return_percentage_of_contract': round(average_trade_return_percentage_of_contract, 2),

            # Risk metrics
            'profit_factor': round(profit_factor, 2),
            'maximum_drawdown_percentage': maximum_drawdown_percentage,
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'value_at_risk': round(value_at_risk, 2),
            'expected_shortfall': round(expected_shortfall, 2),
            'ulcer_index': round(ulcer_index, 2),
        }

    @staticmethod
    def print_summary_metrics(summary):
        """Print summary metrics in a formatted way."""
        green = '\033[92m'
        red = '\033[91m'
        reset = '\033[0m'

        average_trade_return_percentage = summary['average_trade_return_percentage_of_margin']
        if average_trade_return_percentage > 0:
            color = green
        elif average_trade_return_percentage < 0:
            color = red
        else:
            color = reset  # default terminal color

        print('\n====== SUMMARY METRICS ======')

        # ===== BASIC TRADE STATISTICS =====
        print('\n--- BASIC TRADE STATISTICS ---')
        print(f'Total Trades: {summary["total_trades"]}')
        print(f'Winning Trades: {summary["winning_trades"]} ({summary["win_rate"]}%)')
        print(f'Losing Trades: {summary["losing_trades"]}')
        print(f'Avg Trade Duration: {summary["avg_trade_duration_hours"]} hours')

        # ===== PERCENTAGE-BASED METRICS =====
        print('\n--- PERCENTAGE-BASED METRICS ---')
        if summary["total_return_percentage_of_margin"] == 0:
            print(f'Total Return Percentage of Margin: 0.0%')
        else:
            print(f'Total Return Percentage of Margin: {color}{summary["total_return_percentage_of_margin"]}%{reset}')
        if average_trade_return_percentage == 0:
            print(f'Average Trade Return Percentage of Margin: 0.0%')
        else:
            print(f'Average Trade Return Percentage of Margin: {color}{average_trade_return_percentage}%{reset}')
        print(f'Average Win Percentage of Margin: {green}{summary["average_win_percentage_of_margin"]}%{reset}')
        print(f'Average Loss Percentage of Margin: {red}{summary["average_loss_percentage_of_margin"]}%{reset}')
        print(f'Commission Percentage of Margin: {summary["commission_percentage_of_margin"]}%')

        # Contract-based metrics
        print('\n--- CONTRACT-BASED METRICS ---')
        print(f'Total Return Percentage of Contract: {color}{summary["total_return_percentage_of_contract"]}%{reset}')
        print(f'Average Trade Return Percentage of Contract: {color}{summary["average_trade_return_percentage_of_contract"]}%{reset}')

        # ===== RISK METRICS =====
        print('\n--- RISK METRICS ---')
        print(f'Profit Factor: {summary.get("profit_factor", 0)}')
        print(f'Maximum Drawdown Percentage: {summary.get("maximum_drawdown_percentage", 0)}%')
        print(f'Calmar Ratio: {summary.get("calmar_ratio", 0)}')
        print(f'Sharpe Ratio: {summary.get("sharpe_ratio", 0)}')
        print(f'Sortino Ratio: {summary.get("sortino_ratio", 0)}')
        print(f'Value at Risk (95%): {summary.get("value_at_risk", 0)}%')
        print(f'Expected Shortfall (95%): {summary.get("expected_shortfall", 0)}%')
        print(f'Ulcer Index: {summary.get("ulcer_index", 0)}')

        print('=============================\n')

    # --- Private methods ---

    def _has_trades(self):
        """Check if there are any trades."""
        return bool(self.trades)

    def _has_winning_trades(self):
        """Check if there are any winning trades."""
        return bool(self.winning_trades)

    def _has_losing_trades(self):
        """Check if there are any losing trades."""
        return bool(self.losing_trades)

    def _initialize_calculations(self):
        """Perform initial calculations used by multiple metrics."""
        # Calculate win/loss trades
        self._calculate_win_loss_trades()

        # Calculate cumulative PnL for drawdown calculations
        self._calculate_cumulative_pnl()

        # Cache commonly used values
        self.total_return = sum(trade['return_percentage_of_margin'] for trade in self.trades)
        self.total_return_contract = sum(trade['return_percentage_of_contract'] for trade in self.trades)
        self.total_margin_used = sum(trade.get('margin_requirement', 0) for trade in self.trades)
        self.max_drawdown, self.maximum_drawdown_percentage = self._calculate_max_drawdown()

        # Cache commonly used lists
        self.returns = [trade['return_percentage_of_margin'] for trade in self.trades]
        self.durations = [trade.get('duration_hours', 0) for trade in self.trades]

    def _calculate_win_loss_trades(self):
        """Calculate winning and losing trades."""
        self.winning_trades = [t for t in self.trades if t['return_percentage_of_margin'] > 0]
        self.losing_trades = [t for t in self.trades if t['return_percentage_of_margin'] <= 0]
        self.win_count = len(self.winning_trades)
        self.loss_count = len(self.losing_trades)
        self.win_rate = (self.win_count / self.total_trades) * 100 if self.total_trades > 0 else 0

    def _calculate_cumulative_pnl(self):
        """Calculate cumulative PnL for drawdown calculations."""

        # Sort trades by entry_time or date if entry_time is not available
        def get_trade_time(trade):
            if 'entry_time' in trade:
                return trade['entry_time']
            elif 'date' in trade:
                return trade['date']
            return None  # Default case if neither field exists

        sorted_trades = sorted(self.trades, key=get_trade_time)
        net_pnls = [trade['net_pnl'] for trade in sorted_trades]
        return_pcts = [trade['return_percentage_of_margin'] for trade in sorted_trades]
        self.cumulative_pnl_dollars = np.cumsum(net_pnls).tolist()
        self.cumulative_pnl_pct = np.cumsum(return_pcts).tolist()

    def _calculate_max_drawdown(self):
        """Calculate the maximum drawdown given a list of trades."""
        if not self._has_trades():
            return 0, 0

        # Calculate max drawdown in dollars
        peak_dollars = 0
        max_drawdown = 0
        for val in self.cumulative_pnl_dollars:
            if val > peak_dollars:
                peak_dollars = val
            drawdown = peak_dollars - val
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate max drawdown in percentage
        peak_pct = 0
        maximum_drawdown_percentage = 0
        for val in self.cumulative_pnl_pct:
            if val > peak_pct:
                peak_pct = val
            drawdown_percentage = peak_pct - val
            if drawdown_percentage > maximum_drawdown_percentage:
                maximum_drawdown_percentage = drawdown_percentage

        return round(max_drawdown, 2), round(maximum_drawdown_percentage, 2)

    def _calculate_average_win_percentage_of_margin(self):
        """Calculate average win percentage of margin."""
        if not self._has_winning_trades():
            return 0

        return safe_average([trade['return_percentage_of_margin'] for trade in self.winning_trades])

    def _calculate_average_loss_percentage_of_margin(self):
        """Calculate average loss percentage of margin."""
        if not self._has_losing_trades():
            return 0

        return safe_average([trade['return_percentage_of_margin'] for trade in self.losing_trades])

    def _calculate_commission_percentage_of_margin(self):
        """Calculate commission as percentage of margin."""
        if not self._has_trades():
            return 0

        total_commission_paid = sum(trade.get('commission', 0) for trade in self.trades)
        total_margin_used = self.total_margin_used

        commission_percentage_of_margin = (
                                                  total_commission_paid / total_margin_used) * 100 if total_margin_used > 0 else 0
        return commission_percentage_of_margin

    def _calculate_profit_factor(self):
        """Calculate a profit factor using percentage returns: Total Win % / Total Loss %."""
        if not self._has_trades():
            return 0

        total_win_percentage = sum(trade['return_percentage_of_margin'] for trade in self.winning_trades)
        total_loss_percentage = sum(trade['return_percentage_of_margin'] for trade in self.losing_trades)

        if total_loss_percentage == 0:
            return float('inf')  # No losses

        return abs(safe_divide(total_win_percentage, total_loss_percentage))

    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate the Sharpe ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Returns."""
        if not self._has_trades() or len(self.returns) < MIN_RETURNS_FOR_SHARPE:
            return 0

        avg_return = safe_average(self.returns)

        # Calculate standard deviation
        std_dev = np.std(self.returns, ddof=0)

        if std_dev == 0:
            return 0  # Avoid division by zero

        return safe_divide(avg_return - risk_free_rate, std_dev)

    def _calculate_sortino_ratio(self, risk_free_rate=0.0):
        """Calculate Sortino ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Negative Returns."""
        if not self._has_trades():
            return 0

        avg_return = safe_average(self.returns)

        # Calculate downside deviation (returns below the risk-free rate)
        negative_returns = [r - risk_free_rate for r in self.returns if r < risk_free_rate]

        if not negative_returns:
            return float('inf')  # No negative returns

        downside_variance = safe_average([r ** 2 for r in negative_returns])
        downside_deviation = downside_variance ** 0.5

        if downside_deviation == 0:
            return 0  # Avoid division by zero

        return safe_divide(avg_return - risk_free_rate, downside_deviation)

    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio: Annualized Return / Maximum Drawdown."""
        if not self._has_trades():
            return 0

        if self.maximum_drawdown_percentage == 0:
            return float('inf')  # No drawdown

        return safe_divide(self.total_return, self.maximum_drawdown_percentage)

    def _calculate_value_at_risk(self, confidence=0.95):
        """ Returns the loss that won't be exceeded with the given confidence level. """
        if not self._has_trades() or len(self.returns) < MIN_RETURNS_FOR_VAR:
            return 0

        # Sort returns in ascending order (worst to best)
        sorted_returns = sorted(self.returns)

        # Find the index corresponding to the confidence level
        index = int((1 - confidence) * len(sorted_returns))

        # Return the absolute value of the loss at that index
        return abs(sorted_returns[max(0, index)])

    def _calculate_expected_shortfall(self, confidence=0.95):
        """ Returns the average loss in the worst (1-confidence)% of cases."""
        if not self._has_trades() or len(self.returns) < MIN_RETURNS_FOR_VAR:
            return 0

        # Sort returns in ascending order (worst to best)
        sorted_returns = sorted(self.returns)

        # Find the index corresponding to the confidence level
        index = int((1 - confidence) * len(sorted_returns))

        # Calculate the average of the worst returns
        worst_returns = sorted_returns[:max(1, index + 1)]
        return abs(safe_average(worst_returns))

    def _calculate_ulcer_index(self):
        """Calculate Ulcer Index, a measure of downside risk that considers both depth and duration of drawdowns."""
        if not self._has_trades():
            return 0

        # Calculate percentage drawdowns at each point
        drawdowns = []
        peak = 0

        for val in self.cumulative_pnl_pct:
            if val > peak:
                peak = val
                drawdowns.append(0)
            else:
                # Use the same drawdown calculation as in _calculate_max_drawdown
                drawdown_pct = peak - val
                drawdowns.append(drawdown_pct)

        # Calculate Ulcer Index
        return np.sqrt(np.mean(np.array(drawdowns) ** 2))

