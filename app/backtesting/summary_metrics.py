from app.utils.logger import get_logger

logger = get_logger('backtesting/summary_metrics')


def _safe_average(values, count=None):
    """Safely calculate average of values."""
    if not values:
        return 0
    count = count or len(values)
    return sum(values) / count if count > 0 else 0


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

    def _has_trades(self):
        """Check if there are any trades."""
        return bool(self.trades) and len(self.trades) > 0

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

    def _calculate_win_loss_trades(self):
        """Calculate winning and losing trades."""
        self.winning_trades = [t for t in self.trades if t['return_percentage_of_margin'] > 0]
        self.losing_trades = [t for t in self.trades if t['return_percentage_of_margin'] <= 0]
        self.win_count = len(self.winning_trades)
        self.loss_count = len(self.losing_trades)
        self.win_rate = (self.win_count / self.total_trades) * 100 if self.total_trades > 0 else 0

    def _calculate_cumulative_pnl(self):
        """Calculate cumulative PnL for drawdown calculations."""
        self.cumulative_pnl_dollars = []
        self.cumulative_pnl_pct = []
        cum_sum_dollars = 0
        cum_sum_pct = 0

        for trade in self.trades:
            cum_sum_dollars += trade['net_pnl']
            cum_sum_pct += trade['return_percentage_of_margin']
            self.cumulative_pnl_dollars.append(cum_sum_dollars)
            self.cumulative_pnl_pct.append(cum_sum_pct)

    def calculate_max_drawdown(self):
        """Calculate the maximum drawdown given a list of trades."""
        if not self._has_trades():
            return 0, 0

        # Calculate max drawdown in dollars
        peak_dollars = self.cumulative_pnl_dollars[0]
        max_drawdown = 0
        for val in self.cumulative_pnl_dollars:
            if val > peak_dollars:
                peak_dollars = val
            drawdown = peak_dollars - val
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate max drawdown in percentage
        peak_pct = self.cumulative_pnl_pct[0]
        maximum_drawdown_percentage = 0
        for val in self.cumulative_pnl_pct:
            if val > peak_pct:
                peak_pct = val
            drawdown_percentage = peak_pct - val
            if drawdown_percentage > maximum_drawdown_percentage:
                maximum_drawdown_percentage = drawdown_percentage

        return round(max_drawdown, 2), round(maximum_drawdown_percentage, 2)

    def calculate_max_consecutive(self, win=True):
        """Calculate maximum consecutive wins or losses."""
        if not self._has_trades():
            return 0

        # Sort trades by date if available
        if 'date' in self.trades[0]:
            sorted_trades = sorted(self.trades, key=lambda x: x['date'])
        else:
            sorted_trades = self.trades

        # Track consecutive wins/losses
        current_streak = 0
        max_streak = 0

        for trade in sorted_trades:
            is_win = trade['return_percentage_of_margin'] > 0

            if (win and is_win) or (not win and not is_win):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Returns."""
        if not self._has_trades() or len(self.trades) < 2:  # Need at least 2 trades for standard deviation
            return 0

        returns = [trade['return_percentage_of_margin'] for trade in self.trades]
        avg_return = _safe_average(returns)

        # Calculate standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0  # Avoid division by zero

        sharpe_ratio = (avg_return - risk_free_rate) / std_dev
        return sharpe_ratio

    def calculate_sortino_ratio(self, risk_free_rate=0.0):
        """Calculate Sortino ratio: (Average Return - Risk-Free Rate) / Standard Deviation of Negative Returns."""
        if not self._has_trades():
            return 0

        returns = [trade['return_percentage_of_margin'] for trade in self.trades]
        avg_return = _safe_average(returns)

        # Calculate downside deviation (returns below the risk-free rate)
        negative_returns = [r - risk_free_rate for r in returns if r < risk_free_rate]

        if not negative_returns:
            return float('inf')  # No negative returns

        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = downside_variance ** 0.5

        if downside_deviation == 0:
            return 0  # Avoid division by zero

        sortino_ratio = (avg_return - risk_free_rate) / downside_deviation
        return sortino_ratio

    def calculate_calmar_ratio(self):
        """Calculate Calmar ratio: Annualized Return / Maximum Drawdown."""
        if not self._has_trades():
            return 0

        # Calculate total return (assuming percentage returns)
        total_return = sum(trade['return_percentage_of_margin'] for trade in self.trades)

        # Get maximum drawdown percentage
        _, max_drawdown_pct = self.calculate_max_drawdown()

        if max_drawdown_pct == 0:
            return float('inf')  # No drawdown

        calmar_ratio = total_return / max_drawdown_pct
        return calmar_ratio

    def calculate_profit_factor(self):
        """Calculate profit factor: Total Net Profit / Total Net Loss."""
        if not self._has_trades():
            return 0

        total_net_profit = sum(trade['net_pnl'] for trade in self.winning_trades)
        total_net_loss = sum(trade['net_pnl'] for trade in self.losing_trades)

        if total_net_loss == 0:
            return float('inf')  # No losses

        profit_factor = abs(total_net_profit / total_net_loss)
        return profit_factor

    def calculate_return_to_drawdown_ratio(self):
        """Calculate return-to-drawdown ratio: Total Return Percentage / Maximum Drawdown Percentage."""
        if not self._has_trades():
            return 0

        total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in self.trades)

        # Get maximum drawdown percentage
        _, maximum_drawdown_percentage = self.calculate_max_drawdown()

        if maximum_drawdown_percentage == 0:
            return float('inf')  # No drawdown

        return_to_drawdown_ratio = total_return_percentage_of_margin / maximum_drawdown_percentage
        return return_to_drawdown_ratio

    def calculate_win_rate(self):
        """Calculate win rate: (Number of Winning Trades / Total Trades) * 100."""
        # This is already calculated in _initialize_calculations, return the values
        return self.win_rate, self.win_count, self.loss_count, self.winning_trades, self.losing_trades

    def calculate_average_trade_duration(self):
        """Calculate average trade duration in hours."""
        if not self._has_trades():
            return 0

        return _safe_average([trade['duration_hours'] for trade in self.trades], self.total_trades)

    def calculate_total_margin_used(self):
        """Calculate the total margin used across all trades."""
        if not self._has_trades():
            return 0

        total_margin_used = sum(trade.get('margin_requirement', 0) for trade in self.trades)
        return total_margin_used

    def calculate_total_return_percentage_of_margin(self):
        """Calculate total return percentage of margin."""
        if not self._has_trades():
            return 0

        total_return_percentage_of_margin = sum(trade['return_percentage_of_margin'] for trade in self.trades)
        return total_return_percentage_of_margin

    def calculate_average_trade_return_percentage_of_margin(self):
        """Calculate average trade return percentage of margin."""
        if not self._has_trades():
            return 0

        total_return_percentage_of_margin = self.calculate_total_return_percentage_of_margin()
        return _safe_average([total_return_percentage_of_margin], self.total_trades)

    def calculate_average_win_percentage_of_margin(self):
        """Calculate average win percentage of margin."""
        if not self._has_winning_trades():
            return 0

        return _safe_average([trade['return_percentage_of_margin'] for trade in self.winning_trades])

    def calculate_average_loss_percentage_of_margin(self):
        """Calculate average loss percentage of margin."""
        if not self._has_losing_trades():
            return 0

        return _safe_average([trade['return_percentage_of_margin'] for trade in self.losing_trades])

    def calculate_commission_percentage_of_margin(self):
        """Calculate commission as percentage of margin."""
        if not self._has_trades():
            return 0

        total_commission_paid = sum(trade.get('commission', 0) for trade in self.trades)
        total_margin_used = self.calculate_total_margin_used()

        commission_percentage_of_margin = (
                                                  total_commission_paid / total_margin_used) * 100 if total_margin_used > 0 else 0
        return commission_percentage_of_margin

    def calculate_all_metrics(self):
        """Calculate all summary metrics and return as a dictionary."""
        if not self._has_trades():
            logger.error('No trades provided to calculate_all_metrics')
            return {}

        # ===== BASIC TRADE STATISTICS =====
        win_rate, win_count, loss_count, _, _ = self.calculate_win_rate()
        avg_duration_hours = self.calculate_average_trade_duration()

        # ===== NORMALIZED METRICS (PERCENTAGES) =====
        total_return_percentage_of_margin = self.calculate_total_return_percentage_of_margin()
        average_trade_return_percentage_of_margin = self.calculate_average_trade_return_percentage_of_margin()
        average_win_percentage_of_margin = self.calculate_average_win_percentage_of_margin()
        average_loss_percentage_of_margin = self.calculate_average_loss_percentage_of_margin()

        # ===== COMMISSION METRICS =====
        commission_percentage_of_margin = self.calculate_commission_percentage_of_margin()

        # ===== RISK METRICS =====
        profit_factor = self.calculate_profit_factor()
        max_drawdown, maximum_drawdown_percentage = self.calculate_max_drawdown()
        return_to_drawdown_ratio = self.calculate_return_to_drawdown_ratio()
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
        calmar_ratio = self.calculate_calmar_ratio()

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

            # Risk metrics
            'profit_factor': round(profit_factor, 2),
            'maximum_drawdown_percentage': maximum_drawdown_percentage,
            'return_to_drawdown_ratio': round(return_to_drawdown_ratio, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
        }

    def print_metrics(self):
        """Calculate and print all metrics for the trades."""
        metrics = self.calculate_all_metrics()
        if metrics:
            self.print_summary_metrics(metrics)
        else:
            logger.error('No metrics to print')
            print('No metrics available to print')

    @staticmethod
    def print_summary_metrics(summary):
        """Print summary metrics in a formatted way."""
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        average_trade_return_percentage = summary['average_trade_return_percentage_of_margin']
        if average_trade_return_percentage > 0:
            color = GREEN
        elif average_trade_return_percentage < 0:
            color = RED
        else:
            color = RESET  # default terminal color

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
            print(f'Total Return Percentage of Margin: {color}{summary["total_return_percentage_of_margin"]}%{RESET}')
        if average_trade_return_percentage == 0:
            print(f'Average Trade Return Percentage of Margin: 0.0%')
        else:
            print(f'Average Trade Return Percentage of Margin: {color}{average_trade_return_percentage}%{RESET}')
        print(f'Average Win Percentage of Margin: {GREEN}{summary["average_win_percentage_of_margin"]}%{RESET}')
        print(f'Average Loss Percentage of Margin: {RED}{summary["average_loss_percentage_of_margin"]}%{RESET}')
        print(f'Commission Percentage of Margin: {summary["commission_percentage_of_margin"]}%')

        # ===== RISK METRICS =====
        print('\n--- RISK METRICS ---')
        print(f'Profit Factor: {summary["profit_factor"]}')
        print(f'Maximum Drawdown Percentage: {summary.get("maximum_drawdown_percentage", 0)}%')
        print(f'Return to Drawdown Ratio: {summary.get("return_to_drawdown_ratio", 0)}')
        print(f'Sharpe Ratio: {summary.get("sharpe_ratio", 0)}')
        print(f'Sortino Ratio: {summary.get("sortino_ratio", 0)}')
        print(f'Calmar Ratio: {summary.get("calmar_ratio", 0)}')

        print('=============================\n')
