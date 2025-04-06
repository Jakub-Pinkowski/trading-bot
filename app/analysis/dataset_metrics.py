import numpy as np


def calculate_dataset_metrics(trades_df):
    metrics = {}

    # Win Rate (%)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    metrics['win_rate'] = (len(wins) / len(trades_df) * 100) if len(trades_df) > 0 else 0

    # Average Win & Average Loss
    metrics['average_win'] = wins['pnl'].mean() if not wins.empty else 0
    metrics['average_loss'] = losses['pnl'].mean() if not losses.empty else 0

    # Profit Factor (Total Profit / Total Loss)
    total_profit = wins['pnl'].sum() if not wins.empty else 0
    total_loss = abs(losses['pnl'].sum()) if not losses.empty else 0
    metrics['profit_factor'] = (total_profit / total_loss) if total_loss > 0 else np.nan

    # Sharpe Ratio (Mean PnL / Std Dev of PnL)
    pnl_mean = trades_df['pnl'].mean()
    pnl_std = trades_df['pnl'].std()
    metrics['sharpe_ratio'] = (pnl_mean / pnl_std) if pnl_std > 0 else np.nan

    # Sortino Ratio (Mean PnL / Std Dev of Negative PnL)
    downside_std = losses['pnl'].std() if not losses.empty else 0
    metrics['sortino_ratio'] = (pnl_mean / downside_std) if downside_std > 0 else np.nan

    # Cumulative PnL
    metrics['cumulative_pnl'] = trades_df['pnl'].sum()

    # Maximum Drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    metrics['max_drawdown'] = drawdown.max() if not drawdown.empty else 0

    # Total Commission
    metrics['total_commission'] = trades_df['total_commission'].sum()

    # Convert all values to native Python types and round to 2-4 decimal places
    metrics = {key: round(float(value), 4) if isinstance(value, (np.floating, float))
    else round(int(value), 4) if isinstance(value, (np.integer, int))
    else value for key, value in metrics.items()}

    return metrics
