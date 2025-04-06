from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.analysis.performance_metrics.pnl import match_trades


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    matched_trades = match_trades(trades_data)
    print(matched_trades)
    # pnl = calculate_pnl(trades_data)


    # pnl_alerts = calculate_alerts_pnl(alerts_data)
