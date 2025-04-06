from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.analysis.performance_metrics.pnl import calculate_pnl


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    pnl = calculate_pnl(trades_data)
    print(pnl)

    # pnl_alerts = calculate_alerts_pnl(alerts_data)
