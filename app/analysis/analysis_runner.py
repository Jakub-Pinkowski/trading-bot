from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.utils.analysis_utils.analisys_utils import match_trades
from app.analysis.performance_metrics.pnl import calculate_pnl


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    matched_trades = match_trades(trades_data)
    print(matched_trades)

    pnl = calculate_pnl(matched_trades)
    print(pnl)


    # pnl_alerts = calculate_alerts_pnl(alerts_data)
