from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.analysis.performance_metrics.metrics import add_per_trade_metrics
from app.utils.analysis_utils.analisys_utils import match_trades


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    matched_trades = match_trades(trades_data)
    print(matched_trades)

    trades_with_per_trade_metrics = add_per_trade_metrics(matched_trades)
    print(trades_with_per_trade_metrics)

    # pnl_alerts = calculate_alerts_pnl(alerts_data)
