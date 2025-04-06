from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.utils.analysis_utils.analysis_utils import match_trades


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    matched_trades = match_trades(trades_data)
    print(matched_trades)

    trades_with_per_trade_metrics = add_per_trade_metrics(matched_trades)
    print(trades_with_per_trade_metrics)

    dataset_metrics = calculate_dataset_metrics(trades_with_per_trade_metrics)
    print(dataset_metrics)
