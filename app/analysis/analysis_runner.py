from app.analysis.data_fetching import get_alerts_data, get_tw_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.utils.analysis_utils.analysis_utils import match_trades
from app.utils.file_utils import save_to_csv
from config import ALERTS_PER_TRADE_METRICS_FILE_PATH, ALERTS_DATASET_METRICS_FILE_PATH, TRADES_PER_TRADE_METRICS_FILE_PATH, \
    TRADES_DATASET_METRICS_FILE_PATH


def run_analysis():
    alerts_data = get_alerts_data()
    tw_alerts = get_tw_alerts_data()
    trades_data = get_trades_data()

    alerts_matched = match_trades(tw_alerts, is_alerts=True)
    print(alerts_matched)
    trades_matches = match_trades(trades_data)
    print(trades_matches)

    # Per trades metrics
    alerts_with_per_trade_metrics = add_per_trade_metrics(alerts_matched)
    trades_with_per_trade_metrics = add_per_trade_metrics(trades_matches)

    # Dataset metrics
    alerts_dataset_test = calculate_dataset_metrics(alerts_with_per_trade_metrics)
    trades_dataset_metrics = calculate_dataset_metrics(trades_with_per_trade_metrics)

    # TODO: Add automatic Google Sheets integration
    # TODO: Don't save the duplicated data
    # Save data to CSV
    save_to_csv(alerts_dataset_test, ALERTS_PER_TRADE_METRICS_FILE_PATH)
    save_to_csv(alerts_dataset_test, ALERTS_DATASET_METRICS_FILE_PATH, dictionary_columns=["Metric", "Value"])
    save_to_csv(trades_with_per_trade_metrics, TRADES_PER_TRADE_METRICS_FILE_PATH)
    save_to_csv(trades_dataset_metrics, TRADES_DATASET_METRICS_FILE_PATH, dictionary_columns=["Metric", "Value"])
