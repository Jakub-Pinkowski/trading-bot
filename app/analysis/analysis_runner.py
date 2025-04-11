from app.analysis.data_fetching import get_alerts_data, get_tw_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.utils.analysis_utils.analysis_utils import match_trades
from app.utils.file_utils import save_to_csv
from app.analysis.data_cleaning import clean_alerts_data, clean_trades_data
from config import TW_ALERTS_PER_TRADE_METRICS_FILE_PATH, TW_ALERTS_DATASET_METRICS_FILE_PATH, TRADES_PER_TRADE_METRICS_FILE_PATH, \
    TRADES_DATASET_METRICS_FILE_PATH


def run_analysis():
    # Fetch data
    # NOTE: We ignore alerts for now
    alerts_data = get_alerts_data()
    tw_alerts_data = get_tw_alerts_data()
    trades_data = get_trades_data()

    # TODO: Implement this part
    # Clean data
    alerts = clean_alerts_data(alerts_data)
    tw_alerts = clean_alerts_data(tw_alerts_data)
    trades  = clean_trades_data(trades_data)

    # Create trade pairs
    tw_alerts_matched = match_trades(tw_alerts, is_alerts=True)
    trades_matches = match_trades(trades)


    # Per trades metrics
    tw_alerts_with_per_trade_metrics = add_per_trade_metrics(tw_alerts_matched)
    trades_with_per_trade_metrics = add_per_trade_metrics(trades_matches)

    # Dataset metrics
    tw_alerts_dataset_metrics = calculate_dataset_metrics(tw_alerts_with_per_trade_metrics)
    trades_dataset_metrics = calculate_dataset_metrics(trades_with_per_trade_metrics)

    # TODO: Add automatic Google Sheets integration
    # TODO: Don't save the duplicated data
    # Save data to CSV
    save_to_csv(tw_alerts_with_per_trade_metrics, TW_ALERTS_PER_TRADE_METRICS_FILE_PATH)
    save_to_csv(tw_alerts_dataset_metrics, TW_ALERTS_DATASET_METRICS_FILE_PATH, dictionary_columns=["Metric", "Value"])
    save_to_csv(trades_with_per_trade_metrics, TRADES_PER_TRADE_METRICS_FILE_PATH)
    save_to_csv(trades_dataset_metrics, TRADES_DATASET_METRICS_FILE_PATH, dictionary_columns=["Metric", "Value"])
