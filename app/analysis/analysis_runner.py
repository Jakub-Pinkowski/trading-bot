from app.analysis.data_fetching import get_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.utils.analysis_utils.analysis_utils import match_trades
from app.utils.file_utils import save_to_csv
from config import PER_TRADE_METRICS_FILE_PATH, DATASET_METRICS_FILE_PATH


def run_analysis():
    alerts_data = get_alerts_data()
    trades_data = get_trades_data()

    matched_trades = match_trades(trades_data)

    trades_with_per_trade_metrics = add_per_trade_metrics(matched_trades)

    dataset_metrics = calculate_dataset_metrics(trades_with_per_trade_metrics)

    # TODO: Add automatic Google Sheets integration
    # Save data to CSV
    save_to_csv(trades_with_per_trade_metrics, PER_TRADE_METRICS_FILE_PATH)  # Save trades data
    save_to_csv(dataset_metrics, DATASET_METRICS_FILE_PATH, dictionary_columns=["Metric", "Value"])  # Save metrics
