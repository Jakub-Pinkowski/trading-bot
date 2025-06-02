from app.analysis.data_cleaning import clean_alerts_data, clean_trades_data
from app.analysis.data_fetching import get_alerts_data, get_tw_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.analysis.trades_matching import match_trades
from app.utils.analysis_utils.analysis_utils import is_nonempty
from app.utils.file_utils import save_to_csv
from app.utils.logger import get_logger
from config import (TW_ALERTS_PER_TRADE_METRICS_FILE_PATH,
                    TW_ALERTS_DATASET_METRICS_FILE_PATH,
                    TRADES_PER_TRADE_METRICS_FILE_PATH,
                    TRADES_DATASET_METRICS_FILE_PATH)

logger = get_logger()


def run_analysis():
    print('Running analysis...')
    # Fetch raw data
    alerts_data = get_alerts_data()
    tw_alerts_data = get_tw_alerts_data()
    trades_data = get_trades_data()

    # Process TW Alerts
    if is_nonempty(tw_alerts_data):
        tw_alerts = clean_alerts_data(tw_alerts_data, tw_alerts=True)
        if is_nonempty(tw_alerts):
            tw_alerts_matched = match_trades(tw_alerts, is_alerts=True)
            if is_nonempty(tw_alerts_matched):
                tw_alerts_with_per_trade_metrics = add_per_trade_metrics(tw_alerts_matched)
                if is_nonempty(tw_alerts_with_per_trade_metrics):
                    tw_alerts_dataset_metrics = calculate_dataset_metrics(tw_alerts_with_per_trade_metrics)
                    save_to_csv(tw_alerts_with_per_trade_metrics, TW_ALERTS_PER_TRADE_METRICS_FILE_PATH)
                    save_to_csv(
                        tw_alerts_dataset_metrics,
                        TW_ALERTS_DATASET_METRICS_FILE_PATH,
                        dictionary_columns=['Metric', 'Value']
                    )
                    logger.info('TW Alerts processed successfully.')

    # Process Trades
    if is_nonempty(trades_data):
        trades = clean_trades_data(trades_data)
        if is_nonempty(trades):
            trades_matches = match_trades(trades)
            if is_nonempty(trades_matches):
                trades_with_per_trade_metrics = add_per_trade_metrics(trades_matches)
                if is_nonempty(trades_with_per_trade_metrics):
                    trades_dataset_metrics = calculate_dataset_metrics(trades_with_per_trade_metrics)
                    save_to_csv(trades_with_per_trade_metrics, TRADES_PER_TRADE_METRICS_FILE_PATH)
                    save_to_csv(
                        trades_dataset_metrics,
                        TRADES_DATASET_METRICS_FILE_PATH,
                        dictionary_columns=['Metric', 'Value']
                    )
                    logger.info('Trades processed successfully.')
