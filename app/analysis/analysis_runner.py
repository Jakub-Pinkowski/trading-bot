import os

import pandas as pd

from app.analysis.data_cleaning import clean_ibkr_alerts_data, clean_tv_alerts_data, clean_trades_data
from app.analysis.data_fetching import get_ibkr_alerts_data, get_tv_alerts_data, get_trades_data
from app.analysis.dataset_metrics import calculate_dataset_metrics
from app.analysis.per_trade_metrics import add_per_trade_metrics
from app.analysis.trades_matching import match_trades
from app.utils.analysis_utils.analysis_utils import is_nonempty
from app.utils.logger import get_logger
from config import DATA_DIR

logger = get_logger('analysis/runner')

# ==================== Module Paths ====================

ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_IBKR_ALERTS_DIR = ANALYSIS_DIR / "ibkr_alerts"
ANALYSIS_TRADES_DIR = ANALYSIS_DIR / "trades"
ANALYSIS_TV_ALERTS_DIR = ANALYSIS_DIR / "tv_alerts"

IBKR_ALERTS_PER_TRADE_METRICS_FILE_PATH = ANALYSIS_IBKR_ALERTS_DIR / "ibkr_alerts_per_trade_metrics.csv"
IBKR_ALERTS_DATASET_METRICS_FILE_PATH = ANALYSIS_IBKR_ALERTS_DIR / "ibkr_alerts_dataset_metrics.csv"
TRADES_PER_TRADE_METRICS_FILE_PATH = ANALYSIS_TRADES_DIR / "trades_per_trade_metrics.csv"
TRADES_DATASET_METRICS_FILE_PATH = ANALYSIS_TRADES_DIR / "trades_dataset_metrics.csv"
TV_ALERTS_PER_TRADE_METRICS_FILE_PATH = ANALYSIS_TV_ALERTS_DIR / "tv_alerts_per_trade_metrics.csv"
TV_ALERTS_DATASET_METRICS_FILE_PATH = ANALYSIS_TV_ALERTS_DIR / "tv_alerts_dataset_metrics.csv"


def save_to_csv(data, file_path, dictionary_columns=None):
    # Load existing data if a file exists
    if os.path.exists(file_path):
        try:
            existing = pd.read_csv(file_path)
        except Exception as err:
            logger.error(f'Could not read existing CSV for deduplication: {err}')
            existing = None
    else:
        existing = None

    if isinstance(data, pd.DataFrame):
        new_data = data.copy()
    elif isinstance(data, dict):
        columns = dictionary_columns if dictionary_columns else ['Key', 'Value']
        new_data = pd.DataFrame(list(data.items()), columns=columns)
    else:
        raise ValueError('Data must be either a Pandas DataFrame or a dictionary.')

    # Concatenate and deduplicate if a file exists; else just save data
    if existing is not None:
        concat = pd.concat([existing, new_data], ignore_index=True)
        deduped = concat.drop_duplicates()
    else:
        deduped = new_data

    # Save (overwrite) deduped data
    deduped.to_csv(file_path, index=False)


def run_analysis():
    print('Running analysis...')
    # Fetch raw data
    ibkr_alerts_data = get_ibkr_alerts_data()
    tv_alerts_data = get_tv_alerts_data()
    trades_data = get_trades_data()

    # Process IBKR Alerts
    if is_nonempty(ibkr_alerts_data):
        ibkr_alerts = clean_ibkr_alerts_data(ibkr_alerts_data)
        if is_nonempty(ibkr_alerts):
            ibkr_alerts_matched = match_trades(ibkr_alerts, is_ibkr_alerts=True)
            if is_nonempty(ibkr_alerts_matched):
                ibkr_alerts_with_per_trade_metrics = add_per_trade_metrics(ibkr_alerts_matched)
                if is_nonempty(ibkr_alerts_with_per_trade_metrics):
                    ibkr_alerts_dataset_metrics = calculate_dataset_metrics(ibkr_alerts_with_per_trade_metrics)
                    save_to_csv(ibkr_alerts_with_per_trade_metrics, IBKR_ALERTS_PER_TRADE_METRICS_FILE_PATH)
                    save_to_csv(
                        ibkr_alerts_dataset_metrics,
                        IBKR_ALERTS_DATASET_METRICS_FILE_PATH,
                        dictionary_columns=['Metric', 'Value']
                    )
                    logger.info('IBKR Alerts processed successfully.')

    # Process TradingView Alerts
    if is_nonempty(tv_alerts_data):
        tv_alerts = clean_tv_alerts_data(tv_alerts_data)
        if is_nonempty(tv_alerts):
            tv_alerts_matched = match_trades(tv_alerts, is_tv_alerts=True)
            if is_nonempty(tv_alerts_matched):
                tv_alerts_with_per_trade_metrics = add_per_trade_metrics(tv_alerts_matched)
                if is_nonempty(tv_alerts_with_per_trade_metrics):
                    tv_alerts_dataset_metrics = calculate_dataset_metrics(tv_alerts_with_per_trade_metrics)
                    save_to_csv(tv_alerts_with_per_trade_metrics, TV_ALERTS_PER_TRADE_METRICS_FILE_PATH)
                    save_to_csv(
                        tv_alerts_dataset_metrics,
                        TV_ALERTS_DATASET_METRICS_FILE_PATH,
                        dictionary_columns=['Metric', 'Value']
                    )
                    logger.info('TV Alerts processed successfully.')

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
