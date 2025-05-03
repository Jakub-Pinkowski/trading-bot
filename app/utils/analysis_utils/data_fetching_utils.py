import json
import os
from datetime import datetime

from zoneinfo import ZoneInfo


def save_trades_data(trades_json, trades_dir, timezone="Europe/Berlin"):
    trades_by_day = {}

    # Organize trades by extracting the date from each trade's timestamp
    for trade in trades_json:
        trade_datetime = datetime.strptime(trade["trade_time"], "%Y%m%d-%H:%M:%S").astimezone(ZoneInfo(timezone))
        trade_date = trade_datetime.strftime("%Y-%m-%d")
        trades_by_day.setdefault(trade_date, []).append(trade)

    # Save unique trades only
    for date, daily_trades in trades_by_day.items():
        daily_file_path = os.path.join(trades_dir, f'trades_{date}.json')

        # Use execution_id as unique identifier
        unique_trades = {}

        # Load existing trades if a file exists to avoid duplicates
        if os.path.exists(daily_file_path):
            with open(daily_file_path, 'r') as file:
                existing_data = json.load(file)
                if isinstance(existing_data, list):
                    for trade in existing_data:
                        unique_trades[trade["execution_id"]] = trade

        # Add current trades (automatically overriding duplicates)
        for trade in daily_trades:
            unique_trades[trade["execution_id"]] = trade

        # Save back to file
        with open(daily_file_path, 'w') as file:
            json.dump(list(unique_trades.values()), file, indent=4)
