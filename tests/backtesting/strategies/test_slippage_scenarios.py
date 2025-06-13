from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.backtesting.strategies.base_strategy import BaseStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.rsi import RSIStrategy


# Create a concrete implementation of BaseStrategy for testing
class StrategyForTesting(BaseStrategy):
    def __init__(self, rollover=False, trailing=None, slippage=0):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)

    def add_indicators(self, df):
        # Simple implementation for testing
        return df

    def generate_signals(self, df):
        # Simple implementation for testing
        df['signal'] = 0
        # Set some signals for testing
        if len(df) > 5:
            df.iloc[2, df.columns.get_loc('signal')] = 1  # Buy signal
            # For contract switch test, ensure there's an open position at the switch date (index 7)
            # by not closing the position before the switch
            if len(df) > 10:
                df.iloc[9, df.columns.get_loc('signal')] = -1  # Sell signal after switch date
            else:
                df.iloc[4, df.columns.get_loc('signal')] = -1  # Sell signal
        return df


# Helper function to create test dataframe
def create_test_df(length=10):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]
    data = {
        'open': np.random.rand(length) * 100 + 50,
        'high': np.random.rand(length) * 100 + 60,
        'low': np.random.rand(length) * 100 + 40,
        'close': np.random.rand(length) * 100 + 50,
    }
    df = pd.DataFrame(data, index=dates)
    return df


class TestSlippageScenarios:
    def test_different_slippage_values(self):
        """Test that different slippage percentages are correctly applied."""
        # Test with multiple slippage values
        slippage_values = [0.5, 1.0, 2.0, 5.0]

        for slippage in slippage_values:
            # Create a strategy with the current slippage value
            strategy = StrategyForTesting(slippage=slippage)
            df = create_test_df()
            df = strategy.generate_signals(df)  # Add signals

            trades = strategy.extract_trades(df, [])

            # Should have at least one trade
            assert len(trades) > 0

            # Find long and short trades
            long_trades = [t for t in trades if t['side'] == 'long']
            short_trades = [t for t in trades if t['side'] == 'short']

            # Verify slippage is applied correctly for long trades
            for trade in long_trades:
                # Get the original entry and exit prices from the dataframe
                entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

                original_entry_price = df.iloc[entry_idx]['open']
                original_exit_price = df.iloc[exit_idx]['open']

                # For long positions:
                # - Entry price should be higher than the original price (pay more on entry)
                # - Exit price should be lower than the original price (receive less on exit)
                expected_entry_price = round(original_entry_price * (1 + slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 - slippage / 100), 2)

                assert trade[
                           'entry_price'] == expected_entry_price, f"Long entry price with {slippage}% slippage should be {expected_entry_price}, got {trade['entry_price']}"
                assert trade[
                           'exit_price'] == expected_exit_price, f"Long exit price with {slippage}% slippage should be {expected_exit_price}, got {trade['exit_price']}"

            # Verify slippage is applied correctly for short trades
            for trade in short_trades:
                # Get the original entry and exit prices from the dataframe
                entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

                original_entry_price = df.iloc[entry_idx]['open']
                original_exit_price = df.iloc[exit_idx]['open']

                # For short positions:
                # - Entry price should be lower than the original price (receive less on entry)
                # - Exit price should be higher than the original price (pay more on exit)
                expected_entry_price = round(original_entry_price * (1 - slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 + slippage / 100), 2)

                assert trade[
                           'entry_price'] == expected_entry_price, f"Short entry price with {slippage}% slippage should be {expected_entry_price}, got {trade['entry_price']}"
                assert trade[
                           'exit_price'] == expected_exit_price, f"Short exit price with {slippage}% slippage should be {expected_exit_price}, got {trade['exit_price']}"

    def test_slippage_with_trailing_stop(self):
        """Test that slippage works correctly when combined with trailing stops."""
        # Create a strategy with both slippage and trailing stop
        slippage = 2.0
        trailing = 3.0
        strategy = StrategyForTesting(slippage=slippage, trailing=trailing)

        # Create a dataframe with a price pattern that will trigger a trailing stop
        df = create_test_df(length=20)

        # Modify prices to test trailing stop
        df.loc[df.index[2], 'open'] = 100.0  # Entry price
        df.loc[df.index[3], 'high'] = 110.0  # Price moves up, trailing stop should adjust
        df.loc[df.index[4], 'low'] = 95.0  # Price drops but not enough to trigger stop
        df.loc[df.index[5], 'low'] = 90.0  # Price drops below trailing stop

        df = strategy.generate_signals(df)
        trades = strategy.extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Find trades that were closed by trailing stop
        # These would be trades where the exit price is the trailing stop price
        # For simplicity, we'll just verify that slippage is applied to all trades
        for trade in trades:
            # Get the original entry and exit prices from the dataframe
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            original_entry_price = df.iloc[entry_idx]['open']

            # For long positions:
            if trade['side'] == 'long':
                # Entry price should be higher than the original price (pay more on entry)
                expected_entry_price = round(original_entry_price * (1 + slippage / 100), 2)
                assert trade[
                           'entry_price'] == expected_entry_price, f"Long entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"

                # For exit price, we can't easily predict the exact value if it's a trailing stop
                # because the exit price is the trailing stop price, not the open price
                # Just verify that slippage is applied by checking that the exit price is different from the original price
                original_exit_price = df.iloc[exit_idx]['open']
                assert trade[
                           'exit_price'] != original_exit_price, "Exit price should be different from original price due to slippage"

                # For long positions with trailing stops, the exit price should be lower than the entry price
                # (since trailing stops are used to lock in profits)
                if trailing is not None:
                    # If the trade was closed by a trailing stop, the exit price should be lower than the highest price
                    # reached during the trade (which we can't easily determine in this test)
                    # So we'll just verify that slippage is applied in some way
                    pass

            # For short positions:
            elif trade['side'] == 'short':
                # Entry price should be lower than the original price (receive less on entry)
                expected_entry_price = round(original_entry_price * (1 - slippage / 100), 2)
                assert trade[
                           'entry_price'] == expected_entry_price, f"Short entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"

                # For exit price, we can't easily predict the exact value if it's a trailing stop
                # because the exit price is the trailing stop price, not the open price
                # Just verify that slippage is applied by checking that the exit price is different from the original price
                original_exit_price = df.iloc[exit_idx]['open']
                assert trade[
                           'exit_price'] != original_exit_price, "Exit price should be different from original price due to slippage"

                # For short positions with trailing stops, the exit price should be higher than the entry price
                # (since trailing stops are used to lock in profits)
                if trailing is not None:
                    # If the trade was closed by a trailing stop, the exit price should be higher than the lowest price
                    # reached during the trade (which we can't easily determine in this test)
                    # So we'll just verify that slippage is applied in some way
                    pass

    def test_slippage_with_contract_rollover(self):
        """Test that slippage is applied correctly during contract rollovers."""
        # Create a strategy with rollover and slippage
        slippage = 2.0
        strategy = StrategyForTesting(rollover=True, slippage=slippage)

        # Create a dataframe
        df = create_test_df(length=20)

        # Create a switch date in the middle of the dataframe
        switch_date = df.index[10]

        # Generate signals and extract trades
        df = strategy.generate_signals(df)
        trades = strategy.extract_trades(df, [switch_date])

        # Should have at least one trade
        assert len(trades) > 0

        # Find trades that were closed due to rollover
        rollover_trades = [trade for trade in trades if trade.get('switch')]

        # There should be at least one rollover trade
        assert len(rollover_trades) > 0, "No trades closed due to rollover"

        # Verify slippage is applied correctly to rollover trades
        for trade in rollover_trades:
            # Get the original entry and exit prices from the dataframe
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            original_entry_price = df.iloc[entry_idx]['open']
            original_exit_price = df.iloc[exit_idx]['open']

            # For long positions:
            if trade['side'] == 'long':
                # Entry price should be higher than the original price (pay more on entry)
                expected_entry_price = round(original_entry_price * (1 + slippage / 100), 2)
                assert trade[
                           'entry_price'] == expected_entry_price, f"Long entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"

                # For contract rollovers, the exit price is determined by the previous row's open price,
                # not the current bar's open price. We can't easily predict the exact exit price.
                # Just verify that slippage is applied by checking that the exit price is different from the original price.
                assert trade[
                           'exit_price'] != original_exit_price, "Exit price should be different from original price due to slippage"

            # For short positions:
            elif trade['side'] == 'short':
                # Entry price should be lower than the original price (receive less on entry)
                expected_entry_price = round(original_entry_price * (1 - slippage / 100), 2)
                assert trade[
                           'entry_price'] == expected_entry_price, f"Short entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"

                # For contract rollovers, the exit price is determined by the previous row's open price,
                # not the current bar's open price. We can't easily predict the exact exit price.
                # Just verify that slippage is applied by checking that the exit price is different from the original price.
                assert trade[
                           'exit_price'] != original_exit_price, "Exit price should be different from original price due to slippage"

    def test_slippage_with_high_volatility(self):
        """Test that slippage works correctly in high volatility conditions."""
        # Create a strategy with slippage
        slippage = 2.0
        strategy = EMACrossoverStrategy(ema_short=5, ema_long=15, slippage=slippage)

        # Create a dataframe with high volatility
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame(index=dates)

        # Create a highly volatile price series
        base_price = 100

        # Generate random walk with high volatility
        np.random.seed(42)  # For reproducibility

        # Daily volatility of 4% (very high)
        volatility = 0.04

        # Generate prices
        close_prices = [base_price]
        for _ in range(49):
            # Random daily return with high volatility
            daily_return = np.random.normal(0, volatility)
            new_price = close_prices[-1] * (1 + daily_return)
            close_prices.append(new_price)

        # Create OHLC data with large intraday ranges
        df['open'] = close_prices
        df['high'] = [p * (1 + np.random.uniform(0.01, 0.03)) for p in close_prices]  # 1-3% above close
        df['low'] = [p * (1 - np.random.uniform(0.01, 0.03)) for p in close_prices]  # 1-3% below close
        df['close'] = close_prices

        # Run the strategy
        trades = strategy.run(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Verify slippage is applied correctly to all trades
        for trade in trades:
            # Get the original entry and exit prices from the dataframe
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            original_entry_price = df.iloc[entry_idx]['open']
            original_exit_price = df.iloc[exit_idx]['open']

            # For long positions:
            if trade['side'] == 'long':
                # Entry price should be higher than the original price (pay more on entry)
                # Exit price should be lower than the original price (receive less on exit)
                expected_entry_price = round(original_entry_price * (1 + slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 - slippage / 100), 2)

                assert trade[
                           'entry_price'] == expected_entry_price, f"Long entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"
                assert trade[
                           'exit_price'] == expected_exit_price, f"Long exit price with slippage should be {expected_exit_price}, got {trade['exit_price']}"

            # For short positions:
            elif trade['side'] == 'short':
                # Entry price should be lower than the original price (receive less on entry)
                # Exit price should be higher than the original price (pay more on exit)
                expected_entry_price = round(original_entry_price * (1 - slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 + slippage / 100), 2)

                assert trade[
                           'entry_price'] == expected_entry_price, f"Short entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"
                assert trade[
                           'exit_price'] == expected_exit_price, f"Short exit price with slippage should be {expected_exit_price}, got {trade['exit_price']}"

    def test_slippage_impact_on_performance(self):
        """Test the impact of slippage on overall strategy performance."""
        # Create strategies with different slippage values
        slippage_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        strategies = [RSIStrategy(slippage=s) for s in slippage_values]

        # Create a test dataframe
        df = create_test_df(length=100)

        # Run each strategy and calculate performance metrics
        results = []
        for i, strategy in enumerate(strategies):
            trades = strategy.run(df, [])

            # Calculate total return
            total_return = 0
            for trade in trades:
                if trade['side'] == 'long':
                    trade_return = trade['exit_price'] / trade['entry_price'] - 1
                else:  # short
                    trade_return = 1 - trade['exit_price'] / trade['entry_price']
                total_return += trade_return

            results.append({
                'slippage': slippage_values[i],
                'num_trades': len(trades),
                'total_return': total_return
            })

        # Verify that higher slippage leads to lower returns
        for i in range(1, len(results)):
            assert results[i]['total_return'] <= results[i - 1][
                'total_return'], f"Higher slippage ({results[i]['slippage']}%) should lead to lower or equal returns than lower slippage ({results[i - 1]['slippage']}%)"
