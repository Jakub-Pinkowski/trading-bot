# Backtesting Examples - Real Trade Entry/Exit Scenarios

**Last Updated:** January 20, 2026  
**Symbol Used:** ZS (Soybean Futures - CBOT:ZS1!)  
**Contract Size:** 5,000 bushels  
**Price Format:** Cents per bushel (e.g., 1053.75 = $10.5375/bushel)

---

## Table of Contents

1. [Signal Execution Timing](#signal-execution-timing)
2. [Example 1: RSI Strategy (15-minute bars)](#example-1-rsi-strategy-15-minute-bars)
3. [Example 2: EMA Crossover Strategy (5-minute bars)](#example-2-ema-crossover-strategy-5-minute-bars)
4. [Example 3: MACD Strategy (2-hour bars)](#example-3-macd-strategy-2-hour-bars)
5. [Example 4: Bollinger Bands Strategy (Daily bars)](#example-4-bollinger-bands-strategy-daily-bars)
6. [Position Exit Examples](#position-exit-examples)
7. [Slippage Impact](#slippage-impact)
8. [Contract Rollover Example](#contract-rollover-example)

---

## Signal Execution Timing

### Core Principle: 1-Bar Delay

All strategies follow the same execution pattern:

```
Bar N (Signal Bar):
├─ Indicators calculated from close price
├─ Signal generated based on indicator values
└─ Signal QUEUED for next bar

Bar N+1 (Execution Bar):
├─ Queued signal executed at OPEN
├─ Slippage applied to entry/exit price
└─ Position opened/closed
```

### Why This Approach?

1. **Realistic**: You can only see the signal AFTER the bar closes
2. **Industry Standard**: Used by TradingView, Backtrader, QuantConnect
3. **Conservative**: Slippage accounts for execution delay and market movement
4. **No Look-Ahead Bias**: Proper time sequencing (signal → execution)

---

## Example 1: RSI Strategy (15-minute bars)

### Strategy Logic

- **Buy Signal**: RSI crosses below 30 (oversold)
- **Sell Signal**: RSI crosses above 70 (overbought)
- **Parameters**: RSI period = 14

### Real Data Example

#### 📊 Bar N - Signal Generated

**Time:** 2025-02-04 04:15:00

```
Open:  1054.25
High:  1054.25
Low:   1053.50
Close: 1053.75  ← RSI calculated here
Volume: 234

RSI: 30.00 (crosses below 30 threshold)
Previous RSI: 30.15

✅ SIGNAL: BUY (Long Entry)
📌 Action: Signal queued for next bar
```

**What Happened:**

- During this 15-minute bar, RSI crossed below 30
- At 04:30:00 (bar close), the signal is detected
- Signal is queued for execution at next bar's open

---

#### 🎯 Bar N+1 - Position Opened

**Time:** 2025-02-04 04:30:00

```
Open:  1053.75  ← ENTRY PRICE
High:  1053.75
Low:   1052.75
Close: 1053.25
Volume: 189

💰 Entry Details:
├─ Base Entry: 1053.75 (open price)
├─ Slippage (2 ticks × 0.25): +0.50 cents
└─ Actual Entry: 1054.25

Position: LONG 1 contract
Entry Time: 2025-02-04 04:30:00
Entry Price: 1054.25
```

**What Happened:**

- At 04:30:00, when this bar opened, the queued signal was executed
- Position opened at open price (1053.75) plus slippage
- Slippage accounts for execution delay and bid-ask spread

---

#### 📈 Following Bars

```
Bar N+2 (04:45:00):
├─ Open: 1053.00
├─ Close: 1053.50
└─ Position: Still holding

Bar N+3 (05:00:00):
├─ Open: 1053.50
├─ Close: 1054.00
└─ Position: Still holding (waiting for exit signal)
```

---

### Entry Calculation

```python
# Code execution in BaseStrategy._execute_queued_signal()

signal = 1  # Buy signal from previous bar
price_open = 1053.75  # Current bar's open

# Open position
self._open_new_position(direction=1, idx=current_time, price_open=price_open)

# Inside _open_new_position():
entry_price = self.position_manager.apply_slippage_to_entry_price(1, 1053.75)
# For long with ZS (tick_size = 0.25, slippage_ticks = 2):
# entry_price = 1053.75 + (2 × 0.25) = 1054.25

# Trade recorded:
{
    'entry_time': '2025-02-04 04:30:00',
    'entry_price': 1054.25,
    'side': 'long'
}
```

---

### Financial Impact

```
Entry Price: 1054.25 cents/bushel
Contract Size: 5,000 bushels
Contract Value: 1054.25 × 5,000 = $52,712.50

Slippage Cost: 0.50 cents × 5,000 = $25.00 per contract
```

---

## Example 2: EMA Crossover Strategy (5-minute bars)

### Strategy Logic

- **Buy Signal**: Fast EMA (9) crosses above Slow EMA (21)
- **Sell Signal**: Fast EMA (9) crosses below Slow EMA (21)

### Real Data Example

#### 📊 Bar N - Signal Generated

**Time:** 2025-04-14 03:25:00

```
Open:  1039.50
High:  1039.75
Low:   1039.25
Close: 1039.75  ← EMAs calculated here

EMA(9):  1038.61
EMA(21): 1038.61  ← Fast crosses above slow!

Previous Bar:
├─ EMA(9):  1038.59
└─ EMA(21): 1038.62

✅ SIGNAL: BUY (Bullish Crossover)
📌 Action: Signal queued for next bar
```

**Crossover Detection:**

```python
# In strategy code
prev_ema_fast <= prev_ema_slow  # 1038.59 <= 1038.62 = True
ema_fast > ema_slow  # 1038.61 > 1038.61 = True (by tiny margin)
# Result: Crossover detected!
```

---

#### 🎯 Bar N+1 - Position Opened

**Time:** 2025-04-14 03:30:00

```
Open:  1040.00  ← ENTRY PRICE
High:  1040.75
Low:   1039.75
Close: 1040.50

💰 Entry Details:
├─ Base Entry: 1040.00
├─ Slippage (2 ticks × 0.25): +0.50 cents
└─ Actual Entry: 1040.50

Position: LONG 1 contract
```

---

### Why 5-Minute Timeframe?

**Advantages:**

- ✅ More trade opportunities
- ✅ Faster reaction to market changes
- ✅ Good for day trading strategies

**Considerations:**

- ⚠️ More noise in the data
- ⚠️ Higher commission impact (more trades)
- ⚠️ Requires tighter stop losses

---

## Example 3: MACD Strategy (2-hour bars)

### Strategy Logic

- **Buy Signal**: MACD line crosses above Signal line
- **Sell Signal**: MACD line crosses below Signal line
- **Parameters**: MACD(12,26,9)

### Real Data Example

#### 📊 Bar N - Signal Generated

**Time:** 2023-01-06 04:00:00

```
Open:  1477.50
High:  1478.75
Low:   1476.50
Close: 1478.25  ← MACD calculated here

MACD Line: -9.07
Signal Line: -9.33  ← MACD crosses above!

Previous Bar:
├─ MACD: -9.35
└─ Signal: -9.25

✅ SIGNAL: BUY (Bullish Momentum)
📌 Action: Signal queued for next bar
```

**Interpretation:**

- Both MACD and Signal line are negative (below zero)
- But MACD is moving up and crosses above Signal
- This indicates weakening bearish momentum → potential reversal

---

#### 🎯 Bar N+1 - Position Opened

**Time:** 2023-01-06 06:00:00

```
Open:  1478.00  ← ENTRY PRICE
High:  1484.50
Low:   1477.25
Close: 1483.25

💰 Entry Details:
├─ Base Entry: 1478.00
├─ Slippage (2 ticks × 0.25): +0.50 cents
└─ Actual Entry: 1478.50

Position: LONG 1 contract
```

---

### Why 2-Hour Timeframe?

**Advantages:**

- ✅ Filters out intraday noise
- ✅ More reliable trend signals
- ✅ Good for swing trading
- ✅ Less affected by random volatility

**Considerations:**

- ⚠️ Slower to react to market changes
- ⚠️ Larger stop losses needed
- ⚠️ Fewer trading opportunities

---

## Example 4: Bollinger Bands Strategy (Daily bars)

### Strategy Logic

- **Buy Signal**: Price crosses below Lower Band (oversold)
- **Sell Signal**: Price crosses above Upper Band (overbought)
- **Parameters**: SMA(20), 2 standard deviations

### Real Data Example

#### 📊 Bar N - Signal Generated

**Time:** 2020-03-09 (Daily bar)

```
Open:  883.50
High:  885.25
Low:   869.75
Close: 870.00  ← Bollinger Bands calculated here

Upper Band: 912.94
SMA(20):    894.46
Lower Band: 875.98  ← Price crosses below!

Previous Close: 882.50 (above lower band)

✅ SIGNAL: BUY (Oversold Condition)
📌 Action: Signal queued for next day
```

**Market Context:**

- This was during the March 2020 COVID-19 crash
- Extreme volatility causing sharp price drops
- Price broke below lower band = severe oversold

---

#### 🎯 Bar N+1 - Position Opened

**Time:** 2020-03-10 (Next Day)

```
Open:  874.50  ← ENTRY PRICE
High:  878.25
Low:   870.00
Close: 876.25

💰 Entry Details:
├─ Base Entry: 874.50
├─ Slippage (2 ticks × 0.25): +0.50 cents
└─ Actual Entry: 875.00

Position: LONG 1 contract
```

**What Happened:**

- Market opened slightly higher (874.50 vs 870.00 previous close)
- Position entered at open with slippage
- This is a mean reversion play (expecting price to return to SMA)

---

### Why Daily Timeframe?

**Advantages:**

- ✅ Most reliable signals
- ✅ Less noise, clearer trends
- ✅ Suitable for position trading
- ✅ Lower commission impact
- ✅ Less time-intensive monitoring

**Considerations:**

- ⚠️ Very few trading opportunities
- ⚠️ Large capital requirements (larger stops)
- ⚠️ Slower profit realization

---

## Position Exit Examples

### Exit Scenario 1: Opposite Signal (RSI Example)

```
Current Position: LONG from RSI < 30 signal

Bar N (Exit Signal Generated):
├─ Time: 2025-02-04 08:45:00
├─ Close: 1065.00
├─ RSI: 70.05 (crosses above 70)
└─ Signal: SELL (close long, open short)

Bar N+1 (Position Closed & Reversed):
├─ Time: 2025-02-04 09:00:00
├─ Open: 1065.25  ← EXIT PRICE
├─ Exit with slippage (2 ticks × 0.25): 1064.75
└─ Immediately opens SHORT at 1064.75

Trade Summary:
├─ Entry: 1054.25 (2025-02-04 04:30:00)
├─ Exit: 1064.75 (2025-02-04 09:00:00)
├─ Duration: 4.5 hours
├─ Profit: 10.50 cents/bushel
└─ P&L: 10.50 × 5,000 = $525.00
```

### Exit Scenario 2: Trailing Stop (If Enabled)

```
Position: LONG from 1054.25
Trailing Stop: 2% below high

Bar 1 (09:15:00):
├─ High: 1066.00
└─ Trailing Stop: 1066.00 × 0.98 = 1044.68

Bar 2 (09:30:00):
├─ High: 1068.50  ← New high!
└─ Trailing Stop: 1068.50 × 0.98 = 1047.13  ← Moves up

Bar 3 (09:45:00):
├─ High: 1067.00
├─ Low: 1046.50  ← Hits trailing stop!
└─ Exit: 1047.13 (trailing stop price)

Trade Summary:
├─ Entry: 1054.25
├─ Exit: 1047.13 (trailing stop)
├─ Loss: 7.12 cents/bushel
└─ P&L: -$356.00
```

### Exit Scenario 3: Contract Rollover (Conservative)

```
Position: LONG from 1054.25
Current Date: 2025-10-17 20:15:00 (Last bar before switch)
Switch Date: 2025-10-20 01:00:00

Last Bar Before Switch (2025-10-17 20:15:00):
├─ Open: 1062.00  ← EXIT PRICE (conservative)
├─ Close: 1063.50
└─ Next bar will trigger switch

Position Closed:
├─ Exit Price: 1062.00 (open of last bar)
├─ Exit Time: 2025-10-17 20:15:00
├─ Reason: Contract expiration approaching

If Rollover Enabled:
└─ New position opened on 2025-10-20 02:00:00 in next contract

Trade Summary:
├─ Entry: 1054.25
├─ Exit: 1062.00 (contract switch)
├─ Profit: 7.75 cents/bushel
└─ P&L: $387.50
```

---

## Slippage Impact

### What Is Slippage?

Slippage accounts for:

1. **Execution Delay**: Time between signal and order fill
2. **Bid-Ask Spread**: Difference between buy and sell prices
3. **Market Movement**: Price changes during order execution
4. **Liquidity**: Market depth affects fill prices

### Tick-Based Slippage Model

Unlike percentage-based slippage, **tick-based slippage** uses the minimum price movement (tick size) for each contract.

**Why Tick-Based?**

- ✅ More realistic for futures trading
- ✅ Consistent across price levels
- ✅ Reflects actual market mechanics
- ✅ Easier to validate against real execution data

### Tick Sizes by Contract

```python
# Example tick sizes (in price points)
ZS(Soybeans): 0.25
cents / bushel
ZC(Corn): 0.25
cents / bushel
CL(Crude
Oil): 0.01
dollars / barrel
GC(Gold): 0.10
dollars / troy
oz
ES(E - mini
S & P): 0.25
index
points
```

### Slippage Calculation

**Formula:**

```
slippage_amount = slippage_ticks × tick_size
adjusted_price = base_price ± slippage_amount
```

#### Long Entry (Buying)

```python
# You pay MORE than the open price
# Example: ZS with 2 ticks slippage
tick_size = 0.25  # cents
slippage_ticks = 2
entry_price = 1053.75 + (2 × 0.25) = 1054.25
```

#### Long Exit (Selling)

```python
# You receive LESS than the exit price
# Example: ZS with 2 ticks slippage
exit_price = 1065.25 - (2 × 0.25) = 1064.75
```

#### Short Entry (Selling)

```python
# You receive LESS than the open price
# Example: ZS with 2 ticks slippage
entry_price = 1053.75 - (2 × 0.25) = 1053.25
```

#### Short Exit (Buying to cover)

```python
# You pay MORE than the exit price
# Example: ZS with 2 ticks slippage
exit_price = 1065.25 + (2 × 0.25) = 1065.75
```

### Slippage Impact Example

```
Strategy: RSI (15-minute bars)
Symbol: ZS (Soybeans, tick_size = 0.25)
Slippage: 2 ticks

Without Slippage:
├─ Entry: 1053.75
├─ Exit: 1065.25
├─ Profit: 11.50 cents
└─ P&L: $575.00

With Slippage (2 ticks):
├─ Entry: 1054.25 (+0.50)
├─ Exit: 1064.75 (-0.50)
├─ Profit: 10.50 cents
└─ P&L: $525.00

Slippage Cost: $50.00 (8.7% of profit)
```

**Key Takeaway:** Even 2 ticks of slippage (realistic for liquid markets) reduces profitability by 8.7%.
Higher-frequency strategies need tighter slippage assumptions.

### Realistic Slippage Values

**Liquid Markets (ES, NQ, CL, GC):**

- Market orders during active hours: **1-2 ticks**
- Market orders during quiet hours: **2-3 ticks**
- Large orders: **3-5 ticks**

**Agricultural Futures (ZS, ZC, ZW):**

- Standard orders: **2-3 ticks**
- Electronic trading hours: **2-4 ticks**
- Less liquid symbols: **3-5 ticks**

**Conservative Testing:**

- Use **2-3 ticks** for most strategies
- Use **3-5 ticks** for high-frequency or large positions
- Validate with actual execution data when possible

---

## Contract Rollover Example

### Scenario: Quarterly Contract Expiration

```
Current Contract: ZSX25 (November 2025)
Next Contract: ZSF26 (January 2026)
Switch Date: 2025-10-20 01:00:00
```

### Timeline Without Rollover

```
2025-10-17 20:15:00 - Last Bar Before Switch
├─ Position: LONG from 1054.25
├─ Current Price: Open=1062.00, Close=1063.50
└─ Action: Close position at 1062.00 (conservative)

2025-10-20 02:00:00 - First Bar of New Contract
├─ No position (previous closed)
└─ Wait for new signal in new contract

Result:
├─ Position closed
├─ Exit: 1062.00
└─ P&L: +$387.50
```

### Timeline With Rollover Enabled

```
2025-10-17 20:15:00 - Last Bar Before Switch
├─ Position: LONG from 1054.25
├─ Current Price: Open=1062.00, Close=1063.50
├─ Action: Close position at 1062.00
└─ Mark for rollover

2025-10-20 02:00:00 - First Bar of New Contract (ZSF26)
├─ New contract open: 1061.50
├─ Action: Reopen LONG position
└─ Entry: 1062.00 (open + 2 ticks slippage = 1061.50 + 0.50)

Result:
├─ Old position closed: +$387.50
├─ New position opened in ZSF26
└─ Continuous exposure maintained
```

### Rollover Gap Risk

```
Old Contract Last Close: 1063.50
New Contract First Open: 1061.50
Gap: -2.00 cents (typical)

Impact on Rollover:
├─ Exited old: 1062.00
├─ Entered new: 1062.00 (with 2 ticks slippage)
└─ Rollover cost: 0.50 cents = $25.00
```

---

## Key Takeaways

### ✅ Signal Execution Model

1. **1-Bar Delay**: Signal on Bar N close → Execution on Bar N+1 open
2. **Slippage Applied**: Realistic friction modeling
3. **Conservative**: Exits early on rollover (at open of last bar)

### ✅ Timeframe Selection

- **5-minute**: Day trading, high frequency, more noise
- **15-minute**: Intraday, good balance
- **2-hour**: Swing trading, reliable signals
- **Daily**: Position trading, most reliable

### ✅ Strategy Behavior

- **RSI**: Mean reversion (oversold/overbought)
- **EMA Crossover**: Trend following
- **MACD**: Momentum confirmation
- **Bollinger Bands**: Volatility-based mean reversion

### ✅ Cost Considerations

- **Slippage**: Tick-based (default 2 ticks per trade)
- **Commission**: $4 per trade (in/out = $8 total)
- **Rollover**: Additional costs at contract switches

### ✅ Risk Management

- **Trailing Stops**: Dynamic exit based on favorable movement
- **Contract Rollover**: Conservative exit timing
- **Signal Discipline**: No position changes mid-bar

---

## Related Documentation

- **Architecture**: `BACKTESTING_ARCHITECTURE.md` - System design and components
- **Analysis**: `.github/prompts/BACKTESTING_ANALYSIS.md` - Code quality review
- **Code**: `base_strategy.py` - Implementation details

---

**Note**: All examples use real historical data from ZS (Soybean futures) traded on CBOT. Prices and dates are actual
market data, demonstrating realistic backtesting scenarios.
