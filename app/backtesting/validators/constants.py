"""
Validation Constants for Strategy Parameters

This module contains all validation thresholds and constants used for parameter validation
in the backtesting system. Constants are organized by strategy type with clear rationale
for each threshold based on common trading practices and extensive backtesting.

When adding new strategies or indicators, add their validation constants to this file
following the same pattern: MIN, MAX, STANDARD, and any specific thresholds.
"""

# ==================== RSI Parameter Validation ====================
# Rationale: Based on common trading practices and extensive backtesting

RSI_PERIOD_MIN_RECOMMENDED = 10  # Below this: too sensitive to noise
RSI_PERIOD_MAX_RECOMMENDED = 30  # Above this: too slow to catch trends
RSI_PERIOD_STANDARD = 14  # Most widely used RSI period

RSI_LOWER_MIN_AGGRESSIVE = 20  # Below this: very aggressive oversold level
RSI_LOWER_MAX_CONSERVATIVE = 40  # Above this: may miss oversold opportunities
RSI_LOWER_STANDARD = 30  # Most common oversold threshold

RSI_UPPER_MIN_AGGRESSIVE = 60  # Below this: very aggressive overbought level
RSI_UPPER_MAX_CONSERVATIVE = 80  # Above this: may miss overbought opportunities
RSI_UPPER_STANDARD = 70  # Most common overbought threshold

RSI_GAP_MIN = 20  # Minimum recommended gap between thresholds
RSI_GAP_MAX = 50  # Maximum recommended gap between thresholds

# ==================== EMA Crossover Parameter Validation ====================
# Rationale: Based on common moving average crossover strategies

EMA_SHORT_MIN = 5  # Below this: excessive noise
EMA_SHORT_MAX = 21  # Above this: too slow for short-term signals
EMA_SHORT_COMMON_MIN = 9  # Most common range lower bound
EMA_SHORT_COMMON_MAX = 12  # Most common range upper bound

EMA_LONG_MIN = 15  # Below this: too short for trend confirmation
EMA_LONG_MAX = 50  # Above this: too slow to catch trend changes
EMA_LONG_COMMON_MIN = 21  # Most common range lower bound
EMA_LONG_COMMON_MAX = 26  # Most common range upper bound

EMA_RATIO_MIN = 1.5  # Minimum ratio between long and short EMAs
EMA_RATIO_MAX = 3.0  # Maximum ratio between long and short EMAs

# ==================== MACD Parameter Validation ====================
# Rationale: Based on Gerald Appel's original MACD parameters (12/26/9)

MACD_FAST_MIN = 8  # Below this: excessive noise
MACD_FAST_MAX = 15  # Above this: too slow for fast line
MACD_FAST_STANDARD = 12  # Standard MACD fast period

MACD_SLOW_MIN = 20  # Below this: too short for trend confirmation
MACD_SLOW_MAX = 30  # Above this: too slow to detect trend changes
MACD_SLOW_STANDARD = 26  # Standard MACD slow period

MACD_SIGNAL_MIN = 7  # Below this: excessive false signals
MACD_SIGNAL_MAX = 12  # Above this: too slow for timely signals
MACD_SIGNAL_STANDARD = 9  # Standard MACD signal period

# ==================== Bollinger Bands Parameter Validation ====================
# Rationale: Based on John Bollinger's original parameters (20/2.0)

BB_PERIOD_MIN = 15  # Below this: too sensitive
BB_PERIOD_MAX = 25  # Above this: too slow
BB_PERIOD_STANDARD = 20  # Standard Bollinger Bands period

BB_STD_MIN = 1.5  # Below this: bands too narrow, excessive signals
BB_STD_MAX = 2.5  # Above this: bands too wide, miss opportunities
BB_STD_STANDARD = 2.0  # Standard deviation (captures ~95% of price action)

# ==================== Ichimoku Parameter Validation ====================
# Rationale: Based on traditional Japanese Ichimoku settings (9/26/52/26)

ICHIMOKU_TENKAN_MIN = 7  # Below this: too sensitive
ICHIMOKU_TENKAN_MAX = 12  # Above this: too slow for conversion line
ICHIMOKU_TENKAN_STANDARD = 9  # Traditional Tenkan-sen period

ICHIMOKU_KIJUN_MIN = 22  # Below this: too short for baseline
ICHIMOKU_KIJUN_MAX = 30  # Above this: too slow for trend confirmation
ICHIMOKU_KIJUN_STANDARD = 26  # Traditional Kijun-sen period

ICHIMOKU_SENKOU_B_MIN = 44  # Below this: cloud too short
ICHIMOKU_SENKOU_B_MAX = 60  # Above this: cloud too slow
ICHIMOKU_SENKOU_B_STANDARD = 52  # Traditional Senkou Span B period

ICHIMOKU_DISPLACEMENT_MIN = 22  # Below this: cloud projection too short
ICHIMOKU_DISPLACEMENT_MAX = 30  # Above this: cloud projects too far
ICHIMOKU_DISPLACEMENT_STANDARD = 26  # Traditional displacement (matches Kijun-sen)

ICHIMOKU_TENKAN_KIJUN_RATIO_MIN = 2.5  # Minimum ratio for proper separation
ICHIMOKU_TENKAN_KIJUN_RATIO_MAX = 3.5  # Maximum ratio for traditional proportions

ICHIMOKU_KIJUN_SENKOU_RATIO_MIN = 1.8  # Minimum ratio for proper cloud
ICHIMOKU_KIJUN_SENKOU_RATIO_MAX = 2.2  # Maximum ratio for traditional proportions

# ==================== Common Parameters Validation ====================
# Rationale: Based on typical futures trading practices

TRAILING_STOP_MIN = 1.0  # Below this: too tight, frequent stop-outs
TRAILING_STOP_MAX = 5.0  # Above this: too wide, large profit give-backs
TRAILING_STOP_COMMON_MIN = 2.0  # Common range lower bound
TRAILING_STOP_COMMON_MAX = 3.0  # Common range upper bound

SLIPPAGE_MAX = 0.5  # Above this: unrealistically high slippage
SLIPPAGE_TYPICAL_MIN = 0.1  # Typical range lower bound for liquid futures
SLIPPAGE_TYPICAL_MAX = 0.2  # Typical range upper bound for liquid futures

# ==================== Future Strategy Constants ====================
# Add new strategy validation constants below this section following the same pattern:
#
# STRATEGY_NAME_PARAMETER_MIN = value  # Description
# STRATEGY_NAME_PARAMETER_MAX = value  # Description
# STRATEGY_NAME_PARAMETER_STANDARD = value  # Description
#
# Example for a future Stochastic strategy:
# STOCHASTIC_K_PERIOD_MIN = 5
# STOCHASTIC_K_PERIOD_MAX = 21
# STOCHASTIC_K_PERIOD_STANDARD = 14
# STOCHASTIC_D_PERIOD_MIN = 3
# STOCHASTIC_D_PERIOD_MAX = 9
# STOCHASTIC_D_PERIOD_STANDARD = 3
