"""
EMA Crossover Parameter Validator

This module contains the EMA crossover strategy parameter validator.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    EMA_SHORT_MIN,
    EMA_SHORT_MAX,
    EMA_SHORT_COMMON_MIN,
    EMA_SHORT_COMMON_MAX,
    EMA_LONG_MIN,
    EMA_LONG_MAX,
    EMA_LONG_COMMON_MIN,
    EMA_LONG_COMMON_MAX,
    EMA_RATIO_MIN,
    EMA_RATIO_MAX,
)


# ==================== EMA Crossover Validator ====================


class EMAValidator(Validator):
    """
    Validator for EMA (Exponential Moving Average) crossover strategy parameters.

    Validates short and long EMA periods ensuring they are properly ordered
    (short < long) with adequate separation for meaningful crossover signals.
    Also validates common parameters (rollover, trailing, slippage).
    """

    # ==================== Validation Method ====================

    def validate(self, short_ema_period, long_ema_period, **kwargs):
        """
        Enhanced validation for EMA crossover parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - Short EMA: 5-21 (9-12 are most common for short-term signals)
        - Long EMA: 15-50 (21-26 are most common for trend confirmation)
        - Ratio: Long should be 1.5-3x the short period for good separation

        Args:
            short_ema_period: Short EMA period
            long_ema_period: Long EMA period
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages

        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.reset_warnings()

        # --- Type Validation ---

        # Validate both EMA periods are positive integers
        self.validate_positive_integer(short_ema_period, "short EMA period")
        self.validate_positive_integer(long_ema_period, "long EMA period")

        # --- Period Relationship Validation ---

        # Ensure short EMA is actually shorter than long EMA
        if short_ema_period >= long_ema_period:
            raise ValueError(f"Short EMA period ({short_ema_period}) must be less than long EMA period ({long_ema_period})")

        # --- Short EMA Range Validation ---

        # Warn if short EMA period is too small
        if short_ema_period < EMA_SHORT_MIN:
            self.warnings.append(
                f"Short EMA period {short_ema_period} is too short and may generate excessive noise. "
                f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common)."
            )
        # Warn if short EMA period is too large
        elif short_ema_period > EMA_SHORT_MAX:
            self.warnings.append(
                f"Short EMA period {short_ema_period} is too long and may miss crossover signals. "
                f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common)."
            )

        # --- Long EMA Range Validation ---

        # Warn if long EMA period is too small
        if long_ema_period < EMA_LONG_MIN:
            self.warnings.append(
                f"Long EMA period {long_ema_period} is too short and may not confirm trends. "
                f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common)."
            )
        # Warn if long EMA period is too large
        elif long_ema_period > EMA_LONG_MAX:
            self.warnings.append(
                f"Long EMA period {long_ema_period} is too long and may miss trend changes. "
                f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common)."
            )

        # --- Ratio Validation ---

        # Calculate ratio between long and short periods
        ratio = long_ema_period / short_ema_period

        # Warn if periods are too close together
        if ratio < EMA_RATIO_MIN:
            self.warnings.append(
                f"EMA ratio ({ratio:.1f}) is too close - periods {short_ema_period}/{long_ema_period} may generate false signals. "
                f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26)."
            )
        # Warn if periods are too far apart
        elif ratio > EMA_RATIO_MAX:
            self.warnings.append(
                f"EMA ratio ({ratio:.1f}) is too wide and may miss signals. "
                f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26)."
            )

        return self.warnings
