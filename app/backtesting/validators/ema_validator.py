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


class EMAValidator(Validator):
    """Validator for EMA crossover strategy parameters."""

    def validate(self, ema_short, ema_long, **kwargs):
        """
        Enhanced validation for EMA crossover parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - Short EMA: 5-21 (9-12 are most common for short-term signals)
        - Long EMA: 15-50 (21-26 are most common for trend confirmation)
        - Ratio: Long should be 1.5-3x the short period for good separation

        Args:
            ema_short: Short EMA period
            ema_long: Long EMA period
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages
        """
        self.warnings = []

        # Short EMA validation
        if ema_short < EMA_SHORT_MIN:
            self.warnings.append(
                f"Short EMA period {ema_short} is very sensitive and may generate excessive noise. "
                f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common)."
            )
        elif ema_short > EMA_SHORT_MAX:
            self.warnings.append(
                f"Short EMA period {ema_short} may be too slow for crossover signals. "
                f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common)."
            )

        # Long EMA validation
        if ema_long < EMA_LONG_MIN:
            self.warnings.append(
                f"Long EMA period {ema_long} may be too short for trend confirmation. "
                f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common)."
            )
        elif ema_long > EMA_LONG_MAX:
            self.warnings.append(
                f"Long EMA period {ema_long} may be too slow and miss trend changes. "
                f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common)."
            )

        # Ratio validation
        ratio = ema_long / ema_short
        if ratio < EMA_RATIO_MIN:
            self.warnings.append(
                f"EMA ratio ({ratio:.1f}) is too close - periods {ema_short}/{ema_long} may generate false signals. "
                f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26)."
            )
        elif ratio > EMA_RATIO_MAX:
            self.warnings.append(
                f"EMA ratio ({ratio:.1f}) is very wide - periods {ema_short}/{ema_long} may be too slow. "
                f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26)."
            )

        return self.warnings
