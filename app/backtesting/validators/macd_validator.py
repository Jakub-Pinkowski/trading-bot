"""
MACD Parameter Validator

This module contains the MACD strategy parameter validator.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    MACD_FAST_MIN,
    MACD_FAST_MAX,
    MACD_FAST_STANDARD,
    MACD_SLOW_MIN,
    MACD_SLOW_MAX,
    MACD_SLOW_STANDARD,
    MACD_SIGNAL_MIN,
    MACD_SIGNAL_MAX,
    MACD_SIGNAL_STANDARD,
)


# ==================== MACD Validator ====================


class MACDValidator(Validator):
    """Validator for MACD strategy parameters."""

    # ==================== Validation Method ====================

    def validate(self, fast_period, slow_period, signal_period, **kwargs):
        """
        Enhanced validation for MACD parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - Fast period: 8-15 (12 is standard)
        - Slow period: 20-30 (26 is standard)
        - Signal period: 7-12 (9 is standard)
        - Standard MACD: 12/26/9 is most widely used

        Args:
            fast_period: MACD fast EMA period
            slow_period: MACD slow EMA period
            signal_period: MACD signal line period
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages

        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.reset_warnings()

        # --- Type Validation ---

        # Validate all MACD periods are positive integers
        self.validate_positive_integer(fast_period, "fast period")
        self.validate_positive_integer(slow_period, "slow period")
        self.validate_positive_integer(signal_period, "signal period")

        # --- Period Relationship Validation ---

        # Ensure fast period is actually faster than slow period
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")

        # --- Fast Period Range Validation ---

        # Warn if fast period is too small
        if fast_period < MACD_FAST_MIN:
            self.warnings.append(
                f"MACD fast period {fast_period} is very short and may generate excessive noise. "
                f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard)."
            )
        # Warn if fast period is too large
        elif fast_period > MACD_FAST_MAX:
            self.warnings.append(
                f"MACD fast period {fast_period} may be too slow for responsive signals. "
                f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard)."
            )

        # --- Slow Period Range Validation ---

        # Warn if slow period is too small
        if slow_period < MACD_SLOW_MIN:
            self.warnings.append(
                f"MACD slow period {slow_period} may be too short for trend confirmation. "
                f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard)."
            )
        # Warn if slow period is too large
        elif slow_period > MACD_SLOW_MAX:
            self.warnings.append(
                f"MACD slow period {slow_period} may be too slow and miss trend changes. "
                f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard)."
            )

        # --- Signal Period Range Validation ---

        # Warn if signal period is too small
        if signal_period < MACD_SIGNAL_MIN:
            self.warnings.append(
                f"MACD signal period {signal_period} is very short and may generate false signals. "
                f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard)."
            )
        # Warn if signal period is too large
        elif signal_period > MACD_SIGNAL_MAX:
            self.warnings.append(
                f"MACD signal period {signal_period} may be too slow for timely signals. "
                f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard)."
            )

        # --- Standard Configuration Check ---

        # Provide info when using standard MACD configuration
        if (fast_period, slow_period, signal_period) == (MACD_FAST_STANDARD, MACD_SLOW_STANDARD, MACD_SIGNAL_STANDARD):
            self.warnings.append(
                f"Using standard MACD parameters ({MACD_FAST_STANDARD}/{MACD_SLOW_STANDARD}/{MACD_SIGNAL_STANDARD}) - widely tested and reliable."
            )

        return self.warnings
