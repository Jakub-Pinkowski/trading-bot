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


class MACDValidator(Validator):
    """Validator for MACD strategy parameters."""

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
        self.warnings = []

        # Fast period type and value validation
        if not isinstance(fast_period, int):
            raise ValueError(f"fast period must be a positive integer, got {type(fast_period).__name__}")
        if fast_period <= 0:
            raise ValueError("fast period must be a positive integer")

        # Slow period type and value validation
        if not isinstance(slow_period, int):
            raise ValueError(f"slow period must be a positive integer, got {type(slow_period).__name__}")
        if slow_period <= 0:
            raise ValueError("slow period must be a positive integer")

        # Signal period type and value validation
        if not isinstance(signal_period, int):
            raise ValueError(f"signal period must be a positive integer, got {type(signal_period).__name__}")
        if signal_period <= 0:
            raise ValueError("signal period must be a positive integer")

        # Business logic validation
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")

        # Fast period validation
        if fast_period < MACD_FAST_MIN:
            self.warnings.append(
                f"MACD fast period {fast_period} is very short and may generate excessive noise. "
                f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard)."
            )
        elif fast_period > MACD_FAST_MAX:
            self.warnings.append(
                f"MACD fast period {fast_period} may be too slow for responsive signals. "
                f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard)."
            )

        # Slow period validation
        if slow_period < MACD_SLOW_MIN:
            self.warnings.append(
                f"MACD slow period {slow_period} may be too short for trend confirmation. "
                f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard)."
            )
        elif slow_period > MACD_SLOW_MAX:
            self.warnings.append(
                f"MACD slow period {slow_period} may be too slow and miss trend changes. "
                f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard)."
            )

        # Signal period validation
        if signal_period < MACD_SIGNAL_MIN:
            self.warnings.append(
                f"MACD signal period {signal_period} is very short and may generate false signals. "
                f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard)."
            )
        elif signal_period > MACD_SIGNAL_MAX:
            self.warnings.append(
                f"MACD signal period {signal_period} may be too slow for timely signals. "
                f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard)."
            )

        # Standard combination check
        if (fast_period, slow_period, signal_period) == (MACD_FAST_STANDARD, MACD_SLOW_STANDARD, MACD_SIGNAL_STANDARD):
            self.warnings.append(
                f"Using standard MACD parameters ({MACD_FAST_STANDARD}/{MACD_SLOW_STANDARD}/{MACD_SIGNAL_STANDARD}) - widely tested and reliable."
            )

        return self.warnings
