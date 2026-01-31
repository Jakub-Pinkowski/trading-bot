"""
RSI Parameter Validator

This module contains the RSI strategy parameter validator.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    RSI_PERIOD_MIN_RECOMMENDED,
    RSI_PERIOD_MAX_RECOMMENDED,
    RSI_PERIOD_STANDARD,
    RSI_LOWER_MIN_AGGRESSIVE,
    RSI_LOWER_MAX_CONSERVATIVE,
    RSI_LOWER_STANDARD,
    RSI_UPPER_MIN_AGGRESSIVE,
    RSI_UPPER_MAX_CONSERVATIVE,
    RSI_UPPER_STANDARD,
    RSI_GAP_MIN,
    RSI_GAP_MAX,
)


class RSIValidator(Validator):
    """Validator for RSI strategy parameters."""

    def validate(self, rsi_period, lower, upper, **kwargs):
        """
        Enhanced validation for RSI parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - RSI Period: 10-30 (14 is the most common, shorter periods = more sensitive)
        - Lower threshold: 20-40 (30 is standard, lower = more aggressive)
        - Upper threshold: 60-80 (70 is standard, higher = more conservative)
        - Threshold gap: Should be at least 20 points to avoid excessive signals

        Args:
            rsi_period: RSI calculation period
            lower: Lower oversold threshold
            upper: Upper overbought threshold
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages
        
        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.warnings = []

        # RSI period type and value validation
        if not isinstance(rsi_period, int):
            raise ValueError(f"rsi period must be a positive integer, got {type(rsi_period).__name__}")
        if rsi_period <= 0:
            raise ValueError("rsi period must be a positive integer")

        # Lower threshold type and value validation
        if not isinstance(lower, (int, float)):
            raise ValueError(f"lower threshold must be a number, got {type(lower).__name__}")
        if lower < 0 or lower > 100:
            raise ValueError("lower threshold must be between 0 and 100")

        # Upper threshold type and value validation
        if not isinstance(upper, (int, float)):
            raise ValueError(f"upper threshold must be a number, got {type(upper).__name__}")
        if upper < 0 or upper > 100:
            raise ValueError("upper threshold must be between 0 and 100")

        # Business logic validation
        if lower >= upper:
            raise ValueError(f"Lower threshold ({lower}) must be less than upper threshold ({upper})")

        # RSI Period validation
        if rsi_period < RSI_PERIOD_MIN_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is quite short and may generate excessive noise. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )
        elif rsi_period > RSI_PERIOD_MAX_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is quite long and may be too slow to catch trends. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )

        # Lower threshold validation
        if lower < RSI_LOWER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI lower threshold {lower} is very aggressive and may generate many false signals. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )
        elif lower > RSI_LOWER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI lower threshold {lower} is very conservative and may miss opportunities. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )

        # Upper threshold validation
        if upper < RSI_UPPER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI upper threshold {upper} is very aggressive and may generate many false signals. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )
        elif upper > RSI_UPPER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI upper threshold {upper} is very conservative and may miss opportunities. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )

        # Threshold gap validation
        gap = upper - lower
        if gap < RSI_GAP_MIN:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is quite narrow and may generate excessive signals. "
                f"Consider using a gap of at least {RSI_GAP_MIN} points (e.g., {RSI_LOWER_STANDARD}/{RSI_UPPER_STANDARD})."
            )
        elif gap > RSI_GAP_MAX:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is very wide and may miss many opportunities. "
                f"Consider using a gap of {RSI_GAP_MIN}-{RSI_GAP_MAX} points."
            )

        return self.warnings
