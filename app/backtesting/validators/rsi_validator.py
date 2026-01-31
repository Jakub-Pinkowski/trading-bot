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


# ==================== RSI Validator ====================


class RSIValidator(Validator):
    """Validator for RSI strategy parameters."""

    # ==================== Validation Method ====================
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
        self.reset_warnings()

        # --- Type Validation ---

        # Validate RSI period is a positive integer
        self.validate_positive_integer(rsi_period, "rsi period")
        # Validate thresholds are numbers between 0 and 100
        self.validate_type_and_range(lower, "lower threshold", 0, 100)
        self.validate_type_and_range(upper, "upper threshold", 0, 100)

        # --- Threshold Relationship Validation ---

        # Ensure lower threshold is below upper threshold
        if lower >= upper:
            raise ValueError(f"Lower threshold ({lower}) must be less than upper threshold ({upper})")

        # --- RSI Period Range Validation ---

        # Warn if RSI period is too short
        if rsi_period < RSI_PERIOD_MIN_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is quite short and may generate excessive noise. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )
        # Warn if RSI period is too long
        elif rsi_period > RSI_PERIOD_MAX_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is quite long and may be too slow to catch trends. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )

        # --- Lower Threshold Range Validation ---

        # Warn if lower threshold is too aggressive
        if lower < RSI_LOWER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI lower threshold {lower} is very aggressive and may generate many false signals. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )
        # Warn if lower threshold is too conservative
        elif lower > RSI_LOWER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI lower threshold {lower} is very conservative and may miss opportunities. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )

        # --- Upper Threshold Range Validation ---

        # Warn if upper threshold is too aggressive
        if upper < RSI_UPPER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI upper threshold {upper} is very aggressive and may generate many false signals. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )
        # Warn if upper threshold is too conservative
        elif upper > RSI_UPPER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI upper threshold {upper} is very conservative and may miss opportunities. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )

        # --- Threshold Gap Validation ---

        # Calculate gap between thresholds
        gap = upper - lower

        # Warn if gap is too narrow
        if gap < RSI_GAP_MIN:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is quite narrow and may generate excessive signals. "
                f"Consider using a gap of at least {RSI_GAP_MIN} points (e.g., {RSI_LOWER_STANDARD}/{RSI_UPPER_STANDARD})."
            )
        # Warn if gap is too wide
        elif gap > RSI_GAP_MAX:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is very wide and may miss many opportunities. "
                f"Consider using a gap of {RSI_GAP_MIN}-{RSI_GAP_MAX} points."
            )

        return self.warnings
