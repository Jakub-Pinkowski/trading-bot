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
    def validate(self, rsi_period, lower_threshold, upper_threshold, **kwargs):
        """
        Enhanced validation for RSI parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - RSI Period: 10-30 (14 is the most common, shorter periods = more sensitive)
        - Lower threshold: 20-40 (30 is standard, lower = more aggressive)
        - Upper threshold: 60-80 (70 is standard, higher = more conservative)
        - Threshold gap: Should be at least 20 points to avoid excessive signals

        Args:
            rsi_period: RSI calculation period
            lower_threshold: Lower oversold threshold
            upper_threshold: Upper overbought threshold
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
        self.validate_type_and_range(lower_threshold, "lower threshold", 0, 100)
        self.validate_type_and_range(upper_threshold, "upper threshold", 0, 100)

        # --- Threshold Relationship Validation ---

        # Ensure lower threshold is below upper threshold
        if lower_threshold >= upper_threshold:
            raise ValueError(f"Lower threshold ({lower_threshold}) must be less than upper threshold ({upper_threshold})")

        # --- RSI Period Range Validation ---

        # Warn if RSI period is too short
        if rsi_period < RSI_PERIOD_MIN_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is too short and may generate excessive noise. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )
        # Warn if RSI period is too long
        elif rsi_period > RSI_PERIOD_MAX_RECOMMENDED:
            self.warnings.append(
                f"RSI period {rsi_period} is too long and may miss trends. "
                f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                f"({RSI_PERIOD_STANDARD} is most common)."
            )

        # --- Lower Threshold Range Validation ---

        # Warn if lower threshold is too aggressive
        if lower_threshold < RSI_LOWER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI lower threshold {lower_threshold} is too aggressive and may generate false signals. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )
        # Warn if the lower threshold is too conservative
        elif lower_threshold > RSI_LOWER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI lower threshold {lower_threshold} is too conservative and may miss opportunities. "
                f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                f"({RSI_LOWER_STANDARD} is standard)."
            )

        # --- Upper Threshold Range Validation ---

        # Warn if the upper threshold is too aggressive
        if upper_threshold < RSI_UPPER_MIN_AGGRESSIVE:
            self.warnings.append(
                f"RSI upper threshold {upper_threshold} is too aggressive and may generate false signals. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )
        # Warn if the upper threshold is too conservative
        elif upper_threshold > RSI_UPPER_MAX_CONSERVATIVE:
            self.warnings.append(
                f"RSI upper threshold {upper_threshold} is too conservative and may miss opportunities. "
                f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                f"({RSI_UPPER_STANDARD} is standard)."
            )

        # --- Threshold Gap Validation ---

        # Calculate gap between thresholds
        gap = upper_threshold - lower_threshold

        # Warn if the gap is too narrow
        if gap < RSI_GAP_MIN:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is too narrow and may generate excessive signals. "
                f"Consider using a gap of at least {RSI_GAP_MIN} points (e.g., {RSI_LOWER_STANDARD}/{RSI_UPPER_STANDARD})."
            )
        # Warn if tgap is too wide
        elif gap > RSI_GAP_MAX:
            self.warnings.append(
                f"RSI threshold gap ({gap}) is too wide and may miss opportunities. "
                f"Consider using a gap of {RSI_GAP_MIN}-{RSI_GAP_MAX} points."
            )

        return self.warnings
