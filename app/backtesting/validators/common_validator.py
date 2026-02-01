"""
Common Parameters Validator

This module contains the validator for common strategy parameters shared across all strategies.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    TRAILING_STOP_MIN,
    TRAILING_STOP_MAX,
    TRAILING_STOP_COMMON_MIN,
    TRAILING_STOP_COMMON_MAX,
    SLIPPAGE_MAX,
    SLIPPAGE_TYPICAL_MIN,
    SLIPPAGE_TYPICAL_MAX,
)


# ==================== Common Parameters Validator ====================


class CommonValidator(Validator):
    """Validator for common strategy parameters (rollover, trailing, slippage)."""

    # ==================== Validation Method ====================

    def validate(self, rollover, trailing, slippage, **kwargs):
        """
        Enhanced validation for common strategy parameters with guidance.

        Reasonable ranges:
        - Rollover: Boolean (True for continuous contracts, False for single contract)
        - Trailing: None or 1-5% (2-3% is common for futures)
        - Slippage: 0-0.5% (0.1-0.2% is typical for liquid futures)

        Args:
            rollover: Whether to use contract rollover
            trailing: Trailing stop percentage (or None)
            slippage: Slippage percentage (0 = no slippage)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages

        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.reset_warnings()

        # --- Type Validation ---

        # Validate rollover is a boolean value
        self.validate_boolean(rollover, "rollover")

        # Validate trailing is None or a positive number
        if trailing is not None:
            self.validate_positive_number(trailing, "trailing")

        # Validate slippage is a non-negative number (zero allowed)
        self.validate_non_negative_number(slippage, "slippage")

        # --- Trailing Stop Validation ---

        # Only validate trailing stop if it's provided
        if trailing is not None:
            # Warn if trailing stop is too tight
            if trailing < TRAILING_STOP_MIN:
                self.warnings.append(
                    f"Trailing stop {trailing}% is too tight and may be stopped out frequently. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )
            # Warn if trailing stop is too wide
            elif trailing > TRAILING_STOP_MAX:
                self.warnings.append(
                    f"Trailing stop {trailing}% is too wide and may give back large profits. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )

        # --- Slippage Validation ---

        # Warn if slippage is 0 (unrealistically low)
        if slippage == 0:
            self.warnings.append(
                f"Slippage is set to 0% (no slippage). This is unrealistic for live trading. "
                f"Consider using {SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% for liquid futures."
            )
        # Warn if slippage is unrealistically low
        elif 0 < slippage < SLIPPAGE_TYPICAL_MIN:
            self.warnings.append(
                f"Slippage {slippage}% is very low and may be unrealistic. "
                f"Typical range is {SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% for liquid futures."
            )
        # Warn if slippage is unrealistically high
        elif slippage > SLIPPAGE_MAX:
            self.warnings.append(
                f"Slippage {slippage}% is too high and may significantly impact returns. "
                f"Consider using 0-{SLIPPAGE_MAX}% range "
                f"({SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% is typical for liquid futures)."
            )

        return self.warnings
