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
            slippage: Slippage percentage (or None)
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
        self.validate_optional_positive_number(trailing, "trailing")
        # Validate slippage is None or a non-negative number (zero allowed)
        self.validate_optional_non_negative_number(slippage, "slippage")

        # --- Trailing Stop Validation ---

        # Only validate trailing stop if it's provided
        if trailing is not None:
            # Warn if trailing stop is too tight
            if trailing < TRAILING_STOP_MIN:
                self.warnings.append(
                    f"Trailing stop {trailing}% is very tight and may be stopped out frequently. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )
            # Warn if trailing stop is too wide
            elif trailing > TRAILING_STOP_MAX:
                self.warnings.append(
                    f"Trailing stop {trailing}% is very wide and may give back large profits. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )

        # --- Slippage Validation ---

        # Only validate slippage if it's provided
        if slippage is not None:
            # Warn if slippage is unrealistically high
            if slippage > SLIPPAGE_MAX:
                self.warnings.append(
                    f"Slippage {slippage}% is very high and may significantly impact returns. "
                    f"Consider using 0-{SLIPPAGE_MAX}% range "
                    f"({SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% is typical for liquid futures)."
                )
            # Warn if slippage is unrealistically zero
            elif slippage == 0:
                self.warnings.append(
                    f"Slippage {slippage}% is unrealistic - all orders experience some slippage. "
                    f"Consider using {SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% for liquid futures."
                )

        return self.warnings
