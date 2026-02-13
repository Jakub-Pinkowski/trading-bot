"""
Common Parameters Validator

This module contains the validator for common strategy parameters shared across all strategies.
"""

from app.backtesting.validators.base import (Validator,
                                             validate_boolean,
                                             validate_non_negative_number,
                                             validate_positive_number)
from app.backtesting.validators.constants import (
    TRAILING_STOP_MIN,
    TRAILING_STOP_MAX,
    TRAILING_STOP_COMMON_MIN,
    TRAILING_STOP_COMMON_MAX,
)


# ==================== Common Parameters Validator ====================


class CommonValidator(Validator):
    """Validator for common strategy parameters (rollover, trailing, slippage)."""

    # ==================== Validation Method ====================

    def validate(self, rollover, trailing, slippage_ticks, **kwargs):
        """
        Enhanced validation for common strategy parameters with guidance.

        Reasonable ranges:
        - Rollover: Boolean (True for continuous contracts, False for single contract)
        - Trailing: None or 1-5% (2-3% is common for futures)
        - Slippage ticks: 0-10 ticks (1-3 ticks is typical for liquid futures)

        Args:
            rollover: Whether to use contract rollover
            trailing: Trailing stop percentage (or None)
            slippage_ticks: Slippage in ticks (0 = no slippage)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages

        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.reset_warnings()

        # --- Type Validation ---

        # Validate rollover is a boolean value
        validate_boolean(rollover, "rollover")

        # Validate trailing is None or a positive number
        if trailing is not None:
            validate_positive_number(trailing, "trailing")

        # Validate slippage_ticks is a non-negative number (zero allowed)
        validate_non_negative_number(slippage_ticks, "slippage_ticks")

        # --- Trailing Stop Validation ---

        # Only validate trailing stop if it's provided
        if trailing is not None:
            # Warn if the trailing stop is too tight
            if trailing < TRAILING_STOP_MIN:
                self.warnings.append(
                    f"Trailing stop {trailing}% is too tight and may be stopped out frequently. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )
            # Warn if the trailing stop is too wide
            elif trailing > TRAILING_STOP_MAX:
                self.warnings.append(
                    f"Trailing stop {trailing}% is too wide and may give back large profits. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )

        # --- Slippage Ticks Validation ---

        # Warn if slippage is too high
        if slippage_ticks > 10:
            self.warnings.append(
                f"Slippage of {slippage_ticks} ticks is very high. "
                f"Consider using 1-5 ticks (1-3 ticks is typical for liquid futures)."
            )
        elif slippage_ticks > 5:
            self.warnings.append(
                f"Slippage of {slippage_ticks} ticks is high. "
                f"Consider 1-3 ticks for liquid futures."
            )

        return self.warnings
