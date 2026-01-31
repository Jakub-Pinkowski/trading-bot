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


class CommonValidator(Validator):
    """Validator for common strategy parameters (rollover, trailing, slippage)."""

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
            ValueError: If rollover is not a boolean
        """
        self.warnings = []

        # Rollover validation
        if not isinstance(rollover, bool):
            raise ValueError(f"rollover must be a boolean (True or False), got {type(rollover).__name__}")

        # Trailing stop validation
        if trailing is not None:
            if trailing < TRAILING_STOP_MIN:
                self.warnings.append(
                    f"Trailing stop {trailing}% is very tight and may be stopped out frequently. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )
            elif trailing > TRAILING_STOP_MAX:
                self.warnings.append(
                    f"Trailing stop {trailing}% is very wide and may give back large profits. "
                    f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                    f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures)."
                )

        # Slippage validation
        if slippage is not None:
            if slippage > SLIPPAGE_MAX:
                self.warnings.append(
                    f"Slippage {slippage}% is very high and may significantly impact returns. "
                    f"Consider using 0-{SLIPPAGE_MAX}% range "
                    f"({SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% is typical for liquid futures)."
                )
            elif slippage == 0:
                self.warnings.append(
                    f"Slippage {slippage}% is unrealistic - all orders experience some slippage. "
                    f"Consider using {SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% for liquid futures."
                )

        return self.warnings
