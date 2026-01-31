"""
Bollinger Bands Parameter Validator

This module contains the Bollinger Bands strategy parameter validator.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    BB_PERIOD_MIN,
    BB_PERIOD_MAX,
    BB_PERIOD_STANDARD,
    BB_STD_MIN,
    BB_STD_MAX,
    BB_STD_STANDARD,
)


class BollingerValidator(Validator):
    """Validator for Bollinger Bands strategy parameters."""

    def validate(self, period, number_of_standard_deviations, **kwargs):
        """
        Enhanced validation for Bollinger Bands parameters with guidance on reasonable ranges.

        Reasonable ranges based on common trading practices:
        - Period: 15-25 (20 is standard)
        - Standard deviations: 1.5-2.5 (2.0 is standard)
        - Standard BB: 20/2.0 captures ~95% of price action

        Args:
            period: Moving average period for Bollinger Bands
            number_of_standard_deviations: Number of standard deviations for bands
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages
        
        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.warnings = []

        # Period type and value validation
        if not isinstance(period, int):
            raise ValueError(f"period must be a positive integer, got {type(period).__name__}")
        if period <= 0:
            raise ValueError("period must be a positive integer")

        # Standard deviation type and value validation
        if not isinstance(number_of_standard_deviations, (int, float)):
            raise ValueError(f"number of standard deviations must be a positive number, got {type(number_of_standard_deviations).__name__}")
        if number_of_standard_deviations <= 0:
            raise ValueError("number of standard deviations must be positive")

        # Period validation
        if period < BB_PERIOD_MIN:
            self.warnings.append(
                f"Bollinger Bands period {period} is quite short and may be too sensitive. "
                f"Consider using {BB_PERIOD_MIN}-{BB_PERIOD_MAX} range ({BB_PERIOD_STANDARD} is standard)."
            )
        elif period > BB_PERIOD_MAX:
            self.warnings.append(
                f"Bollinger Bands period {period} is quite long and may be too slow. "
                f"Consider using {BB_PERIOD_MIN}-{BB_PERIOD_MAX} range ({BB_PERIOD_STANDARD} is standard)."
            )

        # Standard deviation validation
        if number_of_standard_deviations < BB_STD_MIN:
            self.warnings.append(
                f"Bollinger Bands standard deviation {number_of_standard_deviations} is quite narrow and may generate excessive signals. "
                f"Consider using {BB_STD_MIN}-{BB_STD_MAX} range ({BB_STD_STANDARD} is standard)."
            )
        elif number_of_standard_deviations > BB_STD_MAX:
            self.warnings.append(
                f"Bollinger Bands standard deviation {number_of_standard_deviations} is quite wide and may miss opportunities. "
                f"Consider using {BB_STD_MIN}-{BB_STD_MAX} range ({BB_STD_STANDARD} is standard)."
            )

        # Standard combination check
        if (period, number_of_standard_deviations) == (BB_PERIOD_STANDARD, BB_STD_STANDARD):
            self.warnings.append(
                f"Using standard Bollinger Bands parameters ({BB_PERIOD_STANDARD}/{BB_STD_STANDARD}) - captures ~95% of price action."
            )

        return self.warnings
