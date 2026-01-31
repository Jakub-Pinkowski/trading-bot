"""
Base Validator Class

This module contains the base Validator class that provides reusable validation
methods for all strategy parameter validators. All specific validators should
inherit from this base class.
"""


class Validator:
    """Base class for parameter validators with reusable validation methods."""

    def __init__(self):
        """Initialize the validator with an empty warnings list."""
        self.warnings = []

    def validate_range(
        self, value, name, minimum_value, maximum_value,
        recommended_minimum, recommended_maximum,
        minimum_message=None, maximum_message=None
    ):
        """
        Reusable range validation with warnings for recommended ranges.

        Args:
            value: The value to validate
            name: Parameter name for error messages
            minimum_value: Absolute minimum value (raises error if violated)
            maximum_value: Absolute maximum value (raises error if violated)
            recommended_minimum: Recommended minimum value (generates warning if below)
            recommended_maximum: Recommended maximum value (generates warning if above)
            minimum_message: Optional custom message for below recommended minimum
            maximum_message: Optional custom message for above recommended maximum

        Raises:
            ValueError: If value is outside absolute min/max range
        """
        # Check absolute bounds
        if value < minimum_value or value > maximum_value:
            raise ValueError(f"{name} must be between {minimum_value} and {maximum_value}")

        # Check recommended range and add warnings
        if value < recommended_minimum:
            if minimum_message:
                self.warnings.append(minimum_message)
            else:
                self.warnings.append(f"{name} {value} is below recommended range ({recommended_minimum}-{recommended_maximum})")
        elif value > recommended_maximum:
            if maximum_message:
                self.warnings.append(maximum_message)
            else:
                self.warnings.append(f"{name} {value} is above recommended range ({recommended_minimum}-{recommended_maximum})")

    def validate_gap(
        self, gap, name, minimum_gap, maximum_gap,
        minimum_message=None, maximum_message=None
    ):
        """
        Validate a gap/difference between two parameters.

        Args:
            gap: The gap value to validate
            name: Gap name for messages
            minimum_gap: Minimum recommended gap
            maximum_gap: Maximum recommended gap
            minimum_message: Optional custom message for gap too small
            maximum_message: Optional custom message for gap too large
        """
        if gap < minimum_gap:
            if minimum_message:
                self.warnings.append(minimum_message)
            else:
                self.warnings.append(f"{name} ({gap}) is below recommended minimum {minimum_gap}")
        elif gap > maximum_gap:
            if maximum_message:
                self.warnings.append(maximum_message)
            else:
                self.warnings.append(f"{name} ({gap}) is above recommended maximum {maximum_gap}")

    def validate_ratio(
        self, ratio, name, minimum_ratio, maximum_ratio,
        minimum_message=None, maximum_message=None
    ):
        """
        Validate a ratio between two parameters.

        Args:
            ratio: The ratio value to validate
            name: Ratio name for messages
            minimum_ratio: Minimum recommended ratio
            maximum_ratio: Maximum recommended ratio
            minimum_message: Optional custom message for ratio too small
            maximum_message: Optional custom message for ratio too large
        """
        if ratio < minimum_ratio:
            if minimum_message:
                self.warnings.append(minimum_message)
            else:
                self.warnings.append(f"{name} ratio ({ratio:.1f}) is below recommended minimum {minimum_ratio}")
        elif ratio > maximum_ratio:
            if maximum_message:
                self.warnings.append(maximum_message)
            else:
                self.warnings.append(f"{name} ratio ({ratio:.1f}) is above recommended maximum {maximum_ratio}")

    def add_info_message(self, message):
        """
        Add an informational message to warnings.

        Args:
            message: The informational message to add
        """
        self.warnings.append(message)

    def validate(self, **kwargs):
        """
        Validate parameters. Must be implemented by subclasses.

        Args:
            **kwargs: Strategy-specific parameters

        Returns:
            List of warning messages

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement validate() method")
