"""
Base Validator Class

This module contains the base Validator class that provides reusable validation
methods for all strategy parameter validators. All specific validators should
inherit from this base class.
"""

import math


# ==================== Base Validator Class ====================


class Validator:
    """Base class for parameter validators with reusable validation methods."""

    # ==================== Initialization ====================

    def __init__(self):
        """Initialize the validator with an empty warnings list."""
        self.warnings = []

    # ==================== Warning Management ====================

    def reset_warnings(self):
        """Reset the warnings list at the start of validation."""
        self.warnings = []

    # ==================== Range And Gap Validation ====================

    def validate_range(
        self, value, name, minimum_value, maximum_value,
        recommended_minimum, recommended_maximum
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

        Raises:
            ValueError: If value is outside absolute min/max range
        """
        # Validate value is within absolute bounds
        if value < minimum_value or value > maximum_value:
            raise ValueError(f"{name} must be between {minimum_value} and {maximum_value}")

        # Add warning if value is below recommended minimum
        if value < recommended_minimum:
            self.warnings.append(f"{name} {value} is below recommended range ({recommended_minimum}-{recommended_maximum})")
        # Add warning if value is above recommended maximum
        elif value > recommended_maximum:
            self.warnings.append(f"{name} {value} is above recommended range ({recommended_minimum}-{recommended_maximum})")

    def validate_gap(
        self, gap, name, minimum_gap, maximum_gap
    ):
        """
        Validate a gap/difference between two parameters.

        Args:
            gap: The gap value to validate
            name: Gap name for messages
            minimum_gap: Minimum recommended gap
            maximum_gap: Maximum recommended gap
        """
        # Add warning if gap is too small
        if gap < minimum_gap:
            self.warnings.append(f"{name} ({gap}) is below recommended minimum {minimum_gap}")
        # Add warning if gap is too large
        elif gap > maximum_gap:
            self.warnings.append(f"{name} ({gap}) is above recommended maximum {maximum_gap}")

    def validate_ratio(
        self, ratio, name, minimum_ratio, maximum_ratio
    ):
        """
        Validate a ratio between two parameters.

        Args:
            ratio: The ratio value to validate
            name: Ratio name for messages
            minimum_ratio: Minimum recommended ratio
            maximum_ratio: Maximum recommended ratio
        """
        # Add warning if ratio is too small
        if ratio < minimum_ratio:
            self.warnings.append(f"{name} ratio ({ratio:.1f}) is below recommended minimum {minimum_ratio}")
        # Add warning if ratio is too large
        elif ratio > maximum_ratio:
            self.warnings.append(f"{name} ratio ({ratio:.1f}) is above recommended maximum {maximum_ratio}")

    # ==================== Message Handling ====================

    def add_info_message(self, message):
        """
        Add an informational message to warnings.

        Args:
            message: The informational message to add
        """
        self.warnings.append(message)

    # ==================== Type Checking Helpers ====================

    def _is_bool_or_not_type(self, value, *types):
        """
        Check if value is boolean or not one of the given types.

        This helper prevents booleans from passing type checks since
        bool inherits from int in Python (True == 1, False == 0).

        Args:
            value: The value to check
            *types: One or more types to check against

        Returns:
            True if value is a boolean or not an instance of any given type
        """
        return isinstance(value, bool) or not isinstance(value, types)

    # ==================== Type Validation ====================

    def validate_positive_integer(self, value, param_name):
        """
        Validate that a parameter is a positive integer.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not a positive integer
        """
        if self._is_bool_or_not_type(value, int) or value <= 0:
            raise ValueError(f"{param_name} must be a positive integer")

    def validate_positive_number(self, value, param_name):
        """
        Validate that a parameter is a positive number (int or float).

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not positive
        """
        if self._is_bool_or_not_type(value, int, float):
            raise ValueError(f"{param_name} must be positive")

        # Reject infinity and NaN values for float inputs
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{param_name} must be a finite number")

        # Validate value is positive
        if value <= 0:
            raise ValueError(f"{param_name} must be positive")

    def validate_non_negative_number(self, value, param_name):
        """
        Validate that a parameter is a non-negative number (int or float), zero allowed.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not non-negative
        """
        if self._is_bool_or_not_type(value, int, float):
            raise ValueError(f"{param_name} must be a non-negative number")

        # Reject infinity and NaN values for float inputs
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{param_name} must be a finite number")

        # Validate value is non-negative (zero is allowed)
        if value < 0:
            raise ValueError(f"{param_name} must be a non-negative number")

    def validate_boolean(self, value, param_name):
        """
        Validate that a parameter is a boolean.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError(f"{param_name} must be a boolean (True or False)")

    def validate_type_and_range(self, value, param_name, minimum_value, maximum_value):
        """
        Validate that a parameter is a number within a specified range.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages
            minimum_value: Minimum value (inclusive)
            maximum_value: Maximum value (inclusive)

        Raises:
            ValueError: If value is not a number or is outside the range
        """
        if self._is_bool_or_not_type(value, int, float):
            raise ValueError(f"{param_name} must be between {minimum_value} and {maximum_value}")

        # Reject infinity and NaN values for float inputs
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{param_name} must be a finite number")

        # Validate value is within range
        if value < minimum_value or value > maximum_value:
            raise ValueError(f"{param_name} must be between {minimum_value} and {maximum_value}")

    # ==================== Optional Value Validation ====================

    def validate_optional_positive_number(self, value, param_name):
        """
        Validate that a parameter is None or a positive number.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not None and not a positive number
        """
        # Only validate if value is provided
        if value is not None:
            if self._is_bool_or_not_type(value, int, float):
                raise ValueError(f"{param_name} must be None or a positive number")

            # Reject infinity and NaN values for float inputs
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"{param_name} must be None or a finite number")

            # Validate value is positive
            if value <= 0:
                raise ValueError(f"{param_name} must be None or a positive number")

    def validate_optional_non_negative_number(self, value, param_name):
        """
        Validate that a parameter is None or a non-negative number.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If value is not None and not a non-negative number
        """
        # Only validate if value is provided
        if value is not None:
            if self._is_bool_or_not_type(value, int, float):
                raise ValueError(f"{param_name} must be None or a non-negative number")

            # Reject infinity and NaN values for float inputs
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"{param_name} must be None or a finite number")

            # Validate value is non-negative (zero is allowed)
            if value < 0:
                raise ValueError(f"{param_name} must be None or a non-negative number")

    # ==================== Abstract Method ====================

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
