"""
Base Validator Class

This module contains the base Validator class that provides reusable validation
methods for all strategy parameter validators. All specific validators should
inherit from this base class.
"""

import math


# ==================== Module-Level Helper Functions ====================


def validate_boolean(value, param_name):
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


def _is_bool_or_not_type(value, *types):
    """
    Check if the value is boolean or not one of the given types.

    This helper prevents booleans from passing type checks since
    bool inherits from int in Python (True == 1, False == 0).

    Args:
        value: The value to check
        *types: One or more types to check against

    Returns:
        True if value is a boolean or not an instance of any given type
    """
    return isinstance(value, bool) or not isinstance(value, types)


# ==================== Validation Functions ====================


def validate_positive_integer(value, param_name):
    """
    Validate that a parameter is a positive integer.

    Args:
        value: The value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is not a positive integer
    """
    if _is_bool_or_not_type(value, int) or value <= 0:
        raise ValueError(f"{param_name} must be a positive integer")


def validate_positive_number(value, param_name):
    """
    Validate that a parameter is a positive number (int or float).

    Args:
        value: The value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is not positive
    """
    if _is_bool_or_not_type(value, int, float):
        raise ValueError(f"{param_name} must be positive")

    # Reject infinity and NaN values for float inputs
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{param_name} must be a finite number")

    # Validate value is positive
    if value <= 0:
        raise ValueError(f"{param_name} must be positive")


def validate_non_negative_number(value, param_name):
    """
    Validate that a parameter is a non-negative number (int or float), zero allowed.

    Args:
        value: The value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is not non-negative
    """
    if _is_bool_or_not_type(value, int, float):
        raise ValueError(f"{param_name} must be a non-negative number")

    # Reject infinity and NaN values for float inputs
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{param_name} must be a finite number")

    # Validate value is non-negative (zero is allowed)
    if value < 0:
        raise ValueError(f"{param_name} must be a non-negative number")


def validate_type_and_range(value, param_name, minimum_value, maximum_value):
    """
    Validate that a parameter is a number within a specified range.

    Args:
        value: The value to validate
        param_name: Parameter name for error messages
        minimum_value: Minimum value (inclusive)
        maximum_value: Maximum value (inclusive)

    Raises:
        ValueError: If the value is not a number or is outside the range
    """
    if _is_bool_or_not_type(value, int, float):
        raise ValueError(f"{param_name} must be between {minimum_value} and {maximum_value}")

    # Reject infinity and NaN values for float inputs
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{param_name} must be a finite number")

    # Validate value is within range
    if value < minimum_value or value > maximum_value:
        raise ValueError(f"{param_name} must be between {minimum_value} and {maximum_value}")


def validate_optional_positive_number(value, param_name):
    """
    Validate that a parameter is None or a positive number.

    Args:
        value: The value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is not None and not a positive number
    """
    # Only validate if a value is provided
    if value is not None:
        if _is_bool_or_not_type(value, int, float):
            raise ValueError(f"{param_name} must be None or a positive number")

        # Reject infinity and NaN values for float inputs
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{param_name} must be None or a finite number")

        # Validate value is positive
        if value <= 0:
            raise ValueError(f"{param_name} must be None or a positive number")


def validate_optional_non_negative_number(value, param_name):
    """
    Validate that a parameter is None or a non-negative number.

    Args:
        value: The value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is not None and not a non-negative number
    """
    # Only validate if a value is provided
    if value is not None:
        if _is_bool_or_not_type(value, int, float):
            raise ValueError(f"{param_name} must be None or a non-negative number")

        # Reject infinity and NaN values for float inputs
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{param_name} must be None or a finite number")

        # Validate value is non-negative (zero is allowed)
        if value < 0:
            raise ValueError(f"{param_name} must be None or a non-negative number")


# ==================== Base Validator Class ====================


class Validator:
    """
    Base class for strategy parameter validators.

    Provides warning management for validation results. Subclasses implement
    strategy-specific validation rules using the module-level validation functions.
    Validators collect warnings without raising exceptions, allowing users to proceed
    with cautionary parameters.
    """

    # ==================== Initialization ====================

    def __init__(self):
        """Initialize the validator with an empty warnings list."""
        self.warnings = []

    # ==================== Warning Management ====================

    def reset_warnings(self):
        """Reset the warnings list at the start of validation."""
        self.warnings = []

    # ==================== Abstract Method ====================

    def validate(self, *args, **kwargs):
        """
        Validate parameters. Must be implemented by subclasses.

        Subclasses override this method with their own specific signature.
        For example,
        - CommonValidator.validate(rollover, trailing, slippage_ticks, **kwargs)
        - RSIValidator.validate(rsi_period, lower_threshold, upper_threshold, **kwargs)

        Args:
            *args: Positional arguments (defined by subclass)
            **kwargs: Additional keyword arguments (defined by subclass)

        Returns:
            List of warning messages

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement validate() method")
