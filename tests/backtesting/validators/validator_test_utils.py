"""
Validator Test Utilities

Reusable test utilities for validator testing. These functions consolidate
common test patterns across all validator test files, reducing duplication
and improving maintainability.
"""


# ==================== Validator Base Attribute Assertions ====================

def assert_validator_base_attributes(validator):
    """
    Assert that validator has all required base attributes.

    Checks for:
    - warnings list attribute
    - reset_warnings method
    - validate method
    - warnings initialized as empty list

    Args:
        validator: Validator instance to check

    Raises:
        AssertionError: If any required attribute is missing

    Example:
        validator = RSIValidator()
        assert_validator_base_attributes(validator)
    """
    assert hasattr(validator, 'warnings'), "Validator missing 'warnings' attribute"
    assert hasattr(validator, 'reset_warnings'), "Validator missing 'reset_warnings' method"
    assert hasattr(validator, 'validate'), "Validator missing 'validate' method"
    assert validator.warnings == [], f"Validator warnings not initialized as empty list: {validator.warnings}"


def assert_validators_independent(validator_class, validate_args_1, validate_args_2):
    """
    Assert that multiple validator instances maintain independent state.

    Creates two validator instances, validates each with different parameters,
    and verifies their warning states are independent.

    Args:
        validator_class: Validator class to instantiate
        validate_args_1: Dict of validation arguments for first instance
        validate_args_2: Dict of validation arguments for second instance

    Raises:
        AssertionError: If validator instances share state

    Example:
        assert_validators_independent(
            RSIValidator,
            {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70},
            {'rsi_period': 10, 'lower_threshold': 20, 'upper_threshold': 80}
        )
    """
    validator1 = validator_class()
    validator2 = validator_class()

    validator1.validate(**validate_args_1)
    validator2.validate(**validate_args_2)

    # Validators should have different warning states (unless both have no warnings)
    assert validator1.warnings != validator2.warnings or (
            len(validator1.warnings) == 0 and len(validator2.warnings) == 0
    ), "Validator instances should maintain independent warning state"


# ==================== Warning Reset Assertions ====================

def assert_warnings_reset_between_calls(validator, call_with_warnings, call_without_warnings):
    """
    Assert that validator warnings reset between validate() calls.

    Performs three validations:
    1. Call that generates warnings
    2. Call that generates no warnings (verifies reset)
    3. Call that generates warnings again (verifies reset persistence)

    Args:
        validator: Validator instance
        call_with_warnings: Dict of validation args that trigger warnings
        call_without_warnings: Dict of validation args that don't trigger warnings

    Raises:
        AssertionError: If warnings don't reset properly

    Example:
        assert_warnings_reset_between_calls(
            RSIValidator(),
            {'rsi_period': 5, 'lower_threshold': 30, 'upper_threshold': 70},  # Triggers warning
            {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70}  # No warning
        )
    """
    # First validation with warnings
    warnings1 = validator.validate(**call_with_warnings)
    assert len(warnings1) > 0, "First call should generate warnings"

    # Second validation without warnings
    warnings2 = validator.validate(**call_without_warnings)
    assert len(warnings2) == 0, "Second call should have no warnings (reset failed)"

    # Third validation with warnings again
    warnings3 = validator.validate(**call_with_warnings)
    assert len(warnings3) > 0, "Third call should generate warnings again"


def assert_warnings_list_fresh(validator, validate_args):
    """
    Assert that each validate() call returns an independent warnings list.

    Validates twice with same arguments, then modifies one list to verify
    they are independent objects (not references to same list).

    Args:
        validator: Validator instance
        validate_args: Dict of validation arguments

    Raises:
        AssertionError: If warnings lists share references

    Example:
        assert_warnings_list_fresh(
            RSIValidator(),
            {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70}
        )
    """
    warnings1 = validator.validate(**validate_args)
    warnings2 = validator.validate(**validate_args)

    # Should have same content but be independent lists
    assert warnings1 == warnings2, "Warnings content should be identical"

    # Modify one list and verify the other is unchanged
    warnings1.append("extra")
    assert len(warnings1) != len(warnings2), "Warnings lists should be independent (shared reference detected)"


# ==================== Range Warning Assertions ====================

def assert_no_range_warning_at_boundary(warnings, param_name):
    """
    Assert that warnings list contains no range warnings for specified parameter.

    Filters warnings to find those mentioning the parameter name along with
    range indicators ('too short', 'too long', 'too narrow', 'too wide',
    'too small', 'too large', 'too aggressive', 'too conservative').

    Args:
        warnings: List of warning strings
        param_name: Parameter name to check (e.g., 'period', 'fast period')

    Raises:
        AssertionError: If range warnings found for parameter

    Example:
        validator = BollingerValidator()
        warnings = validator.validate(period=MIN_PERIOD, number_of_standard_deviations=2.0)
        assert_no_range_warning_at_boundary(warnings, 'period')
    """
    range_indicators = [
        'too short', 'too long', 'too narrow', 'too wide',
        'too small', 'too large', 'too aggressive', 'too conservative'
    ]

    param_warnings = [
        w for w in warnings
        if param_name.lower() in w.lower() and any(indicator in w.lower() for indicator in range_indicators)
    ]

    assert len(param_warnings) == 0, (
        f"Expected no range warnings for '{param_name}' at boundary, "
        f"but found: {param_warnings}"
    )


def assert_has_warning_containing(warnings, *keywords):
    """
    Assert that warnings list contains at least one warning with all specified keywords.

    Args:
        warnings: List of warning strings
        *keywords: Keywords that should all appear in at least one warning (case-insensitive)

    Raises:
        AssertionError: If no warning contains all keywords

    Example:
        assert_has_warning_containing(warnings, 'period', 'too short', 'noise')
    """
    for warning in warnings:
        if all(keyword.lower() in warning.lower() for keyword in keywords):
            return  # Found a warning with all keywords

    # No warning found with all keywords
    keywords_str = "', '".join(keywords)
    assert False, f"No warning found containing all keywords: ['{keywords_str}']. Warnings: {warnings}"


# ==================== Standard Settings Verification ====================

def assert_standard_settings_no_warnings(validator, validate_args):
    """
    Assert that standard/recommended settings generate no warnings.

    This is a common test pattern: verify that the "standard" or "traditional"
    parameter values (e.g., RSI 14/30/70, MACD 12/26/9) don't trigger warnings.

    Args:
        validator: Validator instance
        validate_args: Dict of validation arguments (standard settings)

    Raises:
        AssertionError: If warnings are generated for standard settings

    Example:
        assert_standard_settings_no_warnings(
            RSIValidator(),
            {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70}
        )
    """
    warnings = validator.validate(**validate_args)
    assert len(warnings) == 0, (
        f"Standard settings should not generate warnings, but got: {warnings}"
    )
