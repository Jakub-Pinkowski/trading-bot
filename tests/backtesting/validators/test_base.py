"""
Tests for base validator module.

Tests cover:
- Module-level validation functions (validate_boolean, validate_positive_integer, etc.)
- Boolean vs numeric type safety (preventing bool from passing as int)
- Range validation with edge cases
- Optional parameter validation (None handling)
- Infinity and NaN rejection for float inputs
- Validator base class (initialization, warning management, abstract method)
"""

import pytest

from app.backtesting.validators.base import (
    Validator,
    validate_boolean,
    validate_non_negative_number,
    validate_optional_non_negative_number,
    validate_optional_positive_number,
    validate_positive_integer,
    validate_positive_number,
    validate_type_and_range,
    _is_bool_or_not_type,
)


# ==================== Helper Function Tests ====================

class TestIsBoolOrNotType:
    """Test _is_bool_or_not_type helper function."""

    def test_returns_true_for_boolean(self):
        """Boolean values should return True regardless of type check."""
        assert _is_bool_or_not_type(True, int) is True
        assert _is_bool_or_not_type(False, int) is True
        assert _is_bool_or_not_type(True, int, float) is True

    def test_returns_true_when_not_matching_type(self):
        """Non-boolean values not matching type should return True."""
        assert _is_bool_or_not_type("string", int) is True
        assert _is_bool_or_not_type([1, 2], int) is True
        assert _is_bool_or_not_type(None, int, float) is True

    def test_returns_false_when_matching_type(self):
        """Non-boolean values matching type should return False."""
        assert _is_bool_or_not_type(5, int) is False
        assert _is_bool_or_not_type(5.5, float) is False
        assert _is_bool_or_not_type(10, int, float) is False
        assert _is_bool_or_not_type(3.14, int, float) is False

    def test_multiple_type_check(self):
        """Test checking against multiple types."""
        # Integer matches int but not float
        assert _is_bool_or_not_type(5, float) is True
        assert _is_bool_or_not_type(5, int) is False

        # Float matches float but not int
        assert _is_bool_or_not_type(5.5, int) is True
        assert _is_bool_or_not_type(5.5, float) is False


# ==================== Boolean Validation Tests ====================

class TestValidateBoolean:
    """Test validate_boolean function."""

    def test_accepts_true(self):
        """True should be accepted as valid boolean."""
        validate_boolean(True, "test_param")  # Should not raise

    def test_accepts_false(self):
        """False should be accepted as valid boolean."""
        validate_boolean(False, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [
        1, 0, -1, 100,
        1.0, 0.0, -1.5,
        "true", "false", "True", "False",
        None, [], {}, "",
    ])
    def test_rejects_non_boolean(self, invalid_value):
        """Non-boolean values should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be a boolean"):
            validate_boolean(invalid_value, "test_param")

    def test_error_message_includes_param_name(self):
        """Error message should include the parameter name."""
        with pytest.raises(ValueError, match="my_flag must be a boolean"):
            validate_boolean(1, "my_flag")


# ==================== Positive Integer Validation Tests ====================

class TestValidatePositiveInteger:
    """Test validate_positive_integer function."""

    @pytest.mark.parametrize("valid_value", [1, 2, 10, 100, 1000])
    def test_accepts_positive_integers(self, valid_value):
        """Positive integers should be accepted."""
        validate_positive_integer(valid_value, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_value):
        """Zero and negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(invalid_value, "test_param")

    @pytest.mark.parametrize("invalid_type", [1.0, 1.5, "1", None, [1], True, False])
    def test_rejects_non_integers(self, invalid_type):
        """Non-integer types should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(invalid_type, "test_param")

    def test_rejects_boolean_true(self):
        """Boolean True (which equals 1) should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(True, "test_param")

    def test_rejects_boolean_false(self):
        """Boolean False (which equals 0) should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(False, "test_param")


# ==================== Positive Number Validation Tests ====================

class TestValidatePositiveNumber:
    """Test validate_positive_number function."""

    @pytest.mark.parametrize("valid_value", [1, 2, 100, 0.1, 0.01, 1.5, 100.99])
    def test_accepts_positive_numbers(self, valid_value):
        """Positive integers and floats should be accepted."""
        validate_positive_number(valid_value, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [0, 0.0, -1, -0.1, -100])
    def test_rejects_zero_and_negative(self, invalid_value):
        """Zero and negative numbers should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(invalid_value, "test_param")

    def test_rejects_infinity(self):
        """Infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_positive_number(float('inf'), "test_param")

    def test_rejects_negative_infinity(self):
        """Negative infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_positive_number(float('-inf'), "test_param")

    def test_rejects_nan(self):
        """NaN should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_positive_number(float('nan'), "test_param")

    @pytest.mark.parametrize("invalid_type", ["1", None, [1], True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(invalid_type, "test_param")

    def test_rejects_boolean_values(self):
        """Booleans should be rejected even though bool inherits from int."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(True, "test_param")
        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(False, "test_param")


# ==================== Non-Negative Number Validation Tests ====================

class TestValidateNonNegativeNumber:
    """Test validate_non_negative_number function."""

    @pytest.mark.parametrize("valid_value", [0, 0.0, 1, 2, 0.1, 1.5, 100])
    def test_accepts_zero_and_positive(self, valid_value):
        """Zero and positive numbers should be accepted."""
        validate_non_negative_number(valid_value, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [-1, -0.1, -100, -1000.5])
    def test_rejects_negative(self, invalid_value):
        """Negative numbers should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be a non-negative number"):
            validate_non_negative_number(invalid_value, "test_param")

    def test_rejects_infinity(self):
        """Infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_non_negative_number(float('inf'), "test_param")

    def test_rejects_nan(self):
        """NaN should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_non_negative_number(float('nan'), "test_param")

    @pytest.mark.parametrize("invalid_type", ["0", None, [0], True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be a non-negative number"):
            validate_non_negative_number(invalid_type, "test_param")


# ==================== Range Validation Tests ====================

class TestValidateTypeAndRange:
    """Test validate_type_and_range function."""

    def test_accepts_value_at_minimum(self):
        """Value at minimum boundary should be accepted."""
        validate_type_and_range(10, "test_param", 10, 100)  # Should not raise

    def test_accepts_value_at_maximum(self):
        """Value at maximum boundary should be accepted."""
        validate_type_and_range(100, "test_param", 10, 100)  # Should not raise

    def test_accepts_value_in_range(self):
        """Value within range should be accepted."""
        validate_type_and_range(50, "test_param", 10, 100)  # Should not raise
        validate_type_and_range(50.5, "test_param", 10.0, 100.0)  # Should not raise

    def test_rejects_value_below_minimum(self):
        """Value below minimum should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between 10 and 100"):
            validate_type_and_range(9, "test_param", 10, 100)

    def test_rejects_value_above_maximum(self):
        """Value above maximum should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between 10 and 100"):
            validate_type_and_range(101, "test_param", 10, 100)

    def test_rejects_infinity(self):
        """Infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_type_and_range(float('inf'), "test_param", 0, 100)

    def test_rejects_nan(self):
        """NaN should be rejected."""
        with pytest.raises(ValueError, match="test_param must be a finite number"):
            validate_type_and_range(float('nan'), "test_param", 0, 100)

    @pytest.mark.parametrize("invalid_type", ["50", None, [50], True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between 0 and 100"):
            validate_type_and_range(invalid_type, "test_param", 0, 100)

    def test_works_with_negative_ranges(self):
        """Should work with negative value ranges."""
        validate_type_and_range(-50, "test_param", -100, 0)  # Should not raise
        with pytest.raises(ValueError, match="test_param must be between -100 and 0"):
            validate_type_and_range(1, "test_param", -100, 0)

    def test_works_with_float_ranges(self):
        """Should work with float ranges."""
        validate_type_and_range(0.5, "test_param", 0.0, 1.0)  # Should not raise
        validate_type_and_range(0.99, "test_param", 0.0, 1.0)  # Should not raise


# ==================== Optional Positive Number Tests ====================

class TestValidateOptionalPositiveNumber:
    """Test validate_optional_positive_number function."""

    def test_accepts_none(self):
        """None should be accepted."""
        validate_optional_positive_number(None, "test_param")  # Should not raise

    @pytest.mark.parametrize("valid_value", [1, 2, 0.1, 0.01, 100, 1000.5])
    def test_accepts_positive_numbers(self, valid_value):
        """Positive numbers should be accepted."""
        validate_optional_positive_number(valid_value, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [0, 0.0, -1, -0.1, -100])
    def test_rejects_zero_and_negative(self, invalid_value):
        """Zero and negative numbers should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be None or a positive number"):
            validate_optional_positive_number(invalid_value, "test_param")

    def test_rejects_infinity(self):
        """Infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be None or a finite number"):
            validate_optional_positive_number(float('inf'), "test_param")

    def test_rejects_nan(self):
        """NaN should be rejected."""
        with pytest.raises(ValueError, match="test_param must be None or a finite number"):
            validate_optional_positive_number(float('nan'), "test_param")

    @pytest.mark.parametrize("invalid_type", ["1", [1], {}, True, False])
    def test_rejects_non_numeric_non_none_types(self, invalid_type):
        """Non-numeric types (except None) should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be None or a positive number"):
            validate_optional_positive_number(invalid_type, "test_param")


# ==================== Optional Non-Negative Number Tests ====================

class TestValidateOptionalNonNegativeNumber:
    """Test validate_optional_non_negative_number function."""

    def test_accepts_none(self):
        """None should be accepted."""
        validate_optional_non_negative_number(None, "test_param")  # Should not raise

    @pytest.mark.parametrize("valid_value", [0, 0.0, 1, 2, 0.1, 100])
    def test_accepts_zero_and_positive(self, valid_value):
        """Zero and positive numbers should be accepted."""
        validate_optional_non_negative_number(valid_value, "test_param")  # Should not raise

    @pytest.mark.parametrize("invalid_value", [-1, -0.1, -100, -0.001])
    def test_rejects_negative(self, invalid_value):
        """Negative numbers should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be None or a non-negative number"):
            validate_optional_non_negative_number(invalid_value, "test_param")

    def test_rejects_infinity(self):
        """Infinity should be rejected."""
        with pytest.raises(ValueError, match="test_param must be None or a finite number"):
            validate_optional_non_negative_number(float('inf'), "test_param")

    def test_rejects_nan(self):
        """NaN should be rejected."""
        with pytest.raises(ValueError, match="test_param must be None or a finite number"):
            validate_optional_non_negative_number(float('nan'), "test_param")

    @pytest.mark.parametrize("invalid_type", ["0", [0], {}, True, False])
    def test_rejects_non_numeric_non_none_types(self, invalid_type):
        """Non-numeric types (except None) should raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be None or a non-negative number"):
            validate_optional_non_negative_number(invalid_type, "test_param")


# ==================== Validator Base Class Tests ====================

class TestValidatorInitialization:
    """Test Validator class initialization."""

    def test_initializes_with_empty_warnings(self):
        """Validator should initialize with empty warnings list."""
        validator = Validator()

        assert hasattr(validator, 'warnings')
        assert validator.warnings == []
        assert isinstance(validator.warnings, list)

    def test_multiple_instances_have_separate_warnings(self):
        """Multiple validator instances should have independent warnings lists."""
        validator1 = Validator()
        validator2 = Validator()

        validator1.warnings.append("warning 1")
        validator2.warnings.append("warning 2")

        assert validator1.warnings == ["warning 1"]
        assert validator2.warnings == ["warning 2"]


class TestValidatorWarningManagement:
    """Test Validator warning management."""

    def test_reset_warnings_clears_list(self):
        """reset_warnings should clear the warnings list."""
        validator = Validator()
        validator.warnings = ["warning 1", "warning 2", "warning 3"]

        validator.reset_warnings()

        assert validator.warnings == []

    def test_reset_warnings_creates_new_empty_list(self):
        """reset_warnings should create a new empty list."""
        validator = Validator()
        old_warnings = validator.warnings
        validator.warnings.append("warning")

        validator.reset_warnings()

        # Should be empty but potentially a different list object
        assert validator.warnings == []
        assert len(validator.warnings) == 0

    def test_can_add_warnings_after_reset(self):
        """Should be able to add warnings after reset."""
        validator = Validator()
        validator.warnings.append("old warning")
        validator.reset_warnings()

        validator.warnings.append("new warning")

        assert validator.warnings == ["new warning"]


class TestValidatorAbstractMethod:
    """Test Validator abstract validate method."""

    def test_validate_raises_not_implemented_error(self):
        """Base Validator.validate() should raise NotImplementedError."""
        validator = Validator()

        with pytest.raises(NotImplementedError, match="Subclasses must implement validate\\(\\) method"):
            validator.validate()

    def test_validate_raises_with_args(self):
        """Base Validator.validate() should raise NotImplementedError even with args."""
        validator = Validator()

        with pytest.raises(NotImplementedError, match="Subclasses must implement validate\\(\\) method"):
            validator.validate(1, 2, 3)

    def test_validate_raises_with_kwargs(self):
        """Base Validator.validate() should raise NotImplementedError even with kwargs."""
        validator = Validator()

        with pytest.raises(NotImplementedError, match="Subclasses must implement validate\\(\\) method"):
            validator.validate(param1=1, param2=2)


# ==================== Edge Cases and Integration Tests ====================

class TestValidationEdgeCases:
    """Test edge cases across validation functions."""

    def test_very_large_positive_numbers(self):
        """Very large positive numbers should be accepted where valid."""
        large_int = 10 ** 10
        large_float = 10 ** 100.5

        validate_positive_integer(large_int, "test_param")
        validate_positive_number(large_int, "test_param")
        validate_positive_number(large_float, "test_param")

    def test_very_small_positive_numbers(self):
        """Very small positive numbers should be accepted."""
        tiny = 1e-10

        validate_positive_number(tiny, "test_param")
        validate_non_negative_number(tiny, "test_param")
        validate_optional_positive_number(tiny, "test_param")

    def test_exact_zero_handling(self):
        """Zero should be handled correctly by different validators."""
        # Should reject
        with pytest.raises(ValueError):
            validate_positive_integer(0, "test_param")
        with pytest.raises(ValueError):
            validate_positive_number(0, "test_param")
        with pytest.raises(ValueError):
            validate_optional_positive_number(0, "test_param")

        # Should accept
        validate_non_negative_number(0, "test_param")
        validate_non_negative_number(0.0, "test_param")
        validate_optional_non_negative_number(0, "test_param")

    def test_parameter_name_in_all_error_messages(self):
        """All validation functions should include parameter name in errors."""
        param_name = "my_custom_parameter"

        with pytest.raises(ValueError, match=param_name):
            validate_boolean(1, param_name)
        with pytest.raises(ValueError, match=param_name):
            validate_positive_integer(0, param_name)
        with pytest.raises(ValueError, match=param_name):
            validate_positive_number(-1, param_name)
        with pytest.raises(ValueError, match=param_name):
            validate_non_negative_number(-1, param_name)
        with pytest.raises(ValueError, match=param_name):
            validate_type_and_range(200, param_name, 0, 100)
        with pytest.raises(ValueError, match=param_name):
            validate_optional_positive_number(-1, param_name)
        with pytest.raises(ValueError, match=param_name):
            validate_optional_non_negative_number(-1, param_name)


class TestValidatorSubclass:
    """Test that Validator can be properly subclassed."""

    def test_can_create_custom_validator(self):
        """Should be able to create a custom validator subclass."""

        class CustomValidator(Validator):
            def validate(self, param1, param2):
                """Custom validation logic."""
                self.reset_warnings()
                if param1 < 0:
                    self.warnings.append("param1 is negative")
                if param2 > 100:
                    self.warnings.append("param2 is too large")
                return self.warnings

        validator = CustomValidator()
        warnings = validator.validate(-5, 150)

        assert len(warnings) == 2
        assert "param1 is negative" in warnings
        assert "param2 is too large" in warnings

    def test_subclass_inherits_warning_management(self):
        """Subclass should inherit warning management from base class."""

        class TestValidator(Validator):
            def validate(self, value):
                self.reset_warnings()
                if value < 0:
                    self.warnings.append("negative value")
                return self.warnings

        validator = TestValidator()

        # First validation
        result1 = validator.validate(-1)
        assert result1 == ["negative value"]

        # Second validation should reset warnings
        result2 = validator.validate(1)
        assert result2 == []

        # Third validation with warning
        result3 = validator.validate(-5)
        assert result3 == ["negative value"]
