"""
Tests for common_validator module.

Tests cover:
- CommonValidator class initialization and inheritance
- Rollover parameter validation (boolean)
- Trailing stop parameter validation (None or positive number with range warnings)
- Slippage ticks parameter validation (non-negative with range warnings)
- Warning generation for out-of-range values
- Integration with base validator functions
- Edge cases and boundary values
"""
import pytest

from app.backtesting.validators.common_validator import CommonValidator
from app.backtesting.validators.constants import (
    TRAILING_STOP_COMMON_MAX,
    TRAILING_STOP_COMMON_MIN,
    TRAILING_STOP_MAX,
    TRAILING_STOP_MIN,
)


# ==================== Initialization Tests ====================

class TestCommonValidatorInitialization:
    """Test CommonValidator initialization."""

    def test_inherits_from_validator_base(self):
        """CommonValidator should inherit from Validator base class."""
        validator = CommonValidator()

        assert hasattr(validator, 'warnings')
        assert hasattr(validator, 'reset_warnings')
        assert hasattr(validator, 'validate')
        assert validator.warnings == []

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        validator1 = CommonValidator()
        validator2 = CommonValidator()

        validator1.validate(rollover=True, trailing=0.5, slippage_ticks=1)
        validator2.validate(rollover=False, trailing=None, slippage_ticks=0)

        # Validators should have different warning states
        assert validator1.warnings != validator2.warnings or (
                len(validator1.warnings) == 0 and len(validator2.warnings) == 0
        )


# ==================== Rollover Validation Tests ====================

class TestRolloverValidation:
    """Test rollover parameter validation."""

    def test_accepts_true(self):
        """Rollover=True should be accepted."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=0)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    def test_accepts_false(self):
        """Rollover=False should be accepted."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=None, slippage_ticks=0)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_rollover", [
        1, 0, -1,
        "true", "false", "True", "False",
        1.0, 0.0,
        None, [], {},
    ])
    def test_rejects_non_boolean_rollover(self, invalid_rollover):
        """Non-boolean rollover values should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="rollover must be a boolean"):
            validator.validate(rollover=invalid_rollover, trailing=None, slippage_ticks=0)


# ==================== Trailing Stop Validation Tests ====================

class TestTrailingValidation:
    """Test trailing stop parameter validation."""

    def test_accepts_none(self):
        """Trailing=None should be accepted (no trailing stop)."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=0)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("valid_trailing", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    def test_accepts_positive_numbers(self, valid_trailing):
        """Positive trailing values should be accepted."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=True, trailing=valid_trailing, slippage_ticks=0)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_trailing", [0, 0.0, -1, -1.5, -100])
    def test_rejects_zero_and_negative(self, invalid_trailing):
        """Zero and negative trailing values should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="trailing must be positive"):
            validator.validate(rollover=True, trailing=invalid_trailing, slippage_ticks=0)

    @pytest.mark.parametrize("invalid_type", ["1", "1.5", [], {}, True, False])
    def test_rejects_non_numeric_non_none_types(self, invalid_type):
        """Non-numeric types (except None) should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError):
            validator.validate(rollover=True, trailing=invalid_type, slippage_ticks=0)

    def test_rejects_infinity(self):
        """Infinity should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="trailing must be a finite number"):
            validator.validate(rollover=True, trailing=float('inf'), slippage_ticks=0)

    def test_rejects_nan(self):
        """NaN should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="trailing must be a finite number"):
            validator.validate(rollover=True, trailing=float('nan'), slippage_ticks=0)


# ==================== Slippage Ticks Validation Tests ====================

class TestSlippageTicksValidation:
    """Test slippage_ticks parameter validation."""

    @pytest.mark.parametrize("valid_slippage", [0, 0.0, 1, 2, 3, 5, 10])
    def test_accepts_non_negative_values(self, valid_slippage):
        """Non-negative slippage values should be accepted."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=valid_slippage)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    def test_accepts_zero(self):
        """Zero slippage should be accepted (no slippage simulation)."""
        validator = CommonValidator()
        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=0)

        assert isinstance(warnings, list)
        assert len([w for w in warnings if 'slippage' in w.lower()]) == 0

    @pytest.mark.parametrize("invalid_slippage", [-1, -0.1, -100, -1000.5])
    def test_rejects_negative(self, invalid_slippage):
        """Negative slippage values should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="slippage_ticks must be a non-negative number"):
            validator.validate(rollover=True, trailing=None, slippage_ticks=invalid_slippage)

    @pytest.mark.parametrize("invalid_type", ["1", "0", [], {}, True, False, None])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError):
            validator.validate(rollover=True, trailing=None, slippage_ticks=invalid_type)

    def test_rejects_infinity(self):
        """Infinity should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="slippage_ticks must be a finite number"):
            validator.validate(rollover=True, trailing=None, slippage_ticks=float('inf'))

    def test_rejects_nan(self):
        """NaN should raise ValueError."""
        validator = CommonValidator()

        with pytest.raises(ValueError, match="slippage_ticks must be a finite number"):
            validator.validate(rollover=True, trailing=None, slippage_ticks=float('nan'))


# ==================== Trailing Stop Warning Tests ====================

class TestTrailingStopWarnings:
    """Test warning generation for trailing stop values."""

    def test_warns_when_trailing_too_tight(self):
        """Should warn when trailing stop is below minimum threshold."""
        validator = CommonValidator()
        too_tight = TRAILING_STOP_MIN - 0.5  # 0.5% below minimum

        warnings = validator.validate(rollover=True, trailing=too_tight, slippage_ticks=0)

        assert len(warnings) > 0
        assert any('too tight' in w.lower() for w in warnings)
        assert any('stopped out frequently' in w.lower() for w in warnings)
        assert any(str(TRAILING_STOP_MIN) in w for w in warnings)

    def test_warns_when_trailing_too_wide(self):
        """Should warn when trailing stop is above maximum threshold."""
        validator = CommonValidator()
        too_wide = TRAILING_STOP_MAX + 1.0  # 1% above maximum

        warnings = validator.validate(rollover=True, trailing=too_wide, slippage_ticks=0)

        assert len(warnings) > 0
        assert any('too wide' in w.lower() for w in warnings)
        assert any('give back' in w.lower() or 'profit' in w.lower() for w in warnings)
        assert any(str(TRAILING_STOP_MAX) in w for w in warnings)

    def test_no_warning_in_common_range(self):
        """Should not warn when trailing is in common range."""
        validator = CommonValidator()
        common_value = (TRAILING_STOP_COMMON_MIN + TRAILING_STOP_COMMON_MAX) / 2  # 2.5%

        warnings = validator.validate(rollover=True, trailing=common_value, slippage_ticks=0)

        # Should have no trailing-related warnings
        trailing_warnings = [w for w in warnings if 'trailing' in w.lower()]
        assert len(trailing_warnings) == 0

    def test_no_warning_at_minimum_boundary(self):
        """Should not warn when trailing is exactly at minimum threshold."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MIN,
            slippage_ticks=0
        )

        trailing_warnings = [w for w in warnings if 'trailing' in w.lower()]
        assert len(trailing_warnings) == 0

    def test_no_warning_at_maximum_boundary(self):
        """Should not warn when trailing is exactly at maximum threshold."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MAX,
            slippage_ticks=0
        )

        trailing_warnings = [w for w in warnings if 'trailing' in w.lower()]
        assert len(trailing_warnings) == 0

    def test_warns_just_below_minimum(self):
        """Should warn when trailing is just below minimum threshold."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MIN - 0.01,
            slippage_ticks=0
        )

        assert len(warnings) > 0
        assert any('too tight' in w.lower() for w in warnings)

    def test_warns_just_above_maximum(self):
        """Should warn when trailing is just above maximum threshold."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MAX + 0.01,
            slippage_ticks=0
        )

        assert len(warnings) > 0
        assert any('too wide' in w.lower() for w in warnings)

    def test_no_warning_when_trailing_is_none(self):
        """Should not warn about trailing when it's None."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=0)

        trailing_warnings = [w for w in warnings if 'trailing' in w.lower()]
        assert len(trailing_warnings) == 0


# ==================== Slippage Warning Tests ====================

class TestSlippageWarnings:
    """Test warning generation for slippage values."""

    def test_warns_when_slippage_very_high(self):
        """Should warn when slippage is above 10 ticks."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=11)

        assert len(warnings) > 0
        assert any('very high' in w.lower() for w in warnings)
        assert any('11' in w or '11 ticks' in w for w in warnings)

    def test_warns_when_slippage_high(self):
        """Should warn when slippage is 6-10 ticks."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=6)

        assert len(warnings) > 0
        assert any('high' in w.lower() for w in warnings)
        assert any('6' in w for w in warnings)

    def test_no_warning_for_typical_slippage(self):
        """Should not warn for typical slippage values (0-5 ticks)."""
        validator = CommonValidator()

        for slippage in [0, 1, 2, 3, 4, 5]:
            warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=slippage)
            slippage_warnings = [w for w in warnings if 'slippage' in w.lower()]
            assert len(slippage_warnings) == 0, f"Should not warn for slippage={slippage}"

    def test_no_warning_at_boundary_5_ticks(self):
        """Should not warn when slippage is exactly 5 ticks."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=5)

        slippage_warnings = [w for w in warnings if 'slippage' in w.lower()]
        assert len(slippage_warnings) == 0

    def test_warns_at_boundary_6_ticks(self):
        """Should warn when slippage is exactly 6 ticks."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=6)

        assert len(warnings) > 0
        assert any('high' in w.lower() for w in warnings)

    def test_warns_at_boundary_11_ticks(self):
        """Should warn 'very high' when slippage is exactly 11 ticks."""
        validator = CommonValidator()

        warnings = validator.validate(rollover=True, trailing=None, slippage_ticks=11)

        assert len(warnings) > 0
        assert any('very high' in w.lower() for w in warnings)


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for both trailing and slippage if both are out of range."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MAX + 1.0,  # Too wide
            slippage_ticks=11  # Very high
        )

        assert len(warnings) >= 2
        assert any('trailing' in w.lower() for w in warnings)
        assert any('slippage' in w.lower() for w in warnings)

    def test_warnings_are_independent(self):
        """Warnings for different parameters should be independent."""
        validator = CommonValidator()

        # Only trailing warning
        warnings1 = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MIN - 0.5,
            slippage_ticks=2
        )

        # Only slippage warning
        warnings2 = validator.validate(
            rollover=True,
            trailing=2.5,
            slippage_ticks=11
        )

        # Both warnings
        warnings3 = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MIN - 0.5,
            slippage_ticks=11
        )

        assert len(warnings1) == 1
        assert len(warnings2) == 1
        assert len(warnings3) == 2


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        validator = CommonValidator()

        # First validation with warnings
        warnings1 = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MAX + 1.0,
            slippage_ticks=11
        )
        assert len(warnings1) > 0

        # Second validation without warnings
        warnings2 = validator.validate(
            rollover=True,
            trailing=2.5,
            slippage_ticks=2
        )
        assert len(warnings2) == 0

        # Third validation with different warnings
        warnings3 = validator.validate(
            rollover=True,
            trailing=TRAILING_STOP_MIN - 0.5,
            slippage_ticks=0
        )
        assert len(warnings3) > 0
        assert warnings3 != warnings1

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        validator = CommonValidator()

        warnings1 = validator.validate(rollover=True, trailing=10.0, slippage_ticks=0)
        warnings2 = validator.validate(rollover=True, trailing=10.0, slippage_ticks=0)

        # Should have same content but be independent lists
        assert warnings1 == warnings2
        warnings1.append("extra")
        assert len(warnings1) != len(warnings2)


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=None,
            slippage_ticks=0,
            extra_param1="ignored",
            extra_param2=123,
            extra_param3=[1, 2, 3]
        )

        assert isinstance(warnings, list)
        # Should not raise any error

    def test_validate_works_with_named_parameters(self):
        """Validate should work with all named parameters."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=False,
            trailing=2.5,
            slippage_ticks=2
        )

        assert isinstance(warnings, list)


# ==================== Integration Tests ====================

class TestCommonValidatorIntegration:
    """Test CommonValidator integration with various parameter combinations."""

    def test_realistic_conservative_parameters(self):
        """Test with realistic conservative trading parameters."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=2.0,  # Conservative 2% trailing stop
            slippage_ticks=2  # Moderate slippage
        )

        assert len(warnings) == 0

    def test_realistic_aggressive_parameters(self):
        """Test with realistic aggressive trading parameters."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=3.5,  # Wider trailing stop
            slippage_ticks=1  # Low slippage
        )

        assert len(warnings) == 0

    def test_no_trailing_stop_strategy(self):
        """Test parameters for strategy without trailing stop."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert len(warnings) == 0

    def test_zero_slippage_backtest(self):
        """Test parameters for idealized backtest with zero slippage."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=2.5,
            slippage_ticks=0
        )

        assert len(warnings) == 0

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=0.1,  # Too tight
            slippage_ticks=15  # Very high
        )

        assert len(warnings) == 2
        assert any('trailing' in w.lower() for w in warnings)
        assert any('slippage' in w.lower() for w in warnings)


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_trailing_values(self):
        """Test with very small trailing stop values."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=0.01,
            slippage_ticks=1
        )

        assert len(warnings) > 0
        assert any('too tight' in w.lower() for w in warnings)

    def test_very_large_trailing_values(self):
        """Test with very large trailing stop values."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=100.0,
            slippage_ticks=1
        )

        assert len(warnings) > 0
        assert any('too wide' in w.lower() for w in warnings)

    def test_very_large_slippage_values(self):
        """Test with unrealistically large slippage values."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=None,
            slippage_ticks=1000
        )

        assert len(warnings) > 0
        assert any('very high' in w.lower() for w in warnings)

    def test_fractional_slippage_values(self):
        """Test with fractional slippage values."""
        validator = CommonValidator()

        warnings = validator.validate(
            rollover=True,
            trailing=None,
            slippage_ticks=2.5
        )

        # Should accept fractional slippage
        slippage_warnings = [w for w in warnings if 'slippage' in w.lower()]
        assert len(slippage_warnings) == 0
