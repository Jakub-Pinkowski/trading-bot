"""
Tests for bollinger_validator module.

Tests cover:
- BollingerValidator class initialization and inheritance
- Period validation (positive integers, range warnings)
- Standard deviation validation (positive numbers, range warnings)
- Warning generation for out-of-range values
- Integration with base validator functions
- Edge cases and boundary values
- Traditional Bollinger Bands settings (20/2.0)
"""
import pytest

from app.backtesting.validators.bollinger_validator import BollingerValidator
from app.backtesting.validators.constants import (
    BB_PERIOD_MAX,
    BB_PERIOD_MIN,
    BB_PERIOD_STANDARD,
    BB_STD_MAX,
    BB_STD_MIN,
    BB_STD_STANDARD,
)


# ==================== Initialization Tests ====================

class TestBollingerValidatorInitialization:
    """Test BollingerValidator initialization."""

    def test_inherits_from_validator_base(self):
        """BollingerValidator should inherit from Validator base class."""
        validator = BollingerValidator()

        assert hasattr(validator, 'warnings')
        assert hasattr(validator, 'reset_warnings')
        assert hasattr(validator, 'validate')
        assert validator.warnings == []

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        validator1 = BollingerValidator()
        validator2 = BollingerValidator()

        validator1.validate(period=20, number_of_standard_deviations=2.0)
        validator2.validate(period=15, number_of_standard_deviations=1.5)

        # Validators should have different warning states
        assert validator1.warnings != validator2.warnings or (
                len(validator1.warnings) == 0 and len(validator2.warnings) == 0
        )


# ==================== Period Validation Tests ====================

class TestPeriodValidation:
    """Test period parameter validation."""

    @pytest.mark.parametrize("valid_period", [15, 18, 20, 22, 25])
    def test_accepts_valid_positive_integers(self, valid_period):
        """Valid periods should be accepted."""
        validator = BollingerValidator()
        warnings = validator.validate(period=valid_period, number_of_standard_deviations=2.0)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_period", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_period):
        """Zero and negative periods should raise ValueError."""
        validator = BollingerValidator()

        with pytest.raises(ValueError, match="period must be a positive integer"):
            validator.validate(period=invalid_period, number_of_standard_deviations=2.0)

    @pytest.mark.parametrize("invalid_type", [20.5, "20", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for period."""
        validator = BollingerValidator()

        with pytest.raises(ValueError):
            validator.validate(period=invalid_type, number_of_standard_deviations=2.0)

    def test_rejects_float_even_if_whole_number(self):
        """Float values should be rejected even if they represent whole numbers."""
        validator = BollingerValidator()

        with pytest.raises(ValueError, match="period must be a positive integer"):
            validator.validate(period=20.0, number_of_standard_deviations=2.0)


# ==================== Standard Deviation Validation Tests ====================

class TestStandardDeviationValidation:
    """Test number_of_standard_deviations parameter validation."""

    @pytest.mark.parametrize("valid_std", [1.5, 1.8, 2.0, 2.2, 2.5])
    def test_accepts_valid_positive_numbers(self, valid_std):
        """Valid standard deviation values should be accepted."""
        validator = BollingerValidator()
        warnings = validator.validate(period=20, number_of_standard_deviations=valid_std)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("valid_std_int", [1, 2, 3])
    def test_accepts_integers_as_numbers(self, valid_std_int):
        """Integer standard deviation values should be accepted as numbers."""
        validator = BollingerValidator()
        warnings = validator.validate(period=20, number_of_standard_deviations=valid_std_int)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_std", [0, -1.0, -2.5])
    def test_rejects_zero_and_negative(self, invalid_std):
        """Zero and negative standard deviations should raise ValueError."""
        validator = BollingerValidator()

        with pytest.raises(ValueError, match="number of standard deviations must be positive"):
            validator.validate(period=20, number_of_standard_deviations=invalid_std)

    @pytest.mark.parametrize("invalid_type", ["2.0", [], {}, None, True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError for standard deviation."""
        validator = BollingerValidator()

        with pytest.raises(ValueError):
            validator.validate(period=20, number_of_standard_deviations=invalid_type)

    def test_accepts_float_values(self):
        """Float values should be accepted for standard deviation."""
        validator = BollingerValidator()

        warnings = validator.validate(period=20, number_of_standard_deviations=2.5)

        assert isinstance(warnings, list)
        # Should not raise ValueError


# ==================== Period Range Warning Tests ====================

class TestPeriodRangeWarnings:
    """Test warning generation for period values."""

    def test_warns_when_period_too_small(self):
        """Should warn when period is below minimum threshold."""
        validator = BollingerValidator()
        too_small = BB_PERIOD_MIN - 1

        warnings = validator.validate(period=too_small, number_of_standard_deviations=2.0)

        assert len(warnings) > 0
        assert any('period' in w.lower() and 'too short' in w.lower() for w in warnings)
        assert any('excessive signal' in w.lower() for w in warnings)
        assert any(str(BB_PERIOD_MIN) in w for w in warnings)

    def test_warns_when_period_too_large(self):
        """Should warn when period is above maximum threshold."""
        validator = BollingerValidator()
        too_large = BB_PERIOD_MAX + 1

        warnings = validator.validate(period=too_large, number_of_standard_deviations=2.0)

        assert len(warnings) > 0
        assert any('period' in w.lower() and 'too long' in w.lower() for w in warnings)
        assert any('miss' in w.lower() for w in warnings)
        assert any(str(BB_PERIOD_MAX) in w for w in warnings)

    def test_no_warning_for_standard_period(self):
        """Should not warn for standard period value (20)."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_STANDARD,
            number_of_standard_deviations=BB_STD_STANDARD
        )

        period_warnings = [w for w in warnings if 'period' in w.lower() and 'too' in w.lower()]
        assert len(period_warnings) == 0

    def test_no_warning_at_minimum_boundary(self):
        """Should not warn when period is exactly at minimum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MIN,
            number_of_standard_deviations=2.0
        )

        period_warnings = [w for w in warnings if 'period' in w.lower() and 'too short' in w.lower()]
        assert len(period_warnings) == 0

    def test_no_warning_at_maximum_boundary(self):
        """Should not warn when period is exactly at maximum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MAX,
            number_of_standard_deviations=2.0
        )

        period_warnings = [w for w in warnings if 'period' in w.lower() and 'too long' in w.lower()]
        assert len(period_warnings) == 0

    def test_warns_just_below_minimum(self):
        """Should warn when period is just below minimum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MIN - 1,
            number_of_standard_deviations=2.0
        )

        assert len(warnings) > 0
        assert any('too short' in w.lower() for w in warnings)

    def test_warns_just_above_maximum(self):
        """Should warn when period is just above maximum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MAX + 1,
            number_of_standard_deviations=2.0
        )

        assert len(warnings) > 0
        assert any('too long' in w.lower() for w in warnings)


# ==================== Standard Deviation Range Warning Tests ====================

class TestStandardDeviationRangeWarnings:
    """Test warning generation for standard deviation values."""

    def test_warns_when_std_too_small(self):
        """Should warn when standard deviation is below minimum threshold."""
        validator = BollingerValidator()
        too_small = BB_STD_MIN - 0.1

        warnings = validator.validate(period=20, number_of_standard_deviations=too_small)

        assert len(warnings) > 0
        assert any('standard deviation' in w.lower() and 'too narrow' in w.lower() for w in warnings)
        assert any('excessive signal' in w.lower() for w in warnings)

    def test_warns_when_std_too_large(self):
        """Should warn when standard deviation is above maximum threshold."""
        validator = BollingerValidator()
        too_large = BB_STD_MAX + 0.5

        warnings = validator.validate(period=20, number_of_standard_deviations=too_large)

        assert len(warnings) > 0
        assert any('standard deviation' in w.lower() and 'too wide' in w.lower() for w in warnings)
        assert any('miss opportunit' in w.lower() for w in warnings)

    def test_no_warning_for_standard_std(self):
        """Should not warn for standard deviation value (2.0)."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_STANDARD,
            number_of_standard_deviations=BB_STD_STANDARD
        )

        std_warnings = [w for w in warnings if 'standard deviation' in w.lower() and 'too' in w.lower()]
        assert len(std_warnings) == 0

    def test_no_warning_at_minimum_boundary(self):
        """Should not warn when standard deviation is exactly at minimum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MIN
        )

        std_warnings = [w for w in warnings if 'standard deviation' in w.lower() and 'too narrow' in w.lower()]
        assert len(std_warnings) == 0

    def test_no_warning_at_maximum_boundary(self):
        """Should not warn when standard deviation is exactly at maximum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MAX
        )

        std_warnings = [w for w in warnings if 'standard deviation' in w.lower() and 'too wide' in w.lower()]
        assert len(std_warnings) == 0

    def test_warns_just_below_minimum(self):
        """Should warn when standard deviation is just below minimum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MIN - 0.1
        )

        assert len(warnings) > 0
        assert any('too narrow' in w.lower() for w in warnings)

    def test_warns_just_above_maximum(self):
        """Should warn when standard deviation is just above maximum threshold."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MAX + 0.1
        )

        assert len(warnings) > 0
        assert any('too wide' in w.lower() for w in warnings)


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for all out-of-range parameters simultaneously."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MIN - 1,  # Too short
            number_of_standard_deviations=BB_STD_MAX + 0.5  # Too wide
        )

        assert len(warnings) >= 2
        assert any('period' in w.lower() for w in warnings)
        assert any('standard deviation' in w.lower() for w in warnings)

    def test_warnings_are_independent(self):
        """Warnings for different parameters should be independent."""
        validator = BollingerValidator()

        # Only period warning
        warnings1 = validator.validate(
            period=BB_PERIOD_MIN - 1,
            number_of_standard_deviations=2.0
        )

        # Only standard deviation warning
        warnings2 = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MIN - 0.1
        )

        assert any('period' in w.lower() for w in warnings1)
        assert any('standard deviation' in w.lower() for w in warnings2)


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        validator = BollingerValidator()

        # First validation with warnings
        warnings1 = validator.validate(
            period=BB_PERIOD_MIN - 1,
            number_of_standard_deviations=2.0
        )
        assert len(warnings1) > 0

        # Second validation without warnings
        warnings2 = validator.validate(
            period=20,
            number_of_standard_deviations=2.0
        )
        assert len(warnings2) == 0

        # Third validation with different warnings
        warnings3 = validator.validate(
            period=20,
            number_of_standard_deviations=BB_STD_MAX + 0.5
        )
        assert len(warnings3) > 0
        assert warnings3 != warnings1

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        validator = BollingerValidator()

        warnings1 = validator.validate(period=20, number_of_standard_deviations=2.0)
        warnings2 = validator.validate(period=20, number_of_standard_deviations=2.0)

        # Should have same content but be independent lists
        assert warnings1 == warnings2
        warnings1.append("extra")
        assert len(warnings1) != len(warnings2)


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=2.0,
            extra_param1="ignored",
            extra_param2=123,
            rollover=True,
            trailing=2.5,
            slippage_ticks=1
        )

        assert isinstance(warnings, list)
        # Should not raise any error

    def test_validate_works_with_named_parameters(self):
        """Validate should work with all named parameters."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=2.0
        )

        assert isinstance(warnings, list)


# ==================== Integration Tests ====================

class TestBollingerValidatorIntegration:
    """Test BollingerValidator integration with various parameter combinations."""

    def test_standard_bollinger_bands(self):
        """Test standard Bollinger Bands settings (20/2.0)."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=2.0
        )

        assert len(warnings) == 0

    def test_narrow_bands(self):
        """Test narrow Bollinger Bands settings."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=1.5
        )

        assert len(warnings) == 0

    def test_wide_bands(self):
        """Test wide Bollinger Bands settings."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=20,
            number_of_standard_deviations=2.5
        )

        assert len(warnings) == 0

    def test_fast_bollinger_bands(self):
        """Test fast Bollinger Bands settings."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=15,
            number_of_standard_deviations=2.0
        )

        assert len(warnings) == 0

    def test_slow_bollinger_bands(self):
        """Test slow Bollinger Bands settings."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=25,
            number_of_standard_deviations=2.0
        )

        assert len(warnings) == 0

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=BB_PERIOD_MIN - 1,
            number_of_standard_deviations=BB_STD_MAX + 0.5
        )

        assert len(warnings) >= 2
        assert any('period' in w.lower() for w in warnings)
        assert any('standard deviation' in w.lower() for w in warnings)


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_parameters(self):
        """Test with minimal valid parameter values."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=1,
            number_of_standard_deviations=0.1
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_very_large_parameters(self):
        """Test with very large parameter values."""
        validator = BollingerValidator()

        warnings = validator.validate(
            period=100,
            number_of_standard_deviations=5.0
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_boundary_values(self):
        """Test at exact boundary values."""
        validator = BollingerValidator()

        # Test minimum boundaries
        warnings1 = validator.validate(
            period=BB_PERIOD_MIN,
            number_of_standard_deviations=BB_STD_MIN
        )

        # Test maximum boundaries
        warnings2 = validator.validate(
            period=BB_PERIOD_MAX,
            number_of_standard_deviations=BB_STD_MAX
        )

        # Boundary values should not generate range warnings
        period_warnings1 = [
            w for w in warnings1
            if 'period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        std_warnings1 = [
            w for w in warnings1
            if 'standard deviation' in w.lower() and ('too narrow' in w.lower() or 'too wide' in w.lower())
        ]
        assert len(period_warnings1) == 0
        assert len(std_warnings1) == 0

        period_warnings2 = [
            w for w in warnings2
            if 'period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        std_warnings2 = [
            w for w in warnings2
            if 'standard deviation' in w.lower() and ('too narrow' in w.lower() or 'too wide' in w.lower())
        ]
        assert len(period_warnings2) == 0
        assert len(std_warnings2) == 0

    def test_common_variations(self):
        """Test common Bollinger Bands variations."""
        validator = BollingerValidator()

        # Common variations that should not generate warnings
        common_settings = [
            (20, 2.0),  # Standard
            (20, 1.5),  # Narrow
            (20, 2.5),  # Wide
            (15, 2.0),  # Fast
            (25, 2.0),  # Slow
        ]

        for period, std in common_settings:
            warnings = validator.validate(
                period=period,
                number_of_standard_deviations=std
            )
            assert len(warnings) == 0, f"Common settings {period}/{std} should not generate warnings"

    def test_extreme_combinations(self):
        """Test extreme parameter combinations."""
        validator = BollingerValidator()

        # Very fast with very narrow bands
        warnings1 = validator.validate(
            period=10,
            number_of_standard_deviations=1.0
        )
        assert len(warnings1) > 0

        # Very slow with very wide bands
        warnings2 = validator.validate(
            period=30,
            number_of_standard_deviations=3.0
        )
        assert len(warnings2) > 0
