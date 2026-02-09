"""
Tests for rsi_validator module.

Tests cover:
- RSIValidator class initialization and inheritance
- RSI period validation (positive integers, range warnings)
- Lower threshold validation (type and range 0-100)
- Upper threshold validation (type and range 0-100)
- Threshold relationship validation (lower < upper)
- Threshold gap validation (adequate separation)
- Warning generation for out-of-range values
- Integration with base validator functions
- Edge cases and boundary values
- Standard RSI settings (14/30/70)
"""
import pytest

from app.backtesting.validators.constants import (
    RSI_GAP_MAX,
    RSI_GAP_MIN,
    RSI_LOWER_MAX_CONSERVATIVE,
    RSI_LOWER_MIN_AGGRESSIVE,
    RSI_LOWER_STANDARD,
    RSI_PERIOD_MAX_RECOMMENDED,
    RSI_PERIOD_MIN_RECOMMENDED,
    RSI_PERIOD_STANDARD,
    RSI_UPPER_MAX_CONSERVATIVE,
    RSI_UPPER_MIN_AGGRESSIVE,
    RSI_UPPER_STANDARD,
)
from app.backtesting.validators.rsi_validator import RSIValidator


# ==================== Initialization Tests ====================

class TestRSIValidatorInitialization:
    """Test RSIValidator initialization."""

    def test_inherits_from_validator_base(self):
        """RSIValidator should inherit from Validator base class."""
        validator = RSIValidator()

        assert hasattr(validator, 'warnings')
        assert hasattr(validator, 'reset_warnings')
        assert hasattr(validator, 'validate')
        assert validator.warnings == []

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        validator1 = RSIValidator()
        validator2 = RSIValidator()

        validator1.validate(rsi_period=14, lower_threshold=30, upper_threshold=70)
        validator2.validate(rsi_period=10, lower_threshold=20, upper_threshold=80)

        # Validators should have different warning states
        assert validator1.warnings != validator2.warnings or (
                len(validator1.warnings) == 0 and len(validator2.warnings) == 0
        )


# ==================== RSI Period Validation Tests ====================

class TestRSIPeriodValidation:
    """Test rsi_period parameter validation."""

    @pytest.mark.parametrize("valid_period", [10, 12, 14, 20, 30])
    def test_accepts_valid_positive_integers(self, valid_period):
        """Valid RSI periods should be accepted."""
        validator = RSIValidator()
        warnings = validator.validate(
            rsi_period=valid_period,
            lower_threshold=30,
            upper_threshold=70
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_period", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_period):
        """Zero and negative RSI periods should raise ValueError."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="rsi period must be a positive integer"):
            validator.validate(rsi_period=invalid_period, lower_threshold=30, upper_threshold=70)

    @pytest.mark.parametrize("invalid_type", [14.5, "14", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for RSI period."""
        validator = RSIValidator()

        with pytest.raises(ValueError):
            validator.validate(rsi_period=invalid_type, lower_threshold=30, upper_threshold=70)


# ==================== Lower Threshold Validation Tests ====================

class TestLowerThresholdValidation:
    """Test lower_threshold parameter validation."""

    @pytest.mark.parametrize("valid_lower", [20, 25, 30, 35, 40])
    def test_accepts_valid_numbers_in_range(self, valid_lower):
        """Valid lower threshold values should be accepted."""
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=valid_lower, upper_threshold=70)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("valid_lower_float", [20.5, 25.5, 30.5])
    def test_accepts_float_values_in_range(self, valid_lower_float):
        """Float values within range should be accepted for lower threshold."""
        validator = RSIValidator()
        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=valid_lower_float,
            upper_threshold=70
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("out_of_range", [-1, -10, 101, 150])
    def test_rejects_out_of_range_values(self, out_of_range):
        """Values outside 0-100 range should raise ValueError."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="lower threshold must be between 0 and 100"):
            validator.validate(rsi_period=14, lower_threshold=out_of_range, upper_threshold=70)

    @pytest.mark.parametrize("invalid_type", ["30", [], {}, None, True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError for lower threshold."""
        validator = RSIValidator()

        with pytest.raises(ValueError):
            validator.validate(rsi_period=14, lower_threshold=invalid_type, upper_threshold=70)


# ==================== Upper Threshold Validation Tests ====================

class TestUpperThresholdValidation:
    """Test upper_threshold parameter validation."""

    @pytest.mark.parametrize("valid_upper", [60, 65, 70, 75, 80])
    def test_accepts_valid_numbers_in_range(self, valid_upper):
        """Valid upper threshold values should be accepted."""
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=valid_upper)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("valid_upper_float", [65.5, 70.5, 75.5])
    def test_accepts_float_values_in_range(self, valid_upper_float):
        """Float values within range should be accepted for upper threshold."""
        validator = RSIValidator()
        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=valid_upper_float
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("out_of_range", [-1, -10, 101, 150])
    def test_rejects_out_of_range_values(self, out_of_range):
        """Values outside 0-100 range should raise ValueError."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="upper threshold must be between 0 and 100"):
            validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=out_of_range)

    @pytest.mark.parametrize("invalid_type", ["70", [], {}, None, True, False])
    def test_rejects_non_numeric_types(self, invalid_type):
        """Non-numeric types should raise ValueError for upper threshold."""
        validator = RSIValidator()

        with pytest.raises(ValueError):
            validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=invalid_type)


# ==================== Threshold Relationship Validation Tests ====================

class TestThresholdRelationship:
    """Test validation of relationship between lower and upper thresholds."""

    def test_rejects_when_lower_equals_upper(self):
        """Should reject when lower threshold equals upper threshold."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="Lower threshold .* must be less than upper threshold"):
            validator.validate(rsi_period=14, lower_threshold=50, upper_threshold=50)

    def test_rejects_when_lower_greater_than_upper(self):
        """Should reject when lower threshold is greater than upper threshold."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="Lower threshold .* must be less than upper threshold"):
            validator.validate(rsi_period=14, lower_threshold=70, upper_threshold=30)

    def test_accepts_when_lower_less_than_upper(self):
        """Should accept when lower threshold is less than upper threshold."""
        validator = RSIValidator()

        warnings = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=70)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    def test_error_message_includes_both_values(self):
        """Error message should include both threshold values for clarity."""
        validator = RSIValidator()

        with pytest.raises(ValueError, match="70.*30"):
            validator.validate(rsi_period=14, lower_threshold=70, upper_threshold=30)


# ==================== RSI Period Range Warning Tests ====================

class TestRSIPeriodRangeWarnings:
    """Test warning generation for RSI period values."""

    def test_warns_when_period_too_small(self):
        """Should warn when RSI period is below minimum threshold."""
        validator = RSIValidator()
        too_small = RSI_PERIOD_MIN_RECOMMENDED - 1

        warnings = validator.validate(
            rsi_period=too_small,
            lower_threshold=30,
            upper_threshold=70
        )

        assert len(warnings) > 0
        assert any('period' in w.lower() and 'too short' in w.lower() for w in warnings)
        assert any('noise' in w.lower() for w in warnings)

    def test_warns_when_period_too_large(self):
        """Should warn when RSI period is above maximum threshold."""
        validator = RSIValidator()
        too_large = RSI_PERIOD_MAX_RECOMMENDED + 1

        warnings = validator.validate(
            rsi_period=too_large,
            lower_threshold=30,
            upper_threshold=70
        )

        assert len(warnings) > 0
        assert any('period' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_period(self):
        """Should not warn for standard RSI period value (14)."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=RSI_PERIOD_STANDARD,
            lower_threshold=RSI_LOWER_STANDARD,
            upper_threshold=RSI_UPPER_STANDARD
        )

        period_warnings = [w for w in warnings if 'period' in w.lower() and 'too' in w.lower()]
        assert len(period_warnings) == 0


# ==================== Lower Threshold Range Warning Tests ====================

class TestLowerThresholdRangeWarnings:
    """Test warning generation for lower threshold values."""

    def test_warns_when_lower_too_aggressive(self):
        """Should warn when lower threshold is below aggressive minimum."""
        validator = RSIValidator()
        too_aggressive = RSI_LOWER_MIN_AGGRESSIVE - 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=too_aggressive,
            upper_threshold=70
        )

        assert len(warnings) > 0
        assert any('lower threshold' in w.lower() and 'too aggressive' in w.lower() for w in warnings)

    def test_warns_when_lower_too_conservative(self):
        """Should warn when lower threshold is above conservative maximum."""
        validator = RSIValidator()
        too_conservative = RSI_LOWER_MAX_CONSERVATIVE + 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=too_conservative,
            upper_threshold=70
        )

        assert len(warnings) > 0
        assert any('lower threshold' in w.lower() and 'too conservative' in w.lower() for w in warnings)

    def test_no_warning_for_standard_lower(self):
        """Should not warn for standard lower threshold value (30)."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=RSI_PERIOD_STANDARD,
            lower_threshold=RSI_LOWER_STANDARD,
            upper_threshold=RSI_UPPER_STANDARD
        )

        lower_warnings = [w for w in warnings if 'lower threshold' in w.lower() and 'too' in w.lower()]
        assert len(lower_warnings) == 0


# ==================== Upper Threshold Range Warning Tests ====================

class TestUpperThresholdRangeWarnings:
    """Test warning generation for upper threshold values."""

    def test_warns_when_upper_too_aggressive(self):
        """Should warn when upper threshold is below aggressive minimum."""
        validator = RSIValidator()
        too_aggressive = RSI_UPPER_MIN_AGGRESSIVE - 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=too_aggressive
        )

        assert len(warnings) > 0
        assert any('upper threshold' in w.lower() and 'too aggressive' in w.lower() for w in warnings)

    def test_warns_when_upper_too_conservative(self):
        """Should warn when upper threshold is above conservative maximum."""
        validator = RSIValidator()
        too_conservative = RSI_UPPER_MAX_CONSERVATIVE + 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=too_conservative
        )

        assert len(warnings) > 0
        assert any('upper threshold' in w.lower() and 'too conservative' in w.lower() for w in warnings)

    def test_no_warning_for_standard_upper(self):
        """Should not warn for standard upper threshold value (70)."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=RSI_PERIOD_STANDARD,
            lower_threshold=RSI_LOWER_STANDARD,
            upper_threshold=RSI_UPPER_STANDARD
        )

        upper_warnings = [w for w in warnings if 'upper threshold' in w.lower() and 'too' in w.lower()]
        assert len(upper_warnings) == 0


# ==================== Threshold Gap Warning Tests ====================

class TestThresholdGapWarnings:
    """Test warning generation for threshold gap."""

    def test_warns_when_gap_too_narrow(self):
        """Should warn when gap between thresholds is too narrow."""
        validator = RSIValidator()
        # Create gap below minimum
        lower = 40
        upper = lower + RSI_GAP_MIN - 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=lower,
            upper_threshold=upper
        )

        assert len(warnings) > 0
        assert any('gap' in w.lower() and 'too narrow' in w.lower() for w in warnings)

    def test_warns_when_gap_too_wide(self):
        """Should warn when gap between thresholds is too wide."""
        validator = RSIValidator()
        # Create gap above maximum
        lower = 10
        upper = lower + RSI_GAP_MAX + 1

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=lower,
            upper_threshold=upper
        )

        assert len(warnings) > 0
        assert any('gap' in w.lower() and 'too wide' in w.lower() for w in warnings)

    def test_no_warning_for_optimal_gap(self):
        """Should not warn for optimal threshold gaps."""
        validator = RSIValidator()

        # Test standard gap (40 points)
        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70
        )

        gap_warnings = [w for w in warnings if 'gap' in w.lower()]
        assert len(gap_warnings) == 0

    def test_warning_includes_gap_value(self):
        """Warning should include the actual gap value for clarity."""
        validator = RSIValidator()
        lower = 45
        upper = 50  # Gap of 5

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=lower,
            upper_threshold=upper
        )

        assert len(warnings) > 0
        # Should include gap value in warning
        assert any('5' in w for w in warnings)


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for all out-of-range parameters simultaneously."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=RSI_PERIOD_MIN_RECOMMENDED - 1,  # Too short
            lower_threshold=RSI_LOWER_MIN_AGGRESSIVE - 1,  # Too aggressive
            upper_threshold=RSI_UPPER_MAX_CONSERVATIVE + 1  # Too conservative
        )

        assert len(warnings) >= 3
        assert any('period' in w.lower() for w in warnings)
        assert any('lower' in w.lower() for w in warnings)
        assert any('upper' in w.lower() for w in warnings)

    def test_warnings_are_independent(self):
        """Warnings for different parameters should be independent."""
        validator = RSIValidator()

        # Only period warning
        warnings1 = validator.validate(
            rsi_period=RSI_PERIOD_MIN_RECOMMENDED - 1,
            lower_threshold=30,
            upper_threshold=70
        )

        # Only lower threshold warning
        warnings2 = validator.validate(
            rsi_period=14,
            lower_threshold=RSI_LOWER_MIN_AGGRESSIVE - 1,
            upper_threshold=70
        )

        # Only gap warning
        warnings3 = validator.validate(
            rsi_period=14,
            lower_threshold=45,
            upper_threshold=50
        )

        assert any('period' in w.lower() for w in warnings1)
        assert any('lower' in w.lower() for w in warnings2)
        assert any('gap' in w.lower() for w in warnings3)


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        validator = RSIValidator()

        # First validation with warnings
        warnings1 = validator.validate(
            rsi_period=RSI_PERIOD_MIN_RECOMMENDED - 1,
            lower_threshold=30,
            upper_threshold=70
        )
        assert len(warnings1) > 0

        # Second validation without warnings
        warnings2 = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70
        )
        assert len(warnings2) == 0

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        validator = RSIValidator()

        warnings1 = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=70)
        warnings2 = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=70)

        # Should have same content but be independent lists
        assert warnings1 == warnings2
        warnings1.append("extra")
        assert len(warnings1) != len(warnings2)


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            extra_param1="ignored",
            extra_param2=123,
            rollover=True,
            trailing=2.5,
            slippage_ticks=1
        )

        assert isinstance(warnings, list)
        # Should not raise any error


# ==================== Integration Tests ====================

class TestRSIValidatorIntegration:
    """Test RSIValidator integration with various parameter combinations."""

    def test_standard_rsi_settings(self):
        """Test standard RSI settings (14/30/70)."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70
        )

        assert len(warnings) == 0

    def test_aggressive_rsi_settings(self):
        """Test aggressive RSI settings."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=10,
            lower_threshold=25,
            upper_threshold=75
        )

        assert len(warnings) == 0

    def test_conservative_rsi_settings(self):
        """Test conservative RSI settings."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=20,
            lower_threshold=40,
            upper_threshold=60
        )

        assert len(warnings) == 0

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=RSI_PERIOD_MIN_RECOMMENDED - 1,
            lower_threshold=RSI_LOWER_MIN_AGGRESSIVE - 1,
            upper_threshold=RSI_UPPER_MAX_CONSERVATIVE + 1
        )

        assert len(warnings) >= 2  # At least period and threshold warnings


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_parameters(self):
        """Test with minimal valid parameter values."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=1,
            lower_threshold=0,
            upper_threshold=100
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_boundary_values(self):
        """Test at exact boundary values."""
        validator = RSIValidator()

        # Test minimum boundaries
        warnings1 = validator.validate(
            rsi_period=RSI_PERIOD_MIN_RECOMMENDED,
            lower_threshold=RSI_LOWER_MIN_AGGRESSIVE,
            upper_threshold=RSI_UPPER_MIN_AGGRESSIVE
        )

        # Test maximum boundaries
        warnings2 = validator.validate(
            rsi_period=RSI_PERIOD_MAX_RECOMMENDED,
            lower_threshold=RSI_LOWER_MAX_CONSERVATIVE,
            upper_threshold=RSI_UPPER_MAX_CONSERVATIVE
        )

        # Boundary values should not generate range warnings for period
        period_warnings1 = [
            w for w in warnings1
            if 'period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(period_warnings1) == 0

        period_warnings2 = [
            w for w in warnings2
            if 'period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(period_warnings2) == 0

    def test_common_variations(self):
        """Test common RSI variations."""
        validator = RSIValidator()

        # Common variations that should not generate warnings
        common_settings = [
            (14, 30, 70),  # Standard
            (10, 25, 75),  # Aggressive
            (20, 40, 60),  # Conservative
            (14, 25, 75),  # Wide bands
        ]

        for period, lower, upper in common_settings:
            warnings = validator.validate(
                rsi_period=period,
                lower_threshold=lower,
                upper_threshold=upper
            )
            assert len(warnings) == 0, f"Common settings {period}/{lower}/{upper} should not generate warnings"

    def test_extreme_threshold_values(self):
        """Test extreme but valid threshold values."""
        validator = RSIValidator()

        # Very narrow range
        warnings1 = validator.validate(
            rsi_period=14,
            lower_threshold=49,
            upper_threshold=51
        )
        assert len(warnings1) > 0

        # Very wide range
        warnings2 = validator.validate(
            rsi_period=14,
            lower_threshold=5,
            upper_threshold=95
        )
        assert len(warnings2) > 0

    def test_float_threshold_values(self):
        """Test that float threshold values work correctly."""
        validator = RSIValidator()

        warnings = validator.validate(
            rsi_period=14,
            lower_threshold=30.5,
            upper_threshold=69.5
        )

        assert isinstance(warnings, list)
        # Should not raise errors
