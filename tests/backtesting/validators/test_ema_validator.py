"""
Tests for ema_validator module.

Tests cover:
- EMAValidator class initialization and inheritance
- Short EMA period validation (positive integers, range warnings)
- Long EMA period validation (positive integers, range warnings)
- Period relationship validation (short < long)
- EMA ratio validation (separation between short and long periods)
- Warning generation for out-of-range values
- Integration with base validator functions
- Edge cases and boundary values
"""
import pytest

from app.backtesting.validators.constants import (
    EMA_LONG_COMMON_MAX,
    EMA_LONG_COMMON_MIN,
    EMA_LONG_MAX,
    EMA_LONG_MIN,
    EMA_RATIO_MAX,
    EMA_RATIO_MIN,
    EMA_SHORT_COMMON_MAX,
    EMA_SHORT_COMMON_MIN,
    EMA_SHORT_MAX,
    EMA_SHORT_MIN,
)
from app.backtesting.validators.ema_validator import EMAValidator
from tests.backtesting.validators.validator_test_utils import (
    assert_validator_base_attributes,
    assert_validators_independent,
    assert_warnings_list_fresh,
    assert_warnings_reset_between_calls,
)


# ==================== Initialization Tests ====================

class TestEMAValidatorInitialization:
    """Test EMAValidator initialization."""

    def test_inherits_from_validator_base(self):
        """EMAValidator should inherit from Validator base class."""
        validator = EMAValidator()
        assert_validator_base_attributes(validator)

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        assert_validators_independent(
            EMAValidator,
            {'short_ema_period': 9, 'long_ema_period': 21},
            {'short_ema_period': 12, 'long_ema_period': 26}
        )


# ==================== Short EMA Period Validation Tests ====================

class TestShortEMAPeriodValidation:
    """Test short_ema_period parameter validation."""

    @pytest.mark.parametrize("valid_short", [5, 6, 9, 10, 12, 15, 20, 21])
    def test_accepts_valid_positive_integers(self, valid_short):
        """Valid short EMA periods should be accepted."""
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=valid_short, long_ema_period=valid_short * 2)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_short", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_short):
        """Zero and negative short EMA periods should raise ValueError."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="short EMA period must be a positive integer"):
            validator.validate(short_ema_period=invalid_short, long_ema_period=50)

    @pytest.mark.parametrize("invalid_type", [1.5, 9.9, "9", "12", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for short EMA period."""
        validator = EMAValidator()

        with pytest.raises(ValueError):
            validator.validate(short_ema_period=invalid_type, long_ema_period=50)

    def test_rejects_float_even_if_whole_number(self):
        """Float values should be rejected even if they represent whole numbers."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="short EMA period must be a positive integer"):
            validator.validate(short_ema_period=9.0, long_ema_period=21)


# ==================== Long EMA Period Validation Tests ====================

class TestLongEMAPeriodValidation:
    """Test long_ema_period parameter validation."""

    @pytest.mark.parametrize("valid_long", [15, 20, 21, 26, 30, 40, 50])
    def test_accepts_valid_positive_integers(self, valid_long):
        """Valid long EMA periods should be accepted."""
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=9, long_ema_period=valid_long)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_long", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_long):
        """Zero and negative long EMA periods should raise ValueError."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="long EMA period must be a positive integer"):
            validator.validate(short_ema_period=9, long_ema_period=invalid_long)

    @pytest.mark.parametrize("invalid_type", [21.5, 26.9, "21", "26", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for long EMA period."""
        validator = EMAValidator()

        with pytest.raises(ValueError):
            validator.validate(short_ema_period=9, long_ema_period=invalid_type)

    def test_rejects_float_even_if_whole_number(self):
        """Float values should be rejected even if they represent whole numbers."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="long EMA period must be a positive integer"):
            validator.validate(short_ema_period=9, long_ema_period=21.0)


# ==================== Period Relationship Validation Tests ====================

class TestPeriodRelationship:
    """Test validation of relationship between short and long periods."""

    def test_rejects_when_short_equals_long(self):
        """Should reject when short EMA period equals long EMA period."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="Short EMA period .* must be less than long EMA period"):
            validator.validate(short_ema_period=20, long_ema_period=20)

    def test_rejects_when_short_greater_than_long(self):
        """Should reject when short EMA period is greater than long EMA period."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="Short EMA period .* must be less than long EMA period"):
            validator.validate(short_ema_period=26, long_ema_period=12)

    def test_accepts_when_short_less_than_long(self):
        """Should accept when short EMA period is less than long EMA period."""
        validator = EMAValidator()

        warnings = validator.validate(short_ema_period=9, long_ema_period=21)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    def test_error_message_includes_both_values(self):
        """Error message should include both period values for clarity."""
        validator = EMAValidator()

        with pytest.raises(ValueError, match="26.*12"):
            validator.validate(short_ema_period=26, long_ema_period=12)


# ==================== Short EMA Range Warning Tests ====================

class TestShortEMAWarnings:
    """Test warning generation for short EMA period values."""

    def test_warns_when_short_too_small(self):
        """Should warn when short EMA period is below minimum threshold."""
        validator = EMAValidator()
        too_small = EMA_SHORT_MIN - 1

        warnings = validator.validate(short_ema_period=too_small, long_ema_period=too_small * 3)

        assert len(warnings) > 0
        assert any('too short' in w.lower() for w in warnings)
        assert any('noise' in w.lower() for w in warnings)
        assert any(str(EMA_SHORT_MIN) in w for w in warnings)

    def test_warns_when_short_too_large(self):
        """Should warn when short EMA period is above maximum threshold."""
        validator = EMAValidator()
        too_large = EMA_SHORT_MAX + 1

        warnings = validator.validate(short_ema_period=too_large, long_ema_period=too_large * 2)

        assert len(warnings) > 0
        assert any('too long' in w.lower() for w in warnings)
        assert any('miss' in w.lower() for w in warnings)
        assert any(str(EMA_SHORT_MAX) in w for w in warnings)

    def test_no_warning_in_common_range(self):
        """Should not warn when short EMA is in common range."""
        validator = EMAValidator()
        common_value = (EMA_SHORT_COMMON_MIN + EMA_SHORT_COMMON_MAX) // 2

        warnings = validator.validate(short_ema_period=common_value, long_ema_period=common_value * 2)

        # Should have no short EMA-related warnings
        short_warnings = [w for w in warnings if 'short ema' in w.lower()]
        assert len(short_warnings) == 0

    def test_no_warning_at_minimum_boundary(self):
        """Should not warn when short EMA is exactly at minimum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MIN,
            long_ema_period=EMA_SHORT_MIN * 2
        )

        short_warnings = [w for w in warnings if 'short ema' in w.lower() and 'too short' in w.lower()]
        assert len(short_warnings) == 0

    def test_no_warning_at_maximum_boundary(self):
        """Should not warn when short EMA is exactly at maximum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MAX,
            long_ema_period=EMA_SHORT_MAX * 2
        )

        short_warnings = [w for w in warnings if 'short ema' in w.lower() and 'too long' in w.lower()]
        assert len(short_warnings) == 0

    def test_warns_just_below_minimum(self):
        """Should warn when short EMA is just below minimum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MIN - 1,
            long_ema_period=(EMA_SHORT_MIN - 1) * 3
        )

        assert len(warnings) > 0
        assert any('too short' in w.lower() for w in warnings)

    def test_warns_just_above_maximum(self):
        """Should warn when short EMA is just above maximum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MAX + 1,
            long_ema_period=(EMA_SHORT_MAX + 1) * 2
        )

        assert len(warnings) > 0
        assert any('too long' in w.lower() for w in warnings)


# ==================== Long EMA Range Warning Tests ====================

class TestLongEMAWarnings:
    """Test warning generation for long EMA period values."""

    def test_warns_when_long_too_small(self):
        """Should warn when long EMA period is below minimum threshold."""
        validator = EMAValidator()
        too_small = EMA_LONG_MIN - 1

        warnings = validator.validate(short_ema_period=5, long_ema_period=too_small)

        assert len(warnings) > 0
        assert any('long ema' in w.lower() and 'too short' in w.lower() for w in warnings)
        assert any('trend' in w.lower() for w in warnings)
        assert any(str(EMA_LONG_MIN) in w for w in warnings)

    def test_warns_when_long_too_large(self):
        """Should warn when long EMA period is above maximum threshold."""
        validator = EMAValidator()
        too_large = EMA_LONG_MAX + 1

        warnings = validator.validate(short_ema_period=9, long_ema_period=too_large)

        assert len(warnings) > 0
        assert any('long ema' in w.lower() and 'too long' in w.lower() for w in warnings)
        assert any('miss' in w.lower() for w in warnings)
        assert any(str(EMA_LONG_MAX) in w for w in warnings)

    def test_no_warning_in_common_range(self):
        """Should not warn when long EMA is in common range."""
        validator = EMAValidator()
        common_value = (EMA_LONG_COMMON_MIN + EMA_LONG_COMMON_MAX) // 2

        warnings = validator.validate(short_ema_period=9, long_ema_period=common_value)

        # Should have no long EMA-related warnings
        long_warnings = [w for w in warnings if 'long ema' in w.lower()]
        assert len(long_warnings) == 0

    def test_no_warning_at_minimum_boundary(self):
        """Should not warn when long EMA is exactly at minimum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_MIN
        )

        long_warnings = [w for w in warnings if 'long ema' in w.lower() and 'too short' in w.lower()]
        assert len(long_warnings) == 0

    def test_no_warning_at_maximum_boundary(self):
        """Should not warn when long EMA is exactly at maximum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_MAX
        )

        long_warnings = [w for w in warnings if 'long ema' in w.lower() and 'too long' in w.lower()]
        assert len(long_warnings) == 0

    def test_warns_just_below_minimum(self):
        """Should warn when long EMA is just below minimum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=5,
            long_ema_period=EMA_LONG_MIN - 1
        )

        assert len(warnings) > 0
        assert any('long ema' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_just_above_maximum(self):
        """Should warn when long EMA is just above maximum threshold."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_MAX + 1
        )

        assert len(warnings) > 0
        assert any('long ema' in w.lower() and 'too long' in w.lower() for w in warnings)


# ==================== EMA Ratio Warning Tests ====================

class TestEMARatioWarnings:
    """Test warning generation for EMA period ratio."""

    def test_warns_when_ratio_too_small(self):
        """Should warn when long/short ratio is below minimum threshold."""
        validator = EMAValidator()
        # Create periods with ratio below minimum
        short = 10
        long = int(short * (EMA_RATIO_MIN - 0.1))

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        assert len(warnings) > 0
        assert any('ratio' in w.lower() and 'too close' in w.lower() for w in warnings)
        assert any('false signal' in w.lower() for w in warnings)

    def test_warns_when_ratio_too_large(self):
        """Should warn when long/short ratio is above maximum threshold."""
        validator = EMAValidator()
        # Create periods with ratio above maximum
        short = 5
        long = int(short * (EMA_RATIO_MAX + 0.5))

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        assert len(warnings) > 0
        assert any('ratio' in w.lower() and 'too wide' in w.lower() for w in warnings)
        assert any('miss signal' in w.lower() for w in warnings)

    def test_no_warning_for_optimal_ratios(self):
        """Should not warn for optimal EMA ratios."""
        validator = EMAValidator()

        # Test common optimal ratios
        optimal_pairs = [
            (9, 21),  # 2.33x ratio
            (12, 26),  # 2.17x ratio
            (10, 20),  # 2.0x ratio
            (8, 21),  # 2.63x ratio
        ]

        for short, long in optimal_pairs:
            warnings = validator.validate(short_ema_period=short, long_ema_period=long)
            ratio_warnings = [w for w in warnings if 'ratio' in w.lower()]
            assert len(ratio_warnings) == 0, f"Should not warn for {short}/{long} ratio"

    def test_no_warning_at_minimum_ratio_boundary(self):
        """Should not warn when ratio is exactly at minimum threshold."""
        validator = EMAValidator()
        short = 10
        long = int(short * EMA_RATIO_MIN)

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        ratio_warnings = [w for w in warnings if 'ratio' in w.lower() and 'too close' in w.lower()]
        assert len(ratio_warnings) == 0

    def test_no_warning_at_maximum_ratio_boundary(self):
        """Should not warn when ratio is exactly at maximum threshold."""
        validator = EMAValidator()
        short = 10
        long = int(short * EMA_RATIO_MAX)

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        ratio_warnings = [w for w in warnings if 'ratio' in w.lower() and 'too wide' in w.lower()]
        assert len(ratio_warnings) == 0

    def test_warns_just_below_minimum_ratio(self):
        """Should warn when ratio is just below minimum threshold."""
        validator = EMAValidator()
        short = 10
        long = int(short * (EMA_RATIO_MIN - 0.1))

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        assert len(warnings) > 0
        assert any('too close' in w.lower() for w in warnings)

    def test_warns_just_above_maximum_ratio(self):
        """Should warn when ratio is just above maximum threshold."""
        validator = EMAValidator()
        short = 5
        long = int(short * (EMA_RATIO_MAX + 0.5))

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        assert len(warnings) > 0
        assert any('too wide' in w.lower() for w in warnings)

    def test_warning_includes_actual_ratio_value(self):
        """Warning should include the actual ratio value for clarity."""
        validator = EMAValidator()
        short = 10
        long = 11  # Very close ratio

        warnings = validator.validate(short_ema_period=short, long_ema_period=long)

        assert len(warnings) > 0
        # Should include ratio value in warning
        assert any('1.1' in w for w in warnings)


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for all out-of-range parameters simultaneously."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MIN - 1,  # Too short
            long_ema_period=EMA_LONG_MAX + 1  # Too long
        )

        assert len(warnings) >= 2
        assert any('short ema' in w.lower() for w in warnings)
        assert any('long ema' in w.lower() for w in warnings)

    def test_all_three_warning_types(self):
        """Should generate warnings for short, long, and ratio simultaneously."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MIN - 1,  # Too short (warning 1)
            long_ema_period=EMA_SHORT_MIN  # Creates ratio issue (warning 2+)
        )

        assert len(warnings) >= 2

    def test_warnings_are_independent(self):
        """Warnings for different parameters should be independent."""
        validator = EMAValidator()

        # Only short warning
        warnings1 = validator.validate(
            short_ema_period=EMA_SHORT_MIN - 1,
            long_ema_period=30
        )

        # Only long warning
        warnings2 = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_MAX + 1
        )

        # Only ratio warning
        warnings3 = validator.validate(
            short_ema_period=10,
            long_ema_period=11
        )

        assert any('short' in w.lower() for w in warnings1)
        assert any('long' in w.lower() for w in warnings2)
        assert any('ratio' in w.lower() for w in warnings3)


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        assert_warnings_reset_between_calls(
            EMAValidator(),
            {'short_ema_period': EMA_SHORT_MIN - 1, 'long_ema_period': 50},
            {'short_ema_period': 9, 'long_ema_period': 21}
        )

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        assert_warnings_list_fresh(
            EMAValidator(),
            {'short_ema_period': 9, 'long_ema_period': 21}
        )


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=9,
            long_ema_period=21,
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
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=12,
            long_ema_period=26
        )

        assert isinstance(warnings, list)


# ==================== Integration Tests ====================

class TestEMAValidatorIntegration:
    """Test EMAValidator integration with various parameter combinations."""

    def test_realistic_conservative_parameters(self):
        """Test with realistic conservative EMA parameters."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=12,
            long_ema_period=26
        )

        assert len(warnings) == 0

    def test_realistic_aggressive_parameters(self):
        """Test with realistic aggressive EMA parameters."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=5,
            long_ema_period=15
        )

        assert len(warnings) == 0

    def test_classic_trading_pairs(self):
        """Test classic EMA trading pairs."""
        validator = EMAValidator()

        classic_pairs = [
            (9, 21),
            (12, 26),
            (10, 20),
            (8, 21),
            (5, 15),
        ]

        for short, long in classic_pairs:
            warnings = validator.validate(
                short_ema_period=short,
                long_ema_period=long
            )
            # Classic pairs should have minimal or no warnings
            assert len(warnings) <= 1, f"Classic pair {short}/{long} should have minimal warnings"

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=EMA_SHORT_MIN - 1,
            long_ema_period=EMA_LONG_MAX + 1
        )

        assert len(warnings) >= 2
        assert any('short' in w.lower() for w in warnings)
        assert any('long' in w.lower() for w in warnings)


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_periods(self):
        """Test with minimal valid period values."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=1,
            long_ema_period=2
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_very_large_periods(self):
        """Test with very large period values."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=100,
            long_ema_period=200
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_extreme_ratio(self):
        """Test with extreme period ratio."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=5,
            long_ema_period=100
        )

        # Should warn about ratio being too wide
        assert len(warnings) > 0
        assert any('ratio' in w.lower() for w in warnings)

    def test_barely_different_periods(self):
        """Test with barely different period values."""
        validator = EMAValidator()

        warnings = validator.validate(
            short_ema_period=20,
            long_ema_period=21
        )

        # Should warn about ratio being too close
        assert len(warnings) > 0
        assert any('ratio' in w.lower() and 'too close' in w.lower() for w in warnings)

    def test_common_range_boundaries(self):
        """Test at boundaries of common ranges."""
        validator = EMAValidator()

        # Test short EMA at common range boundaries
        warnings1 = validator.validate(
            short_ema_period=EMA_SHORT_COMMON_MIN,
            long_ema_period=EMA_SHORT_COMMON_MIN * 2
        )
        warnings2 = validator.validate(
            short_ema_period=EMA_SHORT_COMMON_MAX,
            long_ema_period=EMA_SHORT_COMMON_MAX * 2
        )

        # Common range values should not generate warnings
        assert len(warnings1) == 0
        assert len(warnings2) == 0

        # Test long EMA at common range boundaries
        warnings3 = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_COMMON_MIN
        )
        warnings4 = validator.validate(
            short_ema_period=9,
            long_ema_period=EMA_LONG_COMMON_MAX
        )

        assert len(warnings3) == 0
        assert len(warnings4) == 0
