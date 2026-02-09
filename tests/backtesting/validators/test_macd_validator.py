"""
Tests for macd_validator module.

Tests cover:
- MACDValidator class initialization and inheritance
- Fast period validation (positive integers, range warnings)
- Slow period validation (positive integers, range warnings)
- Signal period validation (positive integers, range warnings)
- Period relationship validation (fast < slow)
- Warning generation for out-of-range values
- Integration with base validator functions
- Edge cases and boundary values
- Standard MACD settings (12/26/9)
"""
import pytest

from app.backtesting.validators.constants import (
    MACD_FAST_MAX,
    MACD_FAST_MIN,
    MACD_FAST_STANDARD,
    MACD_SIGNAL_MAX,
    MACD_SIGNAL_MIN,
    MACD_SIGNAL_STANDARD,
    MACD_SLOW_MAX,
    MACD_SLOW_MIN,
    MACD_SLOW_STANDARD,
)
from app.backtesting.validators.macd_validator import MACDValidator
from tests.backtesting.validators.validator_test_utils import (
    assert_validator_base_attributes,
    assert_validators_independent,
    assert_warnings_list_fresh,
    assert_warnings_reset_between_calls,
)


# ==================== Initialization Tests ====================

class TestMACDValidatorInitialization:
    """Test MACDValidator initialization."""

    def test_inherits_from_validator_base(self):
        """MACDValidator should inherit from Validator base class."""
        validator = MACDValidator()
        assert_validator_base_attributes(validator)

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        assert_validators_independent(
            MACDValidator,
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'fast_period': 8, 'slow_period': 20, 'signal_period': 7}
        )


# ==================== Fast Period Validation Tests ====================

class TestFastPeriodValidation:
    """Test fast_period parameter validation."""

    @pytest.mark.parametrize("valid_fast", [8, 10, 12, 14, 15])
    def test_accepts_valid_positive_integers(self, valid_fast):
        """Valid fast periods should be accepted."""
        validator = MACDValidator()
        warnings = validator.validate(
            fast_period=valid_fast,
            slow_period=valid_fast * 2,
            signal_period=9
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_fast", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_fast):
        """Zero and negative fast periods should raise ValueError."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="fast period must be a positive integer"):
            validator.validate(fast_period=invalid_fast, slow_period=26, signal_period=9)

    @pytest.mark.parametrize("invalid_type", [12.5, "12", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for fast period."""
        validator = MACDValidator()

        with pytest.raises(ValueError):
            validator.validate(fast_period=invalid_type, slow_period=26, signal_period=9)


# ==================== Slow Period Validation Tests ====================

class TestSlowPeriodValidation:
    """Test slow_period parameter validation."""

    @pytest.mark.parametrize("valid_slow", [20, 22, 26, 28, 30])
    def test_accepts_valid_positive_integers(self, valid_slow):
        """Valid slow periods should be accepted."""
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=valid_slow, signal_period=9)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_slow", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_slow):
        """Zero and negative slow periods should raise ValueError."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="slow period must be a positive integer"):
            validator.validate(fast_period=12, slow_period=invalid_slow, signal_period=9)

    @pytest.mark.parametrize("invalid_type", [26.5, "26", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for slow period."""
        validator = MACDValidator()

        with pytest.raises(ValueError):
            validator.validate(fast_period=12, slow_period=invalid_type, signal_period=9)


# ==================== Signal Period Validation Tests ====================

class TestSignalPeriodValidation:
    """Test signal_period parameter validation."""

    @pytest.mark.parametrize("valid_signal", [7, 8, 9, 10, 12])
    def test_accepts_valid_positive_integers(self, valid_signal):
        """Valid signal periods should be accepted."""
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=26, signal_period=valid_signal)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_signal", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_signal):
        """Zero and negative signal periods should raise ValueError."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="signal period must be a positive integer"):
            validator.validate(fast_period=12, slow_period=26, signal_period=invalid_signal)

    @pytest.mark.parametrize("invalid_type", [9.5, "9", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for signal period."""
        validator = MACDValidator()

        with pytest.raises(ValueError):
            validator.validate(fast_period=12, slow_period=26, signal_period=invalid_type)


# ==================== Period Relationship Validation Tests ====================

class TestPeriodRelationship:
    """Test validation of relationship between fast and slow periods."""

    def test_rejects_when_fast_equals_slow(self):
        """Should reject when fast period equals slow period."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="Fast period .* must be less than slow period"):
            validator.validate(fast_period=20, slow_period=20, signal_period=9)

    def test_rejects_when_fast_greater_than_slow(self):
        """Should reject when fast period is greater than slow period."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="Fast period .* must be less than slow period"):
            validator.validate(fast_period=30, slow_period=12, signal_period=9)

    def test_accepts_when_fast_less_than_slow(self):
        """Should accept when fast period is less than slow period."""
        validator = MACDValidator()

        warnings = validator.validate(fast_period=12, slow_period=26, signal_period=9)

        assert isinstance(warnings, list)
        # Should not raise ValueError

    def test_error_message_includes_both_values(self):
        """Error message should include both period values for clarity."""
        validator = MACDValidator()

        with pytest.raises(ValueError, match="30.*12"):
            validator.validate(fast_period=30, slow_period=12, signal_period=9)


# ==================== Fast Period Range Warning Tests ====================

class TestFastPeriodRangeWarnings:
    """Test warning generation for fast period values."""

    def test_warns_when_fast_too_small(self):
        """Should warn when fast period is below minimum threshold."""
        validator = MACDValidator()
        too_small = MACD_FAST_MIN - 1

        warnings = validator.validate(
            fast_period=too_small,
            slow_period=26,
            signal_period=9
        )

        assert len(warnings) > 0
        assert any('fast period' in w.lower() and 'too short' in w.lower() for w in warnings)
        assert any('noise' in w.lower() for w in warnings)

    def test_warns_when_fast_too_large(self):
        """Should warn when fast period is above maximum threshold."""
        validator = MACDValidator()
        too_large = MACD_FAST_MAX + 1

        warnings = validator.validate(
            fast_period=too_large,
            slow_period=too_large * 2,
            signal_period=9
        )

        assert len(warnings) > 0
        assert any('fast period' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_fast(self):
        """Should not warn for standard fast period value (12)."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=MACD_FAST_STANDARD,
            slow_period=MACD_SLOW_STANDARD,
            signal_period=MACD_SIGNAL_STANDARD
        )

        fast_warnings = [w for w in warnings if 'fast period' in w.lower() and 'too' in w.lower()]
        assert len(fast_warnings) == 0


# ==================== Slow Period Range Warning Tests ====================

class TestSlowPeriodRangeWarnings:
    """Test warning generation for slow period values."""

    def test_warns_when_slow_too_small(self):
        """Should warn when slow period is below minimum threshold."""
        validator = MACDValidator()
        too_small = MACD_SLOW_MIN - 1

        warnings = validator.validate(
            fast_period=12,
            slow_period=too_small,
            signal_period=9
        )

        assert len(warnings) > 0
        assert any('slow period' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_when_slow_too_large(self):
        """Should warn when slow period is above maximum threshold."""
        validator = MACDValidator()
        too_large = MACD_SLOW_MAX + 1

        warnings = validator.validate(
            fast_period=12,
            slow_period=too_large,
            signal_period=9
        )

        assert len(warnings) > 0
        assert any('slow period' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_slow(self):
        """Should not warn for standard slow period value (26)."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=MACD_FAST_STANDARD,
            slow_period=MACD_SLOW_STANDARD,
            signal_period=MACD_SIGNAL_STANDARD
        )

        slow_warnings = [w for w in warnings if 'slow period' in w.lower() and 'too' in w.lower()]
        assert len(slow_warnings) == 0


# ==================== Signal Period Range Warning Tests ====================

class TestSignalPeriodRangeWarnings:
    """Test warning generation for signal period values."""

    def test_warns_when_signal_too_small(self):
        """Should warn when signal period is below minimum threshold."""
        validator = MACDValidator()
        too_small = MACD_SIGNAL_MIN - 1

        warnings = validator.validate(
            fast_period=12,
            slow_period=26,
            signal_period=too_small
        )

        assert len(warnings) > 0
        assert any('signal period' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_when_signal_too_large(self):
        """Should warn when signal period is above maximum threshold."""
        validator = MACDValidator()
        too_large = MACD_SIGNAL_MAX + 1

        warnings = validator.validate(
            fast_period=12,
            slow_period=26,
            signal_period=too_large
        )

        assert len(warnings) > 0
        assert any('signal period' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_signal(self):
        """Should not warn for standard signal period value (9)."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=MACD_FAST_STANDARD,
            slow_period=MACD_SLOW_STANDARD,
            signal_period=MACD_SIGNAL_STANDARD
        )

        signal_warnings = [w for w in warnings if 'signal period' in w.lower() and 'too' in w.lower()]
        assert len(signal_warnings) == 0


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for all out-of-range parameters simultaneously."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=MACD_FAST_MIN - 1,  # Too short
            slow_period=MACD_SLOW_MAX + 1,  # Too long
            signal_period=MACD_SIGNAL_MAX + 1  # Too long
        )

        assert len(warnings) >= 3
        assert any('fast' in w.lower() for w in warnings)
        assert any('slow' in w.lower() for w in warnings)
        assert any('signal' in w.lower() for w in warnings)

    def test_warnings_are_independent(self):
        """Warnings for different parameters should be independent."""
        validator = MACDValidator()

        # Only fast warning
        warnings1 = validator.validate(
            fast_period=MACD_FAST_MIN - 1,
            slow_period=26,
            signal_period=9
        )

        # Only slow warning
        warnings2 = validator.validate(
            fast_period=12,
            slow_period=MACD_SLOW_MAX + 1,
            signal_period=9
        )

        # Only signal warning
        warnings3 = validator.validate(
            fast_period=12,
            slow_period=26,
            signal_period=MACD_SIGNAL_MIN - 1
        )

        assert any('fast' in w.lower() for w in warnings1)
        assert any('slow' in w.lower() for w in warnings2)
        assert any('signal' in w.lower() for w in warnings3)


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        assert_warnings_reset_between_calls(
            MACDValidator(),
            {'fast_period': MACD_FAST_MIN - 1, 'slow_period': 26, 'signal_period': 9},
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        assert_warnings_list_fresh(
            MACDValidator(),
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            extra_param1="ignored",
            extra_param2=123,
            rollover=True,
            trailing=2.5,
            slippage_ticks=1
        )

        assert isinstance(warnings, list)
        # Should not raise any error


# ==================== Integration Tests ====================

class TestMACDValidatorIntegration:
    """Test MACDValidator integration with various parameter combinations."""

    def test_standard_macd_settings(self):
        """Test standard MACD settings (12/26/9)."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )

        assert len(warnings) == 0

    def test_fast_macd_settings(self):
        """Test fast MACD settings."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=8,
            slow_period=20,
            signal_period=7
        )

        assert len(warnings) == 0

    def test_slow_macd_settings(self):
        """Test slow MACD settings."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=15,
            slow_period=30,
            signal_period=12
        )

        assert len(warnings) == 0

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=MACD_FAST_MIN - 1,
            slow_period=MACD_SLOW_MAX + 1,
            signal_period=MACD_SIGNAL_MAX + 1
        )

        assert len(warnings) >= 3


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_periods(self):
        """Test with minimal valid period values."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=1,
            slow_period=2,
            signal_period=1
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_very_large_periods(self):
        """Test with very large period values."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=50,
            slow_period=100,
            signal_period=50
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_boundary_values(self):
        """Test at exact boundary values."""
        validator = MACDValidator()

        # Test minimum boundaries
        warnings1 = validator.validate(
            fast_period=MACD_FAST_MIN,
            slow_period=MACD_SLOW_MIN,
            signal_period=MACD_SIGNAL_MIN
        )

        # Test maximum boundaries
        warnings2 = validator.validate(
            fast_period=MACD_FAST_MAX,
            slow_period=MACD_SLOW_MAX,
            signal_period=MACD_SIGNAL_MAX
        )

        # Boundary values should not generate range warnings
        fast_warnings1 = [
            w for w in warnings1
            if 'fast period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(fast_warnings1) == 0

        fast_warnings2 = [
            w for w in warnings2
            if 'fast period' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(fast_warnings2) == 0

    def test_common_variations(self):
        """Test common MACD variations."""
        validator = MACDValidator()

        # Common variations that should not generate warnings
        common_settings = [
            (12, 26, 9),  # Standard
            (8, 20, 7),  # Fast
            (15, 30, 12),  # Slow
            (10, 22, 9),  # Custom
        ]

        for fast, slow, signal in common_settings:
            warnings = validator.validate(
                fast_period=fast,
                slow_period=slow,
                signal_period=signal
            )
            assert len(warnings) == 0, f"Common settings {fast}/{slow}/{signal} should not generate warnings"

    def test_barely_different_fast_slow(self):
        """Test with barely different fast/slow periods."""
        validator = MACDValidator()

        warnings = validator.validate(
            fast_period=20,
            slow_period=21,
            signal_period=9
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
