"""
Tests for ichimoku_validator module.

Tests cover:
- IchimokuValidator class initialization and inheritance
- Tenkan period validation (positive integers, range warnings)
- Kijun period validation (positive integers, range warnings)
- Senkou Span B period validation (positive integers, range warnings)
- Displacement validation (positive integers, range warnings)
- Tenkan/Kijun ratio validation (traditional 3:1 ratio)
- Kijun/Senkou B ratio validation (traditional 2:1 ratio)
- Displacement/Kijun relationship validation (should match)
- Integration with base validator functions
- Edge cases and boundary values
- Traditional Ichimoku settings (9/26/52/26)
"""
import pytest

from app.backtesting.validators.constants import (
    ICHIMOKU_DISPLACEMENT_MAX,
    ICHIMOKU_DISPLACEMENT_MIN,
    ICHIMOKU_DISPLACEMENT_STANDARD,
    ICHIMOKU_KIJUN_MAX,
    ICHIMOKU_KIJUN_MIN,
    ICHIMOKU_KIJUN_SENKOU_RATIO_MAX,
    ICHIMOKU_KIJUN_SENKOU_RATIO_MIN,
    ICHIMOKU_KIJUN_STANDARD,
    ICHIMOKU_SENKOU_B_MAX,
    ICHIMOKU_SENKOU_B_MIN,
    ICHIMOKU_SENKOU_B_STANDARD,
    ICHIMOKU_TENKAN_KIJUN_RATIO_MAX,
    ICHIMOKU_TENKAN_KIJUN_RATIO_MIN,
    ICHIMOKU_TENKAN_MAX,
    ICHIMOKU_TENKAN_MIN,
    ICHIMOKU_TENKAN_STANDARD,
)
from app.backtesting.validators.ichimoku_validator import IchimokuValidator


# ==================== Initialization Tests ====================

class TestIchimokuValidatorInitialization:
    """Test IchimokuValidator initialization."""

    def test_inherits_from_validator_base(self):
        """IchimokuValidator should inherit from Validator base class."""
        validator = IchimokuValidator()

        assert hasattr(validator, 'warnings')
        assert hasattr(validator, 'reset_warnings')
        assert hasattr(validator, 'validate')
        assert validator.warnings == []

    def test_multiple_instances_independent(self):
        """Multiple validator instances should have independent state."""
        validator1 = IchimokuValidator()
        validator2 = IchimokuValidator()

        validator1.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )
        validator2.validate(
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22
        )

        # Validators should have different warning states
        assert validator1.warnings != validator2.warnings or (
                len(validator1.warnings) == 0 and len(validator2.warnings) == 0
        )


# ==================== Tenkan Period Validation Tests ====================

class TestTenkanPeriodValidation:
    """Test tenkan_period parameter validation."""

    @pytest.mark.parametrize("valid_tenkan", [7, 8, 9, 10, 11, 12])
    def test_accepts_valid_positive_integers(self, valid_tenkan):
        """Valid Tenkan periods should be accepted."""
        validator = IchimokuValidator()
        warnings = validator.validate(
            tenkan_period=valid_tenkan,
            kijun_period=valid_tenkan * 3,
            senkou_span_b_period=valid_tenkan * 6,
            displacement=valid_tenkan * 3
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_tenkan", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_tenkan):
        """Zero and negative Tenkan periods should raise ValueError."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError, match="tenkan period must be a positive integer"):
            validator.validate(
                tenkan_period=invalid_tenkan,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=26
            )

    @pytest.mark.parametrize("invalid_type", [9.5, "9", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for Tenkan period."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError):
            validator.validate(
                tenkan_period=invalid_type,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=26
            )


# ==================== Kijun Period Validation Tests ====================

class TestKijunPeriodValidation:
    """Test kijun_period parameter validation."""

    @pytest.mark.parametrize("valid_kijun", [22, 24, 26, 28, 30])
    def test_accepts_valid_positive_integers(self, valid_kijun):
        """Valid Kijun periods should be accepted."""
        validator = IchimokuValidator()
        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=valid_kijun,
            senkou_span_b_period=valid_kijun * 2,
            displacement=valid_kijun
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_kijun", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_kijun):
        """Zero and negative Kijun periods should raise ValueError."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError, match="kijun period must be a positive integer"):
            validator.validate(
                tenkan_period=9,
                kijun_period=invalid_kijun,
                senkou_span_b_period=52,
                displacement=26
            )

    @pytest.mark.parametrize("invalid_type", [26.5, "26", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for Kijun period."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError):
            validator.validate(
                tenkan_period=9,
                kijun_period=invalid_type,
                senkou_span_b_period=52,
                displacement=26
            )


# ==================== Senkou Span B Period Validation Tests ====================

class TestSenkouSpanBPeriodValidation:
    """Test senkou_span_b_period parameter validation."""

    @pytest.mark.parametrize("valid_senkou", [44, 48, 52, 56, 60])
    def test_accepts_valid_positive_integers(self, valid_senkou):
        """Valid Senkou Span B periods should be accepted."""
        validator = IchimokuValidator()
        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=valid_senkou,
            displacement=26
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_senkou", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_senkou):
        """Zero and negative Senkou Span B periods should raise ValueError."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError, match="senkou span B period must be a positive integer"):
            validator.validate(
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=invalid_senkou,
                displacement=26
            )

    @pytest.mark.parametrize("invalid_type", [52.5, "52", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for Senkou Span B period."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError):
            validator.validate(
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=invalid_type,
                displacement=26
            )


# ==================== Displacement Validation Tests ====================

class TestDisplacementValidation:
    """Test displacement parameter validation."""

    @pytest.mark.parametrize("valid_displacement", [22, 24, 26, 28, 30])
    def test_accepts_valid_positive_integers(self, valid_displacement):
        """Valid displacement values should be accepted."""
        validator = IchimokuValidator()
        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=valid_displacement
        )

        assert isinstance(warnings, list)
        # Should not raise ValueError

    @pytest.mark.parametrize("invalid_displacement", [0, -1, -10, -100])
    def test_rejects_zero_and_negative(self, invalid_displacement):
        """Zero and negative displacement values should raise ValueError."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError, match="displacement must be a positive integer"):
            validator.validate(
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=invalid_displacement
            )

    @pytest.mark.parametrize("invalid_type", [26.5, "26", [], {}, None, True, False])
    def test_rejects_non_integer_types(self, invalid_type):
        """Non-integer types should raise ValueError for displacement."""
        validator = IchimokuValidator()

        with pytest.raises(ValueError):
            validator.validate(
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=invalid_type
            )


# ==================== Tenkan Range Warning Tests ====================

class TestTenkanRangeWarnings:
    """Test warning generation for Tenkan period values."""

    def test_warns_when_tenkan_too_small(self):
        """Should warn when Tenkan period is below minimum threshold."""
        validator = IchimokuValidator()
        too_small = ICHIMOKU_TENKAN_MIN - 1

        warnings = validator.validate(
            tenkan_period=too_small,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )

        assert len(warnings) > 0
        assert any('tenkan' in w.lower() and 'too short' in w.lower() for w in warnings)
        assert any(str(ICHIMOKU_TENKAN_STANDARD) in w for w in warnings)

    def test_warns_when_tenkan_too_large(self):
        """Should warn when Tenkan period is above maximum threshold."""
        validator = IchimokuValidator()
        too_large = ICHIMOKU_TENKAN_MAX + 1

        warnings = validator.validate(
            tenkan_period=too_large,
            kijun_period=too_large * 3,
            senkou_span_b_period=too_large * 6,
            displacement=too_large * 3
        )

        assert len(warnings) > 0
        assert any('tenkan' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_tenkan(self):
        """Should not warn for traditional Tenkan value (9)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        tenkan_warnings = [w for w in warnings if 'tenkan' in w.lower() and 'too' in w.lower()]
        assert len(tenkan_warnings) == 0


# ==================== Kijun Range Warning Tests ====================

class TestKijunRangeWarnings:
    """Test warning generation for Kijun period values."""

    def test_warns_when_kijun_too_small(self):
        """Should warn when Kijun period is below minimum threshold."""
        validator = IchimokuValidator()
        too_small = ICHIMOKU_KIJUN_MIN - 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=too_small,
            senkou_span_b_period=52,
            displacement=too_small
        )

        assert len(warnings) > 0
        assert any('kijun' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_when_kijun_too_large(self):
        """Should warn when Kijun period is above maximum threshold."""
        validator = IchimokuValidator()
        too_large = ICHIMOKU_KIJUN_MAX + 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=too_large,
            senkou_span_b_period=too_large * 2,
            displacement=too_large
        )

        assert len(warnings) > 0
        assert any('kijun' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_kijun(self):
        """Should not warn for traditional Kijun value (26)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        kijun_warnings = [w for w in warnings if 'kijun' in w.lower() and 'too' in w.lower()]
        assert len(kijun_warnings) == 0


# ==================== Senkou Span B Range Warning Tests ====================

class TestSenkouSpanBRangeWarnings:
    """Test warning generation for Senkou Span B period values."""

    def test_warns_when_senkou_too_small(self):
        """Should warn when Senkou Span B period is below minimum threshold."""
        validator = IchimokuValidator()
        too_small = ICHIMOKU_SENKOU_B_MIN - 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=too_small,
            displacement=26
        )

        assert len(warnings) > 0
        assert any('senkou span b' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_when_senkou_too_large(self):
        """Should warn when Senkou Span B period is above maximum threshold."""
        validator = IchimokuValidator()
        too_large = ICHIMOKU_SENKOU_B_MAX + 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=too_large,
            displacement=26
        )

        assert len(warnings) > 0
        assert any('senkou span b' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_senkou(self):
        """Should not warn for traditional Senkou Span B value (52)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        senkou_warnings = [w for w in warnings if 'senkou' in w.lower() and 'too' in w.lower()]
        assert len(senkou_warnings) == 0


# ==================== Displacement Range Warning Tests ====================

class TestDisplacementRangeWarnings:
    """Test warning generation for displacement values."""

    def test_warns_when_displacement_too_small(self):
        """Should warn when displacement is below minimum threshold."""
        validator = IchimokuValidator()
        too_small = ICHIMOKU_DISPLACEMENT_MIN - 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=too_small
        )

        assert len(warnings) > 0
        assert any('displacement' in w.lower() and 'too short' in w.lower() for w in warnings)

    def test_warns_when_displacement_too_large(self):
        """Should warn when displacement is above maximum threshold."""
        validator = IchimokuValidator()
        too_large = ICHIMOKU_DISPLACEMENT_MAX + 1

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=too_large
        )

        assert len(warnings) > 0
        assert any('displacement' in w.lower() and 'too long' in w.lower() for w in warnings)

    def test_no_warning_for_standard_displacement(self):
        """Should not warn for traditional displacement value (26)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        displacement_warnings = [
            w for w in warnings
            if 'displacement' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(displacement_warnings) == 0


# ==================== Tenkan/Kijun Ratio Warning Tests ====================

class TestTenkanKijunRatioWarnings:
    """Test warning generation for Tenkan/Kijun ratio."""

    def test_warns_when_ratio_too_small(self):
        """Should warn when Tenkan/Kijun ratio is below minimum threshold."""
        validator = IchimokuValidator()
        # Create ratio below minimum (e.g., 15/10 = 1.5, below ~2.5 minimum)
        tenkan = 10
        kijun = int(tenkan * (ICHIMOKU_TENKAN_KIJUN_RATIO_MIN - 0.5))

        warnings = validator.validate(
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_span_b_period=kijun * 2,
            displacement=kijun
        )

        assert len(warnings) > 0
        assert any('tenkan/kijun ratio' in w.lower() for w in warnings)

    def test_warns_when_ratio_too_large(self):
        """Should warn when Tenkan/Kijun ratio is above maximum threshold."""
        validator = IchimokuValidator()
        # Create ratio above maximum (e.g., 40/10 = 4.0, above ~3.5 maximum)
        tenkan = 10
        kijun = int(tenkan * (ICHIMOKU_TENKAN_KIJUN_RATIO_MAX + 0.5))

        warnings = validator.validate(
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_span_b_period=kijun * 2,
            displacement=kijun
        )

        assert len(warnings) > 0
        assert any('tenkan/kijun ratio' in w.lower() for w in warnings)

    def test_no_warning_for_traditional_ratio(self):
        """Should not warn for traditional 9/26 ratio (~2.89)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        ratio_warnings = [w for w in warnings if 'tenkan/kijun ratio' in w.lower()]
        assert len(ratio_warnings) == 0


# ==================== Kijun/Senkou B Ratio Warning Tests ====================

class TestKijunSenkouRatioWarnings:
    """Test warning generation for Kijun/Senkou B ratio."""

    def test_warns_when_ratio_too_small(self):
        """Should warn when Kijun/Senkou B ratio is below minimum threshold."""
        validator = IchimokuValidator()
        # Create ratio below minimum (e.g., 26/20 = 1.3, below ~1.8 minimum)
        kijun = 26
        senkou = int(kijun * (ICHIMOKU_KIJUN_SENKOU_RATIO_MIN - 0.3))

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=kijun,
            senkou_span_b_period=senkou,
            displacement=kijun
        )

        assert len(warnings) > 0
        assert any('kijun/senkou b ratio' in w.lower() for w in warnings)

    def test_warns_when_ratio_too_large(self):
        """Should warn when Kijun/Senkou B ratio is above maximum threshold."""
        validator = IchimokuValidator()
        # Create ratio above maximum (e.g., 26/65 = 2.5, above ~2.2 maximum)
        kijun = 26
        senkou = int(kijun * (ICHIMOKU_KIJUN_SENKOU_RATIO_MAX + 0.3))

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=kijun,
            senkou_span_b_period=senkou,
            displacement=kijun
        )

        assert len(warnings) > 0
        assert any('kijun/senkou b ratio' in w.lower() for w in warnings)

    def test_no_warning_for_traditional_ratio(self):
        """Should not warn for traditional 26/52 ratio (2.0)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_STANDARD,
            kijun_period=ICHIMOKU_KIJUN_STANDARD,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_STANDARD,
            displacement=ICHIMOKU_DISPLACEMENT_STANDARD
        )

        ratio_warnings = [w for w in warnings if 'kijun/senkou b ratio' in w.lower()]
        assert len(ratio_warnings) == 0


# ==================== Displacement/Kijun Relationship Warning Tests ====================

class TestDisplacementKijunRelationship:
    """Test warning generation for displacement/Kijun relationship."""

    def test_warns_when_displacement_differs_from_kijun(self):
        """Should warn when displacement doesn't match Kijun period."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=20  # Different from Kijun
        )

        assert len(warnings) > 0
        assert any('displacement' in w.lower() and 'differs from kijun' in w.lower() for w in warnings)

    def test_no_warning_when_displacement_matches_kijun(self):
        """Should not warn when displacement matches Kijun period."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26  # Same as Kijun
        )

        relationship_warnings = [
            w for w in warnings
            if 'displacement' in w.lower() and 'differs' in w.lower()
        ]
        assert len(relationship_warnings) == 0


# ==================== Multiple Warnings Tests ====================

class TestMultipleWarnings:
    """Test that multiple warnings can be generated together."""

    def test_can_generate_multiple_warnings(self):
        """Should generate warnings for all out-of-range parameters simultaneously."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MIN - 1,  # Too short
            kijun_period=ICHIMOKU_KIJUN_MAX + 1,  # Too long
            senkou_span_b_period=ICHIMOKU_SENKOU_B_MAX + 1,  # Too long
            displacement=ICHIMOKU_DISPLACEMENT_MIN - 1  # Too short
        )

        assert len(warnings) >= 4
        assert any('tenkan' in w.lower() for w in warnings)
        assert any('kijun' in w.lower() for w in warnings)
        assert any('senkou' in w.lower() for w in warnings)
        assert any('displacement' in w.lower() for w in warnings)

    def test_all_warning_types(self):
        """Should generate warnings for ranges, ratios, and relationships."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MIN - 1,  # Range warning
            kijun_period=10,  # Creates bad ratio with Tenkan
            senkou_span_b_period=15,  # Creates bad ratio with Kijun
            displacement=20  # Doesn't match Kijun
        )

        # Should have multiple types of warnings
        assert len(warnings) >= 3


# ==================== Warning Reset Tests ====================

class TestWarningReset:
    """Test that warnings are properly reset between validations."""

    def test_warnings_reset_between_calls(self):
        """Warnings should be cleared on each validate() call."""
        validator = IchimokuValidator()

        # First validation with warnings
        warnings1 = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MIN - 1,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )
        assert len(warnings1) > 0

        # Second validation without warnings
        warnings2 = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )
        assert len(warnings2) == 0

    def test_warnings_list_is_fresh_each_call(self):
        """Each validate() call should return a fresh warnings list."""
        validator = IchimokuValidator()

        warnings1 = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )
        warnings2 = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )

        # Should have same content but be independent lists
        assert warnings1 == warnings2
        warnings1.append("extra")
        assert len(warnings1) != len(warnings2)


# ==================== Kwargs Handling Tests ====================

class TestKwargsHandling:
    """Test that extra keyword arguments are properly ignored."""

    def test_ignores_extra_kwargs(self):
        """Extra keyword arguments should be ignored without error."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            extra_param1="ignored",
            rollover=True,
            trailing=2.5
        )

        assert isinstance(warnings, list)
        # Should not raise any error


# ==================== Integration Tests ====================

class TestIchimokuValidatorIntegration:
    """Test IchimokuValidator integration with various parameter combinations."""

    def test_traditional_settings(self):
        """Test traditional Ichimoku settings (9/26/52/26)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )

        assert len(warnings) == 0

    def test_fast_ichimoku_settings(self):
        """Test faster Ichimoku settings for shorter timeframes."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22
        )

        assert len(warnings) == 0

    def test_slow_ichimoku_settings(self):
        """Test slower Ichimoku settings for longer timeframes."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=12,
            kijun_period=30,
            senkou_span_b_period=60,
            displacement=30
        )

        assert len(warnings) == 0

    def test_all_parameters_out_of_range(self):
        """Test with all parameters generating warnings."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MIN - 1,
            kijun_period=ICHIMOKU_KIJUN_MAX + 1,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_MAX + 1,
            displacement=ICHIMOKU_DISPLACEMENT_MIN - 1
        )

        assert len(warnings) >= 4


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_periods(self):
        """Test with minimal valid period values."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=1,
            kijun_period=2,
            senkou_span_b_period=3,
            displacement=2
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_very_large_periods(self):
        """Test with very large period values."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=50,
            kijun_period=150,
            senkou_span_b_period=300,
            displacement=150
        )

        # Should generate warnings but not raise errors
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_all_same_periods(self):
        """Test with all periods set to same value (invalid ratios)."""
        validator = IchimokuValidator()

        warnings = validator.validate(
            tenkan_period=26,
            kijun_period=26,
            senkou_span_b_period=26,
            displacement=26
        )

        # Should warn about ratios
        assert len(warnings) > 0
        assert any('ratio' in w.lower() for w in warnings)

    def test_boundary_values(self):
        """Test at exact boundary values."""
        validator = IchimokuValidator()

        # Test minimum boundaries
        warnings1 = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MIN,
            kijun_period=ICHIMOKU_KIJUN_MIN,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_MIN,
            displacement=ICHIMOKU_DISPLACEMENT_MIN
        )

        # Test maximum boundaries
        warnings2 = validator.validate(
            tenkan_period=ICHIMOKU_TENKAN_MAX,
            kijun_period=ICHIMOKU_KIJUN_MAX,
            senkou_span_b_period=ICHIMOKU_SENKOU_B_MAX,
            displacement=ICHIMOKU_DISPLACEMENT_MAX
        )

        # Boundary values should not generate range warnings
        tenkan_range_warnings1 = [
            w for w in warnings1
            if 'tenkan' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        tenkan_range_warnings2 = [
            w for w in warnings2
            if 'tenkan' in w.lower() and ('too short' in w.lower() or 'too long' in w.lower())
        ]
        assert len(tenkan_range_warnings1) == 0
        assert len(tenkan_range_warnings2) == 0
