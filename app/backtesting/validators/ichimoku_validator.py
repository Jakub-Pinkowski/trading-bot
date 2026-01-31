"""
Ichimoku Cloud Parameter Validator

This module contains the Ichimoku Cloud strategy parameter validator.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.constants import (
    ICHIMOKU_TENKAN_MIN,
    ICHIMOKU_TENKAN_MAX,
    ICHIMOKU_TENKAN_STANDARD,
    ICHIMOKU_KIJUN_MIN,
    ICHIMOKU_KIJUN_MAX,
    ICHIMOKU_KIJUN_STANDARD,
    ICHIMOKU_SENKOU_B_MIN,
    ICHIMOKU_SENKOU_B_MAX,
    ICHIMOKU_SENKOU_B_STANDARD,
    ICHIMOKU_DISPLACEMENT_MIN,
    ICHIMOKU_DISPLACEMENT_MAX,
    ICHIMOKU_DISPLACEMENT_STANDARD,
    ICHIMOKU_TENKAN_KIJUN_RATIO_MIN,
    ICHIMOKU_TENKAN_KIJUN_RATIO_MAX,
    ICHIMOKU_KIJUN_SENKOU_RATIO_MIN,
    ICHIMOKU_KIJUN_SENKOU_RATIO_MAX,
)


class IchimokuValidator(Validator):
    """Validator for Ichimoku Cloud strategy parameters."""

    def validate(self, tenkan_period, kijun_period, senkou_span_b_period, displacement, **kwargs):
        """
        Enhanced validation for Ichimoku parameters with guidance on reasonable ranges.

        Reasonable ranges based on traditional Ichimoku settings:
        - Tenkan-sen: 7-12 (9 is traditional)
        - Kijun-sen: 22-30 (26 is traditional)
        - Senkou Span B: 44-60 (52 is traditional)
        - Displacement: 22-30 (26 is traditional, should match Kijun-sen)
        - Traditional Ichimoku: 9/26/52/26 based on Japanese market cycles

        Args:
            tenkan_period: Tenkan-sen (conversion line) period
            kijun_period: Kijun-sen (base line) period
            senkou_span_b_period: Senkou Span B (leading span B) period
            displacement: Cloud displacement/offset
            **kwargs: Additional parameters (ignored)

        Returns:
            List of warning messages

        Raises:
            ValueError: If parameters have invalid types or values
        """
        self.warnings = []

        # Type validation
        self.validate_positive_integer(tenkan_period, "tenkan period")
        self.validate_positive_integer(kijun_period, "kijun period")
        self.validate_positive_integer(senkou_span_b_period, "senkou span B period")
        self.validate_positive_integer(displacement, "displacement")

        # Tenkan period validation
        if tenkan_period < ICHIMOKU_TENKAN_MIN:
            self.warnings.append(
                f"Ichimoku Tenkan period {tenkan_period} is quite short and may be too sensitive. "
                f"Consider using {ICHIMOKU_TENKAN_MIN}-{ICHIMOKU_TENKAN_MAX} range ({ICHIMOKU_TENKAN_STANDARD} is traditional)."
            )
        elif tenkan_period > ICHIMOKU_TENKAN_MAX:
            self.warnings.append(
                f"Ichimoku Tenkan period {tenkan_period} may be too slow for conversion line. "
                f"Consider using {ICHIMOKU_TENKAN_MIN}-{ICHIMOKU_TENKAN_MAX} range ({ICHIMOKU_TENKAN_STANDARD} is traditional)."
            )

        # Kijun period validation
        if kijun_period < ICHIMOKU_KIJUN_MIN:
            self.warnings.append(
                f"Ichimoku Kijun period {kijun_period} may be too short for baseline. "
                f"Consider using {ICHIMOKU_KIJUN_MIN}-{ICHIMOKU_KIJUN_MAX} range ({ICHIMOKU_KIJUN_STANDARD} is traditional)."
            )
        elif kijun_period > ICHIMOKU_KIJUN_MAX:
            self.warnings.append(
                f"Ichimoku Kijun period {kijun_period} may be too slow for trend confirmation. "
                f"Consider using {ICHIMOKU_KIJUN_MIN}-{ICHIMOKU_KIJUN_MAX} range ({ICHIMOKU_KIJUN_STANDARD} is traditional)."
            )

        # Senkou Span B period validation
        if senkou_span_b_period < ICHIMOKU_SENKOU_B_MIN:
            self.warnings.append(
                f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too short for cloud formation. "
                f"Consider using {ICHIMOKU_SENKOU_B_MIN}-{ICHIMOKU_SENKOU_B_MAX} range ({ICHIMOKU_SENKOU_B_STANDARD} is traditional)."
            )
        elif senkou_span_b_period > ICHIMOKU_SENKOU_B_MAX:
            self.warnings.append(
                f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too slow. "
                f"Consider using {ICHIMOKU_SENKOU_B_MIN}-{ICHIMOKU_SENKOU_B_MAX} range ({ICHIMOKU_SENKOU_B_STANDARD} is traditional)."
            )

        # Displacement validation
        if displacement < ICHIMOKU_DISPLACEMENT_MIN:
            self.warnings.append(
                f"Ichimoku displacement {displacement} may be too short for proper cloud projection. "
                f"Consider using {ICHIMOKU_DISPLACEMENT_MIN}-{ICHIMOKU_DISPLACEMENT_MAX} range ({ICHIMOKU_DISPLACEMENT_STANDARD} is traditional)."
            )
        elif displacement > ICHIMOKU_DISPLACEMENT_MAX:
            self.warnings.append(
                f"Ichimoku displacement {displacement} may project too far into future. "
                f"Consider using {ICHIMOKU_DISPLACEMENT_MIN}-{ICHIMOKU_DISPLACEMENT_MAX} range ({ICHIMOKU_DISPLACEMENT_STANDARD} is traditional)."
            )

        # Traditional ratios validation
        tenkan_kijun_ratio = kijun_period / tenkan_period
        if tenkan_kijun_ratio < ICHIMOKU_TENKAN_KIJUN_RATIO_MIN or tenkan_kijun_ratio > ICHIMOKU_TENKAN_KIJUN_RATIO_MAX:
            self.warnings.append(
                f"Ichimoku Tenkan/Kijun ratio ({tenkan_kijun_ratio:.1f}) deviates from traditional ~3:1 ratio. "
                f"Consider maintaining traditional proportions (e.g., {ICHIMOKU_TENKAN_STANDARD}/{ICHIMOKU_KIJUN_STANDARD})."
            )

        kijun_senkou_ratio = senkou_span_b_period / kijun_period
        if kijun_senkou_ratio < ICHIMOKU_KIJUN_SENKOU_RATIO_MIN or kijun_senkou_ratio > ICHIMOKU_KIJUN_SENKOU_RATIO_MAX:
            self.warnings.append(
                f"Ichimoku Kijun/Senkou B ratio ({kijun_senkou_ratio:.1f}) deviates from traditional 2:1 ratio. "
                f"Consider maintaining traditional proportions (e.g., {ICHIMOKU_KIJUN_STANDARD}/{ICHIMOKU_SENKOU_B_STANDARD})."
            )

        # Displacement vs Kijun check
        if displacement != kijun_period:
            self.warnings.append(
                f"Ichimoku displacement ({displacement}) differs from Kijun period ({kijun_period}). "
                f"Traditional Ichimoku uses same value for both (typically {ICHIMOKU_DISPLACEMENT_STANDARD})."
            )

        # Traditional combination check
        if (tenkan_period, kijun_period, senkou_span_b_period, displacement) == (
                ICHIMOKU_TENKAN_STANDARD,
                ICHIMOKU_KIJUN_STANDARD,
                ICHIMOKU_SENKOU_B_STANDARD,
                ICHIMOKU_DISPLACEMENT_STANDARD
        ):
            self.warnings.append(
                f"Using traditional Ichimoku parameters ({ICHIMOKU_TENKAN_STANDARD}/{ICHIMOKU_KIJUN_STANDARD}/{ICHIMOKU_SENKOU_B_STANDARD}/{ICHIMOKU_DISPLACEMENT_STANDARD}) - based on Japanese market cycles."
            )

        return self.warnings
