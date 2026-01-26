"""
Base Validator Class

This module contains the base Validator class that provides reusable validation
methods for all strategy parameter validators. All specific validators should
inherit from this base class.
"""


class Validator:
    """Base class for parameter validators with reusable validation methods."""

    def __init__(self):
        """Initialize the validator with an empty warnings list."""
        self.warnings = []

    def validate_range(self, value, name, min_val, max_val, rec_min, rec_max, 
                      min_msg_template=None, max_msg_template=None):
        """
        Reusable range validation with warnings for recommended ranges.

        Args:
            value: The value to validate
            name: Parameter name for error messages
            min_val: Absolute minimum value (raises error if violated)
            max_val: Absolute maximum value (raises error if violated)
            rec_min: Recommended minimum value (generates warning if below)
            rec_max: Recommended maximum value (generates warning if above)
            min_msg_template: Optional custom message for below recommended minimum
            max_msg_template: Optional custom message for above recommended maximum

        Raises:
            ValueError: If value is outside absolute min/max range
        """
        # Check absolute bounds
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

        # Check recommended range and add warnings
        if value < rec_min:
            if min_msg_template:
                self.warnings.append(min_msg_template)
            else:
                self.warnings.append(f"{name} {value} is below recommended range ({rec_min}-{rec_max})")
        elif value > rec_max:
            if max_msg_template:
                self.warnings.append(max_msg_template)
            else:
                self.warnings.append(f"{name} {value} is above recommended range ({rec_min}-{rec_max})")

    def validate_gap(self, gap, name, min_gap, max_gap, min_msg=None, max_msg=None):
        """
        Validate a gap/difference between two parameters.

        Args:
            gap: The gap value to validate
            name: Gap name for messages
            min_gap: Minimum recommended gap
            max_gap: Maximum recommended gap
            min_msg: Optional custom message for gap too small
            max_msg: Optional custom message for gap too large
        """
        if gap < min_gap:
            if min_msg:
                self.warnings.append(min_msg)
            else:
                self.warnings.append(f"{name} ({gap}) is below recommended minimum {min_gap}")
        elif gap > max_gap:
            if max_msg:
                self.warnings.append(max_msg)
            else:
                self.warnings.append(f"{name} ({gap}) is above recommended maximum {max_gap}")

    def validate_ratio(self, ratio, name, min_ratio, max_ratio, min_msg=None, max_msg=None):
        """
        Validate a ratio between two parameters.

        Args:
            ratio: The ratio value to validate
            name: Ratio name for messages
            min_ratio: Minimum recommended ratio
            max_ratio: Maximum recommended ratio
            min_msg: Optional custom message for ratio too small
            max_msg: Optional custom message for ratio too large
        """
        if ratio < min_ratio:
            if min_msg:
                self.warnings.append(min_msg)
            else:
                self.warnings.append(f"{name} ratio ({ratio:.1f}) is below recommended minimum {min_ratio}")
        elif ratio > max_ratio:
            if max_msg:
                self.warnings.append(max_msg)
            else:
                self.warnings.append(f"{name} ratio ({ratio:.1f}) is above recommended maximum {max_ratio}")

    def add_info_message(self, message):
        """
        Add an informational message to warnings.

        Args:
            message: The informational message to add
        """
        self.warnings.append(message)

    def validate(self, **kwargs):
        """
        Validate parameters. Must be implemented by subclasses.

        Args:
            **kwargs: Strategy-specific parameters

        Returns:
            List of warning messages

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement validate() method")
