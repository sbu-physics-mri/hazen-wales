"""Application-specific errors."""

from __future__ import annotations

# Local imports
from hazenlib.constants import MEASUREMENT_NAMES, MEASUREMENT_TYPES


class ShapeError(Exception):
    """Base exception for shapes."""

    def __init__(self, shape: str, msg: str | None = None) -> None:
        """Initialise the error."""
        if msg is None:
            # Default message
            msg = f"An error occured with {shape}"
        self.msg = msg
        self.shape = shape


class ShapeDetectionError(ShapeError):
    """Shape not found."""

    def __init__(self, shape: str, msg: str | None = None) -> None:
        """Initialise the error."""
        if msg is None:
            # Default message
            msg = f"Could not find shape: {shape}"
        super(ShapeError, self).__init__(msg)
        self.shape = shape


class MultipleShapesError(ShapeDetectionError):
    """Shape not found."""

    def __init__(self, shape: tuple[str], msg: str | None = None) -> None:
        """Initialise the error."""
        if msg is None:
            # Default message
            msg = (
                f"Multiple {shape}s found."
                " Multiple shape detection is currently unsupported."
            )

        super(ShapeDetectionError, self).__init__(msg)
        self.shape = shape


class ArgumentCombinationError(Exception):
    """Argument combination not valid."""

    def __init__(self, msg: str = "Invalid combination of arguments.") -> None:
        """Initialise the error."""
        super().__init__(msg)


class InvalidMeasurementNameError(ValueError):
    """Invalid Measurement Name Error."""

    def __init__(self, name: str) -> None:
        """Initialise the error."""
        msg = (
            f"Invalid measurement name: {name}."
            f" Must be one of {MEASUREMENT_NAMES}"
        )
        super().__init__(msg)


class InvalidMeasurementTypeError(ValueError):
    """Invalid Measurement Type Error."""

    def __init__(self, measurement_type: str) -> None:
        """Initialise the error."""
        msg = (
            f"Invalid measurement type: {measurement_type}."
            f" Must be one of {MEASUREMENT_TYPES}"
        )
        super().__init__(msg)


class NoDistortionCorrectionError(ValueError):
    """No distortion correction enabled for this dataset."""

    def __init__(self, name: str) -> None:
        """Initialise the error."""
        msg = f"Distortion correction not enabled for {name}"
        super().__init__(msg)
