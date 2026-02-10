"""Tests for the orchestration module."""

# ruff: noqa: PT009 PT027

import pathlib
import unittest
from unittest.mock import Mock, patch


from hazenlib.exceptions import UnknownAcquisitionTypeError, UnknownTaskNameError
from hazenlib.HazenTask import HazenTask
from hazenlib.orchestration import (
    ACRLargePhantomProtocol,
    AcquisitionType,
    Protocol,
    ProtocolResult,
    ProtocolStep,
    init_task,
)
from hazenlib.types import Result

from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestInitTask(unittest.TestCase):
    """Unit tests for task initialization."""

    def test_successful_initialization(self) -> None:
        """Verify successful task creation from registry."""
        # Arrange
        mock_task_class = Mock(spec=HazenTask)
        mock_module = Mock()
        mock_module.SNR = mock_task_class

        with patch(
            "hazenlib.orchestration.importlib.import_module",
            return_value=mock_module,
        ):
            # Act
            args = (
                "snr",
                ["file1.dcm", "file2.dcm"],
            )
            kwargs = {
                "report": True,
                "report_dir": TEST_REPORT_DIR,
            }
            result = init_task(
                *args,
                **kwargs,
            )

            # Assert
            self.assertEqual(result, mock_task_class.return_value)
            mock_task_class.assert_called_once_with(
                input_data = args[1],
                **kwargs,
            )

    def test_unknown_task_raises_value_error(self) -> None:
        """Verify ValueError raised for unknown task names."""
        with self.assertRaises(ValueError) as context:
            init_task(
                "unknown_task", [], report=False, report_dir=TEST_REPORT_DIR,
            )

        self.assertIn("Unknown task", str(context.exception))

    def test_missing_class_raises_import_error(self) -> None:
        """Verify ImportError when module lacks expected class."""
        mock_module = Mock()
        # Mock module without the expected class attribute
        del mock_module.SNR

        with (
            patch(
                "hazenlib.orchestration.importlib.import_module",
                return_value=mock_module,
            ),
            self.assertRaises(ImportError) as context,
        ):
            init_task("snr", [], report=False, report_dir=TEST_REPORT_DIR)

        self.assertIn("has no class", str(context.exception))


class TestAcquisitionType(unittest.TestCase):
    """Unit tests for AcquisitionType enum."""

    def test_from_string_t1(self) -> None:
        """Verify T1 string parsing."""
        result = AcquisitionType.from_string("t1")
        self.assertEqual(result, AcquisitionType.ACR_T1)
        # Case insensitive
        result = AcquisitionType.from_string("T1")
        self.assertEqual(result, AcquisitionType.ACR_T1)

    def test_from_string_t2(self) -> None:
        """Verify T2 string parsing."""
        result = AcquisitionType.from_string("t2")
        self.assertEqual(result, AcquisitionType.ACR_T2)

    def test_from_string_sagittal_variants(self) -> None:
        """Verify sagittal localizer parsing with spelling variants."""
        # Full string
        result = AcquisitionType.from_string("sagittal localiser")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        # American spelling
        result = AcquisitionType.from_string("sagittal localizer")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        # Abbreviations
        result = AcquisitionType.from_string("sagittal")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        result = AcquisitionType.from_string("localizer")
        self.assertEqual(result, AcquisitionType.ACR_SL)

    def test_from_string_invalid_raises_error(self) -> None:
        """Verify UnknownAcquisitionTypeError for invalid strings."""
        with self.assertRaises(UnknownAcquisitionTypeError):
            AcquisitionType.from_string("unknown_type")

        with self.assertRaises(UnknownAcquisitionTypeError):
            AcquisitionType.from_string("t3")


if __name__ == "__main__":
    unittest.main()
