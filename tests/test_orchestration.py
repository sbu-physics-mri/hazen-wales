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


class TestProtocolStep(unittest.TestCase):
    """Unit tests for ProtocolStep dataclass."""

    def test_valid_initialization(self) -> None:
        """Verify ProtocolStep can be created with valid task name."""
        step = ProtocolStep("snr", AcquisitionType.ACR_T1)
        self.assertEqual(step.task_name, "snr")
        self.assertEqual(step.acquisition_type, AcquisitionType.ACR_T1)
        self.assertTrue(step.required)

    def test_valid_optional_parameters(self) -> None:
        """Verify required flag can be set to False."""
        step = ProtocolStep("ghosting", AcquisitionType.ACR_T2, required=False)
        self.assertFalse(step.required)

    def test_invalid_task_name_raises_error(self) -> None:
        """Verify UnknownTaskNameError raised for unregistered tasks."""
        with self.assertRaises(UnknownTaskNameError):
            ProtocolStep("nonexistent_task", AcquisitionType.ACR_T1)


class TestProtocol(unittest.TestCase):
    """Unit tests for Protocol dataclass."""

    def test_basic_initialization(self) -> None:
        """Verify Protocol can be initialized with name and steps."""
        steps = (
            ProtocolStep("snr", AcquisitionType.ACR_T1),
            ProtocolStep("ghosting", AcquisitionType.ACR_T1),
        )
        protocol = Protocol(name="Test Protocol", steps=steps)
        self.assertEqual(protocol.name, "Test Protocol")
        self.assertEqual(len(protocol.steps), 2)

    def test_empty_steps_default(self) -> None:
        """Verify Protocol defaults to empty steps tuple."""
        protocol = Protocol(name="Empty Protocol")
        self.assertEqual(protocol.steps, ())

    def test_from_config_not_implemented(self) -> None:
        """Verify from_config raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Protocol.from_config(TEST_DATA_DIR / "config.json")


class TestProtocolResult(unittest.TestCase):
    """Unit tests for ProtocolResult class."""

    def test_initialization(self) -> None:
        """Verify ProtocolResult initialization."""
        result = ProtocolResult(task="TestProtocol", desc="test description")
        self.assertEqual(result.task, "TestProtocol")
        self.assertEqual(result.desc, "test description")
        self.assertEqual(result.results, ())

    def test_add_result(self) -> None:
        """Verify results can be added to collection."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        mock_result = Result(task="SubTask", desc="subtask result")

        protocol_result.add_result(mock_result)

        self.assertEqual(len(protocol_result.results), 1)
        self.assertEqual(protocol_result.results[0], mock_result)

    def test_results_immutable(self) -> None:
        """Verify results property returns immutable tuple."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        protocol_result.add_result(Result(task="Task1", desc="desc1"))

        results = protocol_result.results
        self.assertIsInstance(results, tuple)

        # Verify immutability
        with self.assertRaises(TypeError):
            results[0] = Result(task="Task2", desc="desc2")

    def test_add_multiple_results(self) -> None:
        """Verify multiple results can be added and retrieved."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        results = [
            Result(task=f"Task{i}", desc=f"result{i}") for i in range(3)
        ]

        for r in results:
            protocol_result.add_result(r)

        self.assertEqual(len(protocol_result.results), 3)
        for i, r in enumerate(protocol_result.results):
            self.assertEqual(r.task, f"Task{i}")


if __name__ == "__main__":
    unittest.main()
