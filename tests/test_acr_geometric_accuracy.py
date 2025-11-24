import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    L1 = 191.41, 187.5
    L5 = 191.41, 187.5, 191.41, 191.41
    distortion_metrics = [0.11, 2.5, 0.97]

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_geometric_accuracy_task = ACRGeometricAccuracy(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm_1 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[0]
        self.dcm_5 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[4]

    def test_geometric_accuracy_slice_1(self):
        slice1_vals = self.acr_geometric_accuracy_task.get_geometric_accuracy(0)

        slice1_vals = np.round(slice1_vals, 2)

        print("\ntest_geo_accuracy.py::TestGeoAccuracy::test_geo_accuracy_slice1")
        print("new_release:", slice1_vals)
        print("fixed value:", self.L1)

        assert (slice1_vals == self.L1).all() == True

    def test_geometric_accuracy_slice_5(self):
        slice5_vals = np.array(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(4)
        )

        slice5_vals = np.round(slice5_vals, 2)

        print("\ntest_geo_accuracy.py::TestGeoAccuracy::test_geo_accuracy_slice5")
        print("new_release:", slice5_vals)
        print("fixed value:", self.L5)
        assert (slice5_vals == self.L5).all() == True

    def test_distortion_metrics(self):
        metrics = np.array(
            self.acr_geometric_accuracy_task.get_distortion_metrics(self.L1 + self.L5)
        )
        metrics = np.round(metrics, 2)
        print(f'Distortion metrics => {metrics}')
        assert (metrics == self.distortion_metrics).all() == True

# TODO: Add unit tests for Philips datasets (when Philips data is available).
class TestACRGeometricAccuracyGE(TestACRGeometricAccuracySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    L1 = 190.93, 188.9
    L5 = 190.42, 189.41, 190.43, 189.41
    distortion_metrics = [-0.08, 1.1, 0.38]

class TestACRGeometricAccuracySiemensSolaFit(TestACRGeometricAccuracySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    L1 = 190.43, 190.43
    L5 = 189.45, 190.43, 191.41, 191.41
    distortion_metrics = [0.59, 1.41, 0.35]

class TestACRGeometricAccuracyPhilipsAchieva(TestACRGeometricAccuracySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    L1 = 190.43, 189.45
    L5 = 190.43, 189.45, 189.45, 189.45
    distortion_metrics = [-0.22, 0.55, 0.24]
