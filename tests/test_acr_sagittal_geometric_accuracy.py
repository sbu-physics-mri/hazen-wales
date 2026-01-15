import pathlib
import unittest

import numpy as np
from hazenlib.tasks.acr_sagittal_geometric_accuracy import (
    ACRSagittalGeometricAccuracy,
)
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSagittalGeometricAccuracySiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFitLocalizer")
    L1 = 149.41

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_geometric_accuracy_task = ACRSagittalGeometricAccuracy(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm_1 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[0]
        self.slice1_val = np.round(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(
                self.dcm_1
            ),
            2,
        )

    def test_geometric_accuracy_slice_1(self):
        print(
            "\ntest_geo_accuracy.py::TestACRSagittalGeometricAccuracy::test_geometric_accuracy_slice_1"
        )
        print("new_release:", self.slice1_val)
        print("fixed value:", self.L1)

        assert self.slice1_val == self.L1


class TestACRGeometricAccuracyPhilipsAchieva(
    TestACRSagittalGeometricAccuracySiemens
):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchievaLocalizer")
    L1 = 146.48
