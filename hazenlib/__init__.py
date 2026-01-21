"""
Welcome to the hazen Command Line Interface

The following Tasks are available:
- ACR phantom:
acr_snr | acr_slice_position | acr_slice_thickness |
acr_spatial_resolution | acr_uniformity | acr_ghosting | acr_geometric_accuracy |
acr_low_contrast_object_detectability
- MagNET Test Objects:
snr | snr_map | slice_position | slice_width | spatial_resolution | uniformity | ghosting
- Caliber phantom:
relaxometry
"""

import argparse
import importlib
import logging
import os

from hazenlib._version import __version__
from hazenlib.formatters import write_result
from hazenlib.logger import logger
from hazenlib.types import PhantomType, TaskMetadata
from hazenlib.utils import get_dicom_files

TASK_REGISTRY = {
    # MagNET #
    "snr": TaskMetadata(
        module_name="snr",
        class_name="SNR",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "ghosting": TaskMetadata(
        module_name="ghosting",
        class_name="Ghosting",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "uniformity": TaskMetadata(
        module_name="uniformity",
        class_name="Uniformity",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "spatial_resolution": TaskMetadata(
        module_name="spatial_resolution",
        class_name="SpatialResolution",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "slice_width": TaskMetadata(
        module_name="slice_width",
        class_name="SliceWidth",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "slice_position": TaskMetadata(
        module_name="slice_position",
        class_name="SlicePosition",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "snr_map": TaskMetadata(
        module_name="snr_map",
        class_name="SNRMap",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    # ACR #
    "acr_geometric_accuracy": TaskMetadata(
        module_name="acr_geometric_accuracy",
        class_name="ACRGeometricAccuracy",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_ghosting": TaskMetadata(
        module_name="acr_ghosting",
        class_name="ACRGhosting",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_low_contrast_object_detectability": TaskMetadata(
        module_name="acr_low_contrast_object_detectability",
        class_name="ACRLowContrastObjectDetectability",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_object_detectability": TaskMetadata(
        module_name="acr_object_detectability",
        class_name="ACRObjectDetectability",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_slice_position": TaskMetadata(
        module_name="acr_slice_position",
        class_name="ACRSlicePosition",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_slice_thickness": TaskMetadata(
        module_name="acr_slice_thickness",
        class_name="ACRSliceThickness",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_snr": TaskMetadata(
        module_name="acr_snr",
        class_name="ACRSNR",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_spatial_resolution": TaskMetadata(
        module_name="acr_spatial_resolution",
        class_name="ACRSpatialResolution",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_sagittal_geometric_accuracy": TaskMetadata(
        module_name="acr_sagittal_geometric_accuracy",
        class_name="ACRSagittalGeometricAccuracy",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_uniformity": TaskMetadata(
        module_name="acr_uniformity",
        class_name="ACRUniformity",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    # Caliber
    "relaxometry": TaskMetadata(
        module_name="relaxometry",
        class_name="Relaxometry",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
}


def init_task(selected_task, files, report, report_dir, **kwargs):
    """Initialise object of the correct HazenTask class.

    Args:
        selected_task (string): name of task script/module to load
        files (list): list of filepaths to DICOM images
        report (bool): whether to generate report images
        report_dir (string): path to folder to save report images to
        kwargs: any other key word arguments

    Returns:
        an object of the specified HazenTask class

    """
    try:
        meta = TASK_REGISTRY[selected_task]
    except KeyError:
        msg = f"Unknown task: {selected_task}"
        logger.error(
            "%s. Supported tasks are:\n%s", "\n\t".join(TASK_REGISTRY)
        )
        raise ValueError(msg)

    # Import module
    task_module = importlib.import_module(f"hazenlib.tasks.{meta.module_name}")

    # Get explicit class
    try:
        task_class = getattr(task_module, meta.class_name)
    except AttributeError as err:
        raise ImportError(
            f"Module {meta.module_name} has no class '{meta.class_name}'"
        ) from err

    return task_class(
        input_data=files, report=report, report_dir=report_dir, **kwargs
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "task",
        choices=list(TASK_REGISTRY.keys()),
        help="The task to run",
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing DICOM files",
    )

    # General options available for all tasks
    parser.add_argument(
        "--report",
        action="store_true",
        help="Whether to generate visualisation of the measurement steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Provide a folder where report images are to be saved",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Whether to provide additional metadata about the calculation "
            "in the result (slice position and relaxometry tasks)"
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help=(
            "Set the level of logging based on severity. "
            'Available levels are "debug", "warning", "error", "critical", '
            'with "info" as default'
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "tsv"],
        help="Output format for test results. Choices: json (default), csv or tsv",
    )
    parser.add_argument(
        "--result",
        type=str,
        default="-",
        help='Path to the results path. If "-", default, will write to stdout',
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    # Task-specific options
    parser.add_argument(
        "--measured_slice_width",
        type=float,
        default=None,
        help=(
            "Provide a slice width to be used for SNR measurement, "
            "by default it is parsed from the DICOM "
            "(optional for acr_snr and snr)"
        ),
    )
    parser.add_argument(
        "--subtract",
        type=str,
        default=None,
        help=(
            "Provide a second folder path to calculate SNR by subtraction "
            "for the ACR phantom (optional for acr_snr)"
        ),
    )
    parser.add_argument(
        "--coil",
        type=str,
        default=None,
        choices=["head", "body"],
        help="Coil type for SNR measurement (optional for snr)",
    )
    parser.add_argument(
        "--calc",
        type=str,
        default=None,
        choices=["T1", "T2"],
        help=(
            "Choose 'T1' or 'T2' for relaxometry measurement "
            "(required for relaxometry)"
        ),
    )
    parser.add_argument(
        "--plate_number",
        type=int,
        default=None,
        choices=[4, 5],
        help="Which plate to use for measurement: 4 or 5 (required for relaxometry)",
    )
    return parser


def main():
    """Main entrypoint to hazen"""
    parser = get_parser()
    args = parser.parse_args()
    single_image_tasks = [
        task for task in TASK_REGISTRY.values() if task.single_image
    ]

    # Set common options
    log_levels = {
        "critical": logging.CRITICAL,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    level = log_levels.get(args.log, logging.INFO)
    logging.getLogger().setLevel(level)

    report = args.report
    report_dir = args.output
    verbose = args.verbose
    fmt = args.format
    result_file = args.result

    logger.info(f"Hazen version: {__version__}")
    logger.debug("The following files were identified as valid DICOMs:")
    files = get_dicom_files(args.folder)
    logger.debug("%s task will be set off on %s images", args.task, len(files))

    # Parse the task and optional arguments:
    selected_task = args.task.lower()

    if selected_task == "snr":
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            measured_slice_width=args.measured_slice_width,
            coil=args.coil,
        )
        result = task.run()
    elif selected_task == "acr_snr":
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            subtract=args.subtract,
            measured_slice_width=args.measured_slice_width,
        )
        result = task.run()
    elif selected_task == "relaxometry":
        missing_args = []
        if args.calc is None:
            missing_args.append("--calc")
        if args.plate_number is None:
            missing_args.append("--plate_number")
        if missing_args:
            parser.error(
                f"relaxometry task requires the following arguments: "
                f"{', '.join(missing_args)}"
            )
        task = init_task(selected_task, files, report, report_dir)
        result = task.run(
            calc=args.calc,
            plate_number=args.plate_number,
            verbose=verbose,
        )
    else:
        if selected_task in single_image_tasks:
            # Ghosting, Uniformity, Spatial resolution, SNR map, Slice width
            # for now these are most likely not enhanced, single-frame
            for f in files:
                task = init_task(selected_task, [f], report, report_dir)
                result = task.run()
                write_result(result, fmt=fmt, path=result_file)
            return
        # Slice Position task, all ACR tasks except SNR
        # may be enhanced, may be multi-frame
        fns = [os.path.basename(fn) for fn in files]
        logger.info("Processing: %s", fns)
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            verbose=verbose,
        )
        result = task.run()

        task = init_task(
            selected_task, files, report, report_dir, verbose=verbose
        )
        result = task.run()
        write_result(result, fmt=fmt, path=result_file)
        return

    write_result(result, fmt=fmt, path=result_file)


if __name__ == "__main__":
    main()
