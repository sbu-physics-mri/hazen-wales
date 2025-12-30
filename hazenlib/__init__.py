"""
Welcome to the hazen Command Line Interface

The following Tasks are available:
- ACR phantom:
acr_all | acr_snr | acr_slice_position | acr_slice_thickness |
acr_spatial_resolution | acr_uniformity | acr_ghosting | acr_geometric_accuracy |
acr_low_contrast_object_detectability
- MagNET Test Objects:
snr | snr_map | slice_position | slice_width | spatial_resolution | uniformity | ghosting
- Caliber phantom:
relaxometry
"""

import argparse
import importlib
import inspect
import logging
import os
import sys

from hazenlib._version import __version__
from hazenlib.formatters import write_result
from hazenlib.logger import logger
from hazenlib.utils import get_dicom_files

"""Hazen is designed to measure the same parameters from multiple images.
    While some tasks require a set of multiple images (within the same folder),
    such as slice position, SNR and all ACR tasks,
    the majority of the calculations are performed on a single image at a time,
    and bulk processing all images in the input folder with the same task.

    In Sep 2023 a design decision was made to pass the minimum number of files
    to the task.run() functions.
    Below is a list of the single image tasks where the task.run() will be called
    on each image in the folder, while other tasks are being passed ALL image files.
"""
single_image_tasks = [
    "ghosting",
    "uniformity",
    "spatial_resolution",
    "slice_width",
    "snr_map",
]

# Mapping of task names to their corresponding modules
TASK_MAP = {
    # ACR phantom tasks
    "acr_geometric_accuracy": "acr_geometric_accuracy",
    "acr_ghosting": "acr_ghosting",
    "acr_low_contrast_object_detectability": "acr_low_contrast_object_detectability",
    "acr_slice_position": "acr_slice_position",
    "acr_slice_thickness": "acr_slice_thickness",
    "acr_snr": "acr_snr",
    "acr_spatial_resolution": "acr_spatial_resolution",
    "acr_uniformity": "acr_uniformity",
    # MagNET Test Objects tasks
    "ghosting": "ghosting",
    "slice_position": "slice_position",
    "slice_width": "slice_width",
    "snr": "snr",
    "snr_map": "snr_map",
    "spatial_resolution": "spatial_resolution",
    "uniformity": "uniformity",
    # Caliber phantom tasks
    "relaxometry": "relaxometry",
    # Special combined task
    "acr_all": "acr_all",
}


def init_task(selected_task, files, report, report_dir, **kwargs):
    """Initialise object of the correct HazenTask class

    Args:
        selected_task (string): name of task script/module to load
        files (list): list of filepaths to DICOM images
        report (bool): whether to generate report images
        report_dir (string): path to folder to save report images to
        kwargs: any other key word arguments

    Returns:
        an object of the specified HazenTask class
    """
    task_module = importlib.import_module(f"hazenlib.tasks.{selected_task}")

    try:
        task = getattr(task_module, selected_task.capitalize())(
            input_data=files, report=report, report_dir=report_dir, **kwargs
        )
    except AttributeError:
        class_list = [
            cls.__name__
            for _, cls in inspect.getmembers(
                sys.modules[task_module.__name__],
                lambda x: inspect.isclass(x) and (x.__module__ == task_module.__name__),
            )
        ]
        if len(class_list) == 1:
            task = getattr(task_module, class_list[0])(
                input_data=files, report=report, report_dir=report_dir, **kwargs
            )
        else:
            msg = (
                f"Task {task_module} has multiple class definitions:"
                " {class_list}"
            )
            logger.error(msg)
            raise Exception(msg)

    return task


def main():
    """Main entrypoint to hazen"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "task",
        choices=list(TASK_MAP.keys()),
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
            'Set the level of logging based on severity. '
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

    args = parser.parse_args()

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

    logger.debug("The following files were identified as valid DICOMs:")
    files = get_dicom_files(args.folder)
    logger.debug("%s task will be set off on %s images", args.task, len(files))

    # Parse the task and optional arguments:
    selected_task = args.task

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
        if args.calc is None or args.plate_number is None:
            parser.error("relaxometry task requires --calc and --plate_number arguments")
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
            for file in files:
                task = init_task(selected_task, [file], report, report_dir)
                result = task.run()
                result_string = result.to_json()
                print(result_string)
            return
        # Slice Position task, all ACR tasks except SNR
        # may be enhanced, may be multi-frame
        fns = [os.path.basename(fn) for fn in files]
        logger.info("Processing: %s", fns)
        if selected_task == "acr_all":
            selected_tasks = [
                task_name
                for task_name in TASK_MAP.keys()
                if task_name.startswith("acr") and task_name != "acr_all"
            ]
        else:
            selected_tasks = [selected_task]

        for selected_task in selected_tasks:
            task = init_task(
                selected_task, files, report, report_dir, verbose=verbose)
            result = task.run()
            write_result(result, fmt=fmt, path=result_file)
        return

    write_result(result, fmt=fmt, path=result_file)


if __name__ == "__main__":
    main()
