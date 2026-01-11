"""Low-Contrast Object Detectability.

As per section 7 of the Large and Medium Phantom Test Guidance for the
ACR MRI Accreditation Program:

https://www.acraccreditation.org/-/media/ACRAccreditation/Documents/MRI/ACR-Large-Med-Phantom-Guidance-102022.pdf

```
the low contrast detectability test assesses the extent to which objects
of low contrast are discernible in the images.
```

These are performed on slices 8 through 11 by counting
the number of visible spokes.

The implementation follows that of:

A statistical approach to automated analysis of the
low-contrast object detectability test for the large ACR MRI phantom

DOI = {10.1002/acm2.70173}
journal = {Journal of Applied Clinical Medical Physics},
author = {Golestani, Ali M. and Gee, Julia M.},
year = {2025},
month = jul

An implementation by the authors can be found on GitHub:
    https://github.com/aligoles/ACR-Low-Contrast-Object-Detectability

With the original paper:
    https://doi.org/10.1002/acm2.70173

ACR Low Contrast Object Detectability:
    https://mriquestions.com/uploads/3/4/5/7/34572113/largephantomguidance.pdf

Notes from the paper:

- Images with acquired with:
        - 3.0T Siemens MAGNETOM Vida.
        - 1.5T Philips scanner integrated into an Elekta Unity MR-Linac System.
- 40 Datasets analyzed (20 for each scanner).


Implementation overview:

- Normalise image intensity for each slice (independently) to within [0, 1].
- Background removal process performed using histogram thresholding.
- Contrast disk is identified.
        - Detect center of phantom, crop and then find a large circle.
- 90 Angular radials profile in a specific angle are generated.
- Known phantom geometry and rotation used to calculate position of first spoke.
        - Circles at 12.5, 25.0 and 38.0mm from CoG.
- 2nd order polynomial fitted to the model and added to general linear model
    (GLM) regressors.
- 3 GLM regressors created for each 1D profile.
- Test passes if every GLM regressors exceed the significance level.
        - Significance level for each slice is set to 0.0125.
- Significance within each slice is adjusted using the Benjamini-Hochberg
    false discovery rate.

Implemented for Hazen by Alex Drysdale: alexander.drysdale@wales.nhs.uk
"""

# Typing
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

# Python imports
import copy
import logging
import traceback
from pathlib import Path
from types import MappingProxyType

# Module imports
import cv2
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage.transform
import statsmodels
import statsmodels.api as sm
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import (FailedStatsModel, LCODTemplate, Measurement,
                            P_HazenTask, Result, SpokeReportData,
                            StatsParameters)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)


class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom.

    Attributes:
        SLICE_ANGLE_OFFSET : Angular offset between each subsequent slice
                in radians (9 degrees converted to radians).
        START_ANGLE : Starting angle for slice 0 in radians
                (90 degrees converted to radians).

    """

    SLICE_ANGLE_OFFSET: float = 9
    START_ANGLE: float = 0
    LCOD_DISC_SIZE: float = 43  # mm

    BINARIZATION_THRESHOLD: MappingProxyType = MappingProxyType(
        {
            1.5: 97.8,
            3.0: 97.5,
        },
    )

    _DETREND_POLYNOMIAL_ORDER: int = 3
    _STD_TOL: float = 0.01

    _RADIAL_PROFILE_LENGTH: int = 128
    _ALPHA: float = 0.05
    _OPTIMIZER: str = "TBPSA"
    _BUDGET: int = 100

    NLOPT_METHOD: str = "Nelder-Mead"

    OBJECT_RIBBON_COLORS = ("#1E88E5", "#FFC107", "#004D40")

    def __init__(
        self,
        alpha: float | None = None,
        **kwargs: P_HazenTask.kwargs,
    ) -> None:
        """Initialise the LCOD object."""
        # TODO(abdrysdale) : Validate and remove this warning.
        logger.warning(
            "The ACR Low Contrast Object Detectability test has not been"
            " calibrated. Do not use these values for actual QA!",
        )
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)

        self.alpha = self._ALPHA if alpha is None else alpha

        # Start at last slice (highest contrast) and work backwards
        self.slice_range = slice(10, 6, -1)

        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self.rotation = self.ACR_obj.determine_rotation(
            self.ACR_obj.slice_stack[0].pixel_array,
        )
        self.lcod_center = None

        # Pass threshold is at least N spokes total for both the T1 and T2
        # acquisitions where:
        # @ 1.5T, N =  7
        # @ 3.0T, N = 37
        match float(
            self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value,
        ):
            case 3.0:
                self.pass_threshold = 37
            case 1.5:
                self.pass_threshold = 7
            case _:
                logger.error(
                    "No LCOD pass threshold specified for %s T systems"
                    " assuming a pass threshold of at least 7 spokes for"
                    " each sequence",
                    self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value,
                )
                self.pass_threshold = 7

        # Only used in reporting
        self.slice_report_data = {}

    def run(self) -> Result:
        """Run the LCOD analysis."""
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())

        # TODO(abdrysdale) : Validate and remove this description
        results.desc = f"{results.desc} - INVALID!"

        results.files = [
            self.img_desc(f)
            for f in self.ACR_obj.slice_stack[self.slice_range]
        ]

        total_spokes = 0
        # TODO(abdrysdale) : Use wait_on_parallel_results to collect results.
        for i, dcm in enumerate(self.ACR_obj.slice_stack[self.slice_range]):
            slice_no = 1 + self.slice_range.step * i + self.slice_range.start
            result = self.count_spokes(
                dcm,
                slice_no=slice_no,
                alpha=self.alpha,
            )
            try:
                num_spokes = min(i for i, r in enumerate(result) if not r)
            except ValueError:
                num_spokes = len(result)

            # Add individual spoke measurements for debugging
            # and further analysis
            # If this results in log flooding then it might be best to remove
            for j, r in enumerate(result):
                spoke_no = j + 1
                results.add_measurement(
                    Measurement(
                        name="LowContrastObjectDetectability",
                        type="measured",
                        subtype=f"slice {slice_no} spoke {spoke_no}",
                        value=r,
                    ),
                )
            total_spokes += num_spokes
            results.add_measurement(
                Measurement(
                    name="LowContrastObjectDetectability",
                    type="measured",
                    subtype=f"slice {slice_no}",
                    value=num_spokes,
                ),
            )

        results.add_measurement(
            Measurement(
                name="LowContrastObjectDetectability",
                type="measured",
                subtype="total",
                value=total_spokes,
            ),
        )

        results.add_measurement(
            Measurement(
                name="LowContrastObjectDetectability",
                type="measured",
                subtype="pass/fail",
                value=total_spokes >= self.pass_threshold,
            ),
        )

        if self.report:
            results.add_report_image(self.report_files)

        return results

    def _get_params_and_p_vals(
        self,
        template: LCODTemplate,
        dcm: pydicom.Dataset,
    ) -> StatsParameters:
        """Get a list of parameters and associated p-values."""
        sp = StatsParameters()
        for spoke in template.spokes:
            profile, object_mask = spoke.profile(
                dcm,
                size=self._RADIAL_PROFILE_LENGTH,
                return_object_mask=True,
            )

            p_vals, params = self._analyze_profile(
                profile,
                object_mask,
            )
            sp.p_vals.append(p_vals)
            sp.params.append(params)

        return sp

    def _fdrcorrection(
        self,
        sp: StatsParameters,
        alpha: float,
    ) -> tuple[np.ndarray]:
        p_vals_fdr = statsmodels.stats.multitest.fdrcorrection(
            sp.p_vals_all,
            alpha=alpha,
            method="indep",
            is_sorted=False,
        )[0].reshape(-1, len(sp.p_vals[-1]))
        params_fdr = np.array(sp.params_all).reshape(-1, len(sp.params[-1]))

        return (p_vals_fdr, params_fdr)

    def count_spokes(
        self,
        raw: pydicom.Dataset,
        slice_no: int = -1,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Count spokes with optional report data capture."""
        dcm = self._preprocess(raw)
        template = self.get_current_slice_template(slice_no)
        spokes = template.spokes

        # Get analysis data
        sp = StatsParameters()
        report_data = [] if self.report else None

        for spoke_id, spoke in enumerate(spokes):
            profile, (x_coords, y_coords), object_mask = spoke.profile(
                dcm,
                size=self._RADIAL_PROFILE_LENGTH,
                return_coords=True,
                return_object_mask=True,
            )

            # Analyze profile
            if self.report:
                p_vals, params, detrended, trend = self._analyze_profile(
                    profile,
                    object_mask,
                    return_intermediate=True,
                )
                # Store data for reporting
                report_data.append(
                    SpokeReportData(
                        spoke_id=spoke_id,
                        profile=profile,
                        detrended=detrended,
                        trend=trend,
                        object_mask=object_mask,
                        x_coords=x_coords,
                        y_coords=y_coords,
                        p_vals=p_vals,
                        params=params,
                        detected=[],  # Will be filled after FDR correction
                        objects=spoke.objects,
                    ),
                )
            else:
                p_vals, params = self._analyze_profile(profile, object_mask)

            sp.p_vals.append(p_vals)
            sp.params.append(params)

        # FDR correction
        p_vals_fdr, params_fdr = self._fdrcorrection(sp, alpha=alpha)

        # Update detection status
        for spoke_number, spoke in enumerate(spokes):
            for i, obj in enumerate(spoke):
                obj.detected = (
                    p_vals_fdr[spoke_number, i]
                    and params_fdr[spoke_number, i] > 0
                )
            spoke.passed = all(obj.detected for obj in spoke)

            # Update report data with detection status
            if self.report:
                report_data[spoke_number].detected = [
                    obj.detected for obj in spoke
                ]
                report_data[spoke_number].p_vals = sp.p_vals[spoke_number]
                report_data[spoke_number].params = sp.params[spoke_number]

        # Store report data if reporting enabled
        if self.report:
            self.slice_report_data[slice_no] = report_data
            self.generate_slice_report(dcm, slice_no, report_data)

        return [s.passed for s in spokes]

    def _improve_template_with_optimiser(
        self,
        template: LCODTemplate,
        current_slice: int,     # ACR slice notation not index (i.e. 11 not 10)
        *,
        spokes: list[int] | tuple[int] = (0,),
        center_search_tol: float = 2,  # in mm
        theta_tol: float = 3,
    ) -> LCODTemplate:
        """Improve the template with a non-linear optimiser."""
        dcm = self.ACR_obj.slice_stack[current_slice - 1]

        selected_spokes = [
            spoke for idx, spoke in enumerate(template.spokes) if idx in spokes
        ] if spokes else [0]

        # Collect profiles, coords, and object masks for debug plotting
        if self.report:
            profiles, coords, object_masks = zip(
                *[
                    spoke.profile(
                        dcm,
                        size=self._RADIAL_PROFILE_LENGTH,
                        return_coords=True,
                        return_object_mask=True,
                    )
                    for spoke in selected_spokes
                ],
                strict=True,
            )
            spoke_ids = list(spokes)
        else:
            profiles, coords = zip(
                *[
                    spoke.profile(
                        dcm,
                        size=self._RADIAL_PROFILE_LENGTH,
                        return_coords=True,
                    )
                    for spoke in selected_spokes
                ],
                strict=True,
            )
            object_masks = None
            spoke_ids = None

        intersection_points: tuple[tuple[float, float], ...] = (
            self._get_intersection_points(
                profiles,
                coords,
                object_masks=object_masks,
                spoke_ids=spoke_ids,
                slice_no=current_slice,
            )
        )

        distances = [
            d
            for spoke in selected_spokes
            for idx, d in enumerate(spoke.dist)
            for _ in range(2)   # Two intersection points per object.
            if all(
                p is not None for p in intersection_points[2 * idx: 2 * idx + 2]
            )
        ]
        radii = [
            spoke.diameter / 2
            for spoke in selected_spokes
            for _ in intersection_points
        ]

        complete_intersection_points = [
            p
            for idx, p in enumerate(intersection_points)
            if (
                p is not None
                and intersection_points[
                    idx + int(1 - (idx % 2) / 0.5)  # idx + 1 or idx - 1
                ] is not None
            )
        ]

        def minimiser(vec: np.ndarray) -> float:
            cx = vec[0]
            cy = vec[1]
            theta = np.deg2rad(vec[2])

            return sum(
                (
                    (xi - cx - di * np.sin(theta)) ** 2
                    + (yi - cy - di * np.cos(theta)) ** 2
                    - ri ** 2
                )
                for (xi, yi), di, ri in zip(
                    complete_intersection_points, distances, radii,
                    strict=True,
                )
            )

        c_tol = center_search_tol / self.ACR_obj.dx
        res = sp.optimize.minimize(
            minimiser,
            [template.cx, template.cy, template.theta],
            method=self.NLOPT_METHOD,
            bounds=(
                [template.cx - c_tol, template.cx + c_tol],
                [template.cy - c_tol, template.cy + c_tol],
                [template.theta - theta_tol, template.theta + theta_tol],
            ),
        )
        cx, cy, theta = res.x

        return LCODTemplate(cx, cy, theta)

    def find_center(self, crop_ratio: float = 0.55) -> tuple[float]:
        """Find the center of the LCOD phantom."""
        if self.lcod_center is not None:
            return self.lcod_center

        # Get ACR Phantom Center
        dcm = self.ACR_obj.slice_stack[0]
        (main_cx, main_cy), main_radius = self.ACR_obj.find_phantom_center(
            dcm.pixel_array,
            self.ACR_obj.dx,
            self.ACR_obj.dy,
        )

        # Get cropped image of LCOD disk
        dcm = self.ACR_obj.slice_stack[-1]
        r = main_radius * crop_ratio
        lcod_cy = main_cy + 5 / self.ACR_obj.dy

        offset_y = max(0, int(lcod_cy - r))
        offset_x = max(0, int(main_cx - r))
        cropped_image = dcm.pixel_array[
            offset_y : int(lcod_cy + r + 1),
            offset_x : int(main_cx + r + 1),
        ]
        cropped_image = (
            (cropped_image - cropped_image.min())
            * 255.0
            / (cropped_image.max() - cropped_image.min())
        ).astype(np.uint8)

        # Pre-processing for circle detection
        img_blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)
        img_grad = img_blur.max() - img_blur

        lcod_r_init = self.LCOD_DISC_SIZE  # mm
        try:
            detected_circles = cv2.HoughCircles(
                img_grad,
                method=cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=cropped_image.shape[0] // 2,
                minRadius=int((lcod_r_init - 2) / self.ACR_obj.dy),
                maxRadius=int((lcod_r_init + 2) / self.ACR_obj.dy),
            ).flatten()
        except AttributeError:
            logger.warning("Failed to find LCOD center, using defaults.")
            detected_circles = (main_cx, lcod_cy, lcod_r_init)
            lcod_center = detected_circles[:2]

        else:
            lcod_center = tuple(
                (dc + offset) * dv
                for dc, offset, dv in zip(
                    detected_circles[:2],
                    (offset_x, offset_y),
                    (self.ACR_obj.dx, self.ACR_obj.dy),
                    strict=True,
                )
            )

        self.lcod_center = lcod_center
        lcod_r = detected_circles[2]

        if self.report:
            fig, axes = plt.subplots(2, 2, constrained_layout=True)

            # Initial Estimate
            axes[0, 0].imshow(dcm.pixel_array, cmap="gray")
            axes[0, 0].scatter(
                main_cx / self.ACR_obj.dx,
                lcod_cy / self.ACR_obj.dy,
                marker="x",
                color="red",
            )
            circle = Circle(
                (
                    main_cx / self.ACR_obj.dx,
                    lcod_cy / self.ACR_obj.dy,
                ),
                lcod_r_init / self.ACR_obj.dx,
                fill=False,
                edgecolor="red",
                linewidth=0.5,
            )
            axes[0, 0].add_patch(circle)
            axes[0, 0].set_title("Initial Estimate")

            # Cropped
            axes[0, 1].imshow(cropped_image)
            axes[0, 1].scatter(
                main_cx / self.ACR_obj.dx - offset_x,
                lcod_cy / self.ACR_obj.dy - offset_y,
                marker="x",
                color="red",
            )
            circle = Circle(
                (
                    main_cx / self.ACR_obj.dx - offset_x,
                    lcod_cy / self.ACR_obj.dy - offset_y,
                ),
                lcod_r_init / self.ACR_obj.dx,
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
            axes[0, 1].add_patch(circle)
            axes[0, 1].set_title("Cropped Image")

            # Initial guess
            cx, cy = detected_circles[:2]
            circle = Circle(
                (cx / self.ACR_obj.dx, cy / self.ACR_obj.dy),
                lcod_r / self.ACR_obj.dx,
                fill=False,
                edgecolor="blue",
                linewidth=2,
            )
            axes[1, 0].imshow(cropped_image, cmap="gray")
            axes[1, 0].scatter(
                detected_circles[0],
                detected_circles[1],
                marker="x",
                color="blue",
            )
            axes[1, 0].add_patch(circle)
            axes[1, 0].set_title("Detected Circle")

            # Final guess
            cx = self.lcod_center[0] / self.ACR_obj.dx - offset_x
            cy = self.lcod_center[1] / self.ACR_obj.dy - offset_y
            circle = Circle(
                (cx, cy),
                lcod_r / self.ACR_obj.dx,
                fill=False,
                edgecolor="blue",
                linewidth=2,
            )
            axes[1, 1].imshow(cropped_image, cmap="gray")
            axes[1, 1].scatter(
                cx,
                cy,
                marker="x",
                color="blue",
            )
            axes[1, 1].add_patch(circle)
            axes[1, 1].set_title(
                "Detected Circle",
            )

            fig.suptitle("LCOD Center Detection")

            data_path = Path(self.dcm_list[0].filename).parent.name
            img_path = (
                Path(self.report_path)
                / f"{data_path}_center_{self.img_desc(dcm)}.png"
            )
            fig.savefig(img_path)
            plt.close()
            self.report_files.append(img_path)

        return lcod_center

    def _current_slice_rotation(self, current_slice: int) -> float:
        rotation_offset = self.START_ANGLE + self.SLICE_ANGLE_OFFSET * abs(
            current_slice - 8,
        )
        return self.rotation + rotation_offset

    def get_current_slice_template(
        self,
        current_slice: int,
    ) -> LCODTemplate:
        """Find the position of the spokes within the LCOD disk."""
        # Rotation offset
        theta = self._current_slice_rotation(current_slice)

        if self.lcod_center is None:
            cx_0, cy_0 = self.find_center()  # updates lcod_center
        else:
            cx_0, cy_0 = self.lcod_center

        template = self._improve_template_with_optimiser(
            LCODTemplate(cx_0, cy_0, theta),
            current_slice,
        )

        logger.info(
            "Template generated for slice %i:"
            "\nCenter:\t(%f, %f)"
            "\nRotation:\t%f (initial: %f + offset: %f)",
            current_slice,
            template.cx,
            template.cy,
            template.theta,
            self.rotation,
            template.theta - self.rotation,
        )

        return template

    def _analyze_profile(
        self,
        profile: np.ndarray,
        object_mask: np.ndarray,
        *,
        mask_padding: int = 5,
        return_intermediate: bool = False,
    ) -> tuple:
        """Analyze radial profile with optional intermediate returns."""
        # Apply binary dilation (mask padding)
        footprint_n = mask_padding * 2 + 1
        object_mask = skimage.morphology.binary_dilation(
            object_mask,
            footprint=np.array(
                [[0] * footprint_n, [1] * footprint_n, [0] * footprint_n],
            ).T,
        )

        # De-trend with robust polynomial fitting
        if np.std(profile) > self._STD_TOL:
            x = np.linspace(0, 1, len(profile))
            coeffs = np.polyfit(x, profile, self._DETREND_POLYNOMIAL_ORDER)
            trend = np.polyval(coeffs, x)
        else:
            trend = np.zeros_like(profile)

        detrended = profile - trend
        kernel = np.ones(3) / 3
        smoothed = np.convolve(
            detrended - np.mean(detrended),
            kernel,
            mode="same",
        ).reshape((profile.size, 1))

        # Prepare GLM
        data = np.column_stack((object_mask, np.ones_like(profile)))

        # Fit GLM
        try:
            model = sm.GLM(smoothed, data).fit()
        except ValueError:
            logger.exception("Fit could not be obtained - failing detection")
            model = FailedStatsModel()

        if return_intermediate:
            return model.pvalues[:3], model.params[:3], detrended, trend
        return model.pvalues[:3], model.params[:3]

    # Add the report generation method
    def generate_slice_report(
        self,
        dcm: pydicom.Dataset,
        slice_no: int,
        report_data: list[SpokeReportData],
    ) -> None:
        """Generate comprehensive report for a single slice."""
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.2, wspace=0.2)

        # Main image (top-left, spans 1 row and 1 column)
        ax_main = fig.add_subplot(gs[0, 0])
        self._plot_main_image(ax_main, dcm, report_data, slice_no)

        # Profile plots for each spoke
        n_spokes = len(report_data)
        for i, spoke_data in enumerate(report_data):
            if i > n_spokes:  # Limit to available grid space (11 slots max)
                break

            # Calculate grid position (skip main image at [0,0])
            row = (i + 1) // 4
            col = (i + 1) % 4

            # Create broken axis within this grid cell
            self._create_broken_axis_plot(fig, gs[row, col], spoke_data)

        # Summary table (bottom row, last column)
        ax_table = fig.add_subplot(gs[2, 3])
        self._plot_summary_table(ax_table, report_data)

        legend_elements = [
            mpatches.Patch(
                facecolor=self.OBJECT_RIBBON_COLORS[0],
                alpha=0.4,
                label="Inner Object",
            ),
            mpatches.Patch(
                facecolor=self.OBJECT_RIBBON_COLORS[1],
                alpha=0.4,
                label="Middle Object",
            ),
            mpatches.Patch(
                facecolor=self.OBJECT_RIBBON_COLORS[2],
                alpha=0.4,
                label="Outer Object",
            ),
        ]
        ax_main.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=6,
            framealpha=0.9,
        )

        # Save figure
        data_path = Path(self.dcm_list[0].filename).parent.name
        img_path = (
            Path(self.report_path)
            / f"{data_path}_lcod_slice_{str(slice_no).zfill(2)}.png"
        )
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.report_files.append(img_path)

    def _plot_main_image(
        self,
        ax: plt.Axes,
        dcm: pydicom.Dataset,
        report_data: list[SpokeReportData],
        slice_no: int,
    ) -> None:
        """Plot cropped image focused on LCOD object region."""
        px_x, px_y = self.ACR_obj.dx, self.ACR_obj.dy

        # Calculate bounds of all objects across all spokes
        all_x_coords = [
            obj.x for spoke_data in report_data for obj in spoke_data.objects
        ]
        all_y_coords = [
            obj.y for spoke_data in report_data for obj in spoke_data.objects
        ]

        # Determine crop region with 10% margin
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        margin_x = (max_x - min_x) * 0.1
        margin_y = (max_y - min_y) * 0.1

        # Convert to pixel coordinates
        x_min = int(max(0, (min_x - margin_x) / px_x))
        x_max = int(min(dcm.pixel_array.shape[1], (max_x + margin_x) / px_x))
        y_min = int(max(0, (min_y - margin_y) / px_y))
        y_max = int(min(dcm.pixel_array.shape[0], (max_y + margin_y) / px_y))

        # Crop image
        cropped_img = dcm.pixel_array[y_min:y_max, x_min:x_max]

        vmin, vmax = self._window(dcm)
        # Display cropped region
        ax.imshow(
            cropped_img,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        # Adjust drawing coordinates for crop offset
        offset_x = x_min * px_x
        offset_y = y_min * px_y

        # Draw overlays
        for spoke_data in report_data:
            if (
                spoke_data.x_coords is not None
                and spoke_data.y_coords is not None
            ):
                ax.plot(
                    (spoke_data.x_coords - x_min),
                    (spoke_data.y_coords - y_min),
                    "y-",
                    linewidth=1.5,
                    alpha=0.8,
                )

            for obj, detected in zip(
                spoke_data.objects, spoke_data.detected, strict=True,
            ):
                color = "green" if detected else "red"
                circle = Circle(
                    ((obj.x - offset_x) / px_x, (obj.y - offset_y) / px_y),
                    (obj.diameter / 2) / px_x,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.5,
                )
                ax.add_patch(circle)

        ax.set_title(f"Slice {slice_no}", fontsize=10, fontweight="bold")
        ax.axis("off")

        # Add legend
        detected_patch = mpatches.Patch(color="green", label="Detected")
        not_detected_patch = mpatches.Patch(color="red", label="Not Detected")
        ax.legend(
            handles=[detected_patch, not_detected_patch],
            loc="upper right",
            fontsize=6,
        )

    def _create_broken_axis_plot(
        self,
        fig: plt.Figure,
        gs_cell: mpl.gridspec.SubplotSpec,
        spoke_data: SpokeReportData,
    ) -> None:
        """Create broken axis plot with prominent object ribbons."""
        cell_bbox = gs_cell.get_position(fig)
        top_height = cell_bbox.height * 0.5
        bottom_height = cell_bbox.height * 0.5

        ax_top = fig.add_axes(
            [
                cell_bbox.x0,
                cell_bbox.y0 + bottom_height,
                cell_bbox.width,
                top_height,
            ],
        )
        ax_bottom = fig.add_axes(
            [cell_bbox.x0, cell_bbox.y0, cell_bbox.width, bottom_height],
            sharex=ax_top,
        )

        x = np.arange(len(spoke_data.profile))

        # Plot profiles
        ax_top.plot(
            x,
            spoke_data.profile,
            "k-",
            label="Original",
            linewidth=1,
            zorder=3,
        )
        ax_top.plot(
            x,
            spoke_data.trend,
            "g--",
            label="Trend",
            linewidth=0.8,
            zorder=3,
        )
        ax_bottom.plot(x, spoke_data.detrended, "k-", linewidth=1, zorder=3)

        # Set y-limits
        orig_min, orig_max = (
            np.min(spoke_data.profile),
            np.max(spoke_data.profile),
        )
        det_min, det_max = (
            np.min(spoke_data.detrended),
            np.max(spoke_data.detrended),
        )
        orig_range = orig_max - orig_min
        det_range = det_max - det_min
        ax_top.set_ylim(
            orig_min - 0.05 * orig_range,
            orig_max + 0.05 * orig_range,
        )
        ax_bottom.set_ylim(
            det_min - 0.05 * det_range,
            det_max + 0.05 * det_range,
        )

        # Style axes
        ax_top.spines["bottom"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False, labelsize=6)
        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.xaxis.tick_bottom()
        ax_bottom.tick_params(labelbottom=True, labelsize=6)

        # Diagonal break markers
        d = 0.5
        kwargs = {
            "marker": [(-1, -d), (1, d)],
            "markersize": 8,
            "linestyle": "none",
            "color": "k",
            "mec": "k",
            "mew": 1,
            "clip_on": False,
        }
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

        # **Enhanced vertical ribbons for objects**
        n_objects = spoke_data.object_mask.shape[1]
        for obj_idx in range(n_objects):
            obj_mask = spoke_data.object_mask[
                :,
                obj_idx,
            ]  # Get mask for this object
            color = self.OBJECT_RIBBON_COLORS[obj_idx]

            obj_indices = np.where(obj_mask)[0]
            if len(obj_indices) > 0:
                # Expand region for prominence (20% padding)
                pad = max(1, len(obj_indices) // 5)
                start = max(0, obj_indices[0] - pad)
                end = min(len(x), obj_indices[-1] + pad)

                # Draw prominent vertical ribbons (behind profile lines)
                ax_top.axvspan(start, end, alpha=0.45, color=color, zorder=0)
                ax_bottom.axvspan(
                    start,
                    end,
                    alpha=0.45,
                    color=color,
                    zorder=0,
                )

                # Add subtle edge
                ax_top.axvline(
                    start,
                    color=color,
                    alpha=0.7,
                    linewidth=0.5,
                    zorder=1,
                )
                ax_top.axvline(
                    end,
                    color=color,
                    alpha=0.7,
                    linewidth=0.5,
                    zorder=1,
                )
                ax_bottom.axvline(
                    start,
                    color=color,
                    alpha=0.7,
                    linewidth=0.5,
                    zorder=1,
                )
                ax_bottom.axvline(
                    end,
                    color=color,
                    alpha=0.7,
                    linewidth=0.5,
                    zorder=1,
                )

                # Annotate p-value at object center
                mid_x = int(np.mean(obj_indices))
                mid_y = spoke_data.detrended[mid_x] + spoke_data.trend[mid_x]
                ax_bottom.annotate(
                    f"p={spoke_data.p_vals[obj_idx]:.3f}",
                    xy=(mid_x, mid_y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=5,
                    ha="left",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "yellow",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                    zorder=4,
                )

        # Labels
        ax_bottom.set_xlabel("Profile Position", fontsize=7)
        ax_top.set_ylabel("Original", fontsize=7)
        ax_bottom.set_ylabel("Detrended", fontsize=7)

        # Title
        status = "PASS" if all(spoke_data.detected) else "FAIL"
        n_detected = sum(spoke_data.detected)
        ax_top.set_title(
            f"Spoke {spoke_data.spoke_id + 1}:"
            f" {n_detected}/{len(spoke_data.detected)} {status}",
            fontsize=8,
            fontweight="bold",
        )

        ax_top.grid(visible=True, alpha=0.3)
        ax_bottom.grid(visible=True, alpha=0.3)

    def _plot_summary_table(
        self,
        ax: plt.Axes,
        report_data: list[SpokeReportData],
    ) -> None:
        """Plot condensed summary table (one row per spoke)."""
        ax.axis("off")

        # One row per spoke, objects shown as columns
        table_data = []
        headers = [
            "Spoke",
            "O1 p-val",
            "O1 param",
            "O2 p-val",
            "O2 param",
            "O3 p-val",
            "O3 param",
        ]

        for spoke_data in report_data:
            row = [f"{spoke_data.spoke_id + 1}"]
            for p_val, param in zip(
                spoke_data.p_vals,
                spoke_data.params,
                strict=True,
            ):
                # Compact formatting
                p_str = (
                    f"{p_val:.2e}"
                    if p_val < 1e-3  # noqa: PLR2004
                    else f"{p_val:.4f}"
                )
                param_str = f"{param:.3f}"
                row.extend([p_str, param_str])

            table_data.append(row)

        # Create table
        cw = 0.16
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=[0.10, cw, cw, cw, cw, cw, cw],
        )

        # Style
        table.auto_set_font_size(False)  # noqa: FBT003
        table.set_fontsize(6)  # Smaller font for compactness
        table.scale(1, 1.8)

        # Header styling
        for i, _ in enumerate(headers):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color cells: light green for pass, light red for fail
        f_color = "#FAADAD"
        p_color = "#C6E0B4"
        for i, spoke_data in enumerate(report_data):
            for j in range(3):  # For each of 3 objects
                is_fail = spoke_data.p_vals[j] > self._ALPHA
                color = f_color if is_fail else p_color
                table[(i + 1, j * 2 + 1)].set_facecolor(color)  # p-value cell

                is_fail = spoke_data.params[j] <= 0
                color = f_color if is_fail else p_color
                table[(i + 1, j * 2 + 2)].set_facecolor(
                    color,
                )  # parameter cell

    def _window(
        self,
        dcm: pydicom.FileDataset,
        idx: int | None = None,
        squeeze: float = 2,
    ) -> tuple[float]:
        """Return vmin, vmax values based on simple window method."""
        (cx, cy) = self.lcod_center
        r = (self.LCOD_DISC_SIZE - squeeze) * self.ACR_obj.dx
        y_grid, x_grid = np.meshgrid(
            np.arange(0, dcm.pixel_array.shape[0]),
            np.arange(0, dcm.pixel_array.shape[1]),
        )

        mask = (y_grid - cy) ** 2 + (x_grid - cx) ** 2 <= r**2

        try:
            vdata = (
                dcm.pixel_array * mask[None, :, :]
                if idx is None
                else dcm.pixel_array[idx, :, :] * mask
            )
        except IndexError:
            vdata = dcm.pixel_array * mask

        return (np.min(vdata[vdata != 0]), np.max(vdata))

    @staticmethod
    def _preprocess(
        dcm: pydicom.FileDataset,
        threshold: tuple[float] = (0.05, 0.65),
        threshold_step: float = 0.001,
        bounds: tuple[float] = (0.1, 0.2),
    ) -> pydicom.FileDataset:
        """Preprocess the DICOM."""
        processed = copy.deepcopy(dcm)
        data = processed.pixel_array

        # Normalise
        fdata = data / np.max(data)

        # Threshold
        threshold_min, threshold_max = threshold
        lower, upper = bounds
        structure = np.ones((3, 3), dtype=int)
        for thr in np.arange(threshold_min, threshold_max, threshold_step):
            _, thresh = cv2.threshold(fdata, thr, 1, 0)
            labelled, _ = sp.ndimage.label(
                thresh,
                structure,
            )
            thresh_inner = labelled == np.max(labelled)
            if lower < np.sum(thresh_inner != 0) / np.sum(fdata != 0) < upper:
                break

        # Erode the circular mask
        data *= skimage.morphology.erosion(
            thresh_inner,
        )

        processed.set_pixel_data(
            data,
            dcm[(0x0028, 0x0004)].value,  # Photometric Interpretation
            dcm[(0x0028, 0x0101)].value,  # Bits Stored
        )
        return processed

    def _get_intersection_points(
        self,
        profiles: list[np.ndarray],
        coords: list[tuple[np.ndarray, np.ndarray]],
        object_masks: list[np.ndarray] | None = None,
        spoke_ids: list[int] | None = None,
        slice_no: int | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Extract half-maximum intersection points from radial profiles.

        This method processes radial intensity profiles to locate object
        boundaries at half-maximum intensity (FWHM-like). These points can be
        used to construct the loss function L(c_x, c_y, theta_c) described in
        GitHub issue #56 for optimizing the LCOD template parameters.

        For each 1D profile (containing 3 Gaussian-like peaks from the
        low-contrast objects q1, q2, q3), the method finds the left and right
        points where the intensity equals half of each peak's maximum. This
        provides robust boundary detection that is less sensitive to noise and
        intensity variations than using peak centers alone.

        Args:
            profiles: List of 1D intensity profiles along radial lines P
            coords: List of (x_coords, y_coords) arrays for each profile point
            object_masks: Optional list of object masks for debug plotting
            spoke_ids: Optional list of spoke IDs for debug plotting
            slice_no: Optional slice number for debug plotting

        Returns:
            Tuple of (x, y) intersection points. Each profile contributes
            up to 6 points (2 per peak), giving N_profiles x 6 total points.

        """
        all_points = []

        for profile_idx, (profile, (x_coords, y_coords)) in enumerate(
            zip(profiles, coords, strict=True),
        ):
            try:
                # Validate coordinate dimensions
                if (
                    len(profile) != len(x_coords)
                    or len(profile) != len(y_coords)
                ):
                    logger.warning(
                        "Profile %i: coordinate length mismatch. "
                        "Profile: %i, X: %i, Y: %i",
                        profile_idx,
                        len(profile),
                        len(x_coords),
                        len(y_coords),
                    )
                    continue

                # De-trend profile to remove baseline variations
                detrended = self._detrend_profile(profile)

                # Apply light smoothing to reduce noise sensitivity
                kernel = np.ones(3) / 3
                smoothed = np.convolve(detrended, kernel, mode="same")

                # Find the three main peaks by prominence
                peak_indices = self._find_peaks(smoothed, num_peaks=3)

                if len(peak_indices) != 3:
                    logger.warning(
                        "Profile %i: found %i peaks, expected 3."
                        " Skipping profile.",
                        profile_idx,
                        len(peak_indices),
                    )
                    continue

                # Collect intersection points for this profile
                profile_intersection_points = []
                for peak_idx in peak_indices:
                    # Get half-maximum points for this peak
                    points = self._get_fwhm_points(
                        smoothed, peak_idx, x_coords, y_coords,
                    )

                    if len(points) < 2:
                        logger.debug(
                            "Profile %i, peak %i: "
                            "Found only %i edges",
                            profile_idx,
                            peak_idx,
                            len(points),
                        )
                        profile_intersection_points.extend([None, None])
                    else:
                        profile_intersection_points.extend(points)

                all_points.extend(profile_intersection_points)

                # Generate debug plot if reporting is enabled
                if (
                    self.report
                    and object_masks is not None
                    and spoke_ids is not None
                    and slice_no is not None
                    and profile_idx < len(object_masks)
                    and profile_idx < len(spoke_ids)
                ):
                    self._plot_profile_debug(
                        smoothed,
                        peak_indices,
                        profile_intersection_points,
                        spoke_ids[profile_idx],
                        slice_no,
                        object_masks[profile_idx],
                        x_coords,
                        y_coords,
                    )

            except Exception as e:
                logger.error(
                    f"Error processing profile {profile_idx}: {e}. "
                    f"Traceback: {traceback.format_exc()}"
                )
                continue

        return tuple(all_points)

    def _detrend_profile(self, profile: np.ndarray) -> np.ndarray:
        """Remove polynomial trend from profile using robust fitting."""
        if np.std(profile) > self._STD_TOL:
            x = np.linspace(0, 1, len(profile))
            coeffs = np.polyfit(x, profile, self._DETREND_POLYNOMIAL_ORDER)
            trend = np.polyval(coeffs, x)
            return profile - trend
        return profile.copy()

    def _find_peaks(
        self,
        profile: np.ndarray,
        num_peaks: int = 3,
    ) -> np.ndarray:
        """Find indices of top N peaks by prominence.

        Uses scipy.signal.find_peaks with prominence-based filtering to
        identify the most significant peaks, then selects the top N.
        """
        # Find peaks with prominence and distance constraints
        peak_indices, properties = sp.signal.find_peaks(
            profile,
            prominence=0.05 * np.ptp(profile),  # 5% of full range
            distance=len(profile) // (num_peaks * 2),  # Enforce separation
            width=1,  # Minimum width of 1 sample
        )

        if len(peak_indices) < num_peaks:
            logger.debug(
                f"Found only {len(peak_indices)} peaks with prominence filter. "
                "Falling back to highest points."
            )
            # Fallback: use highest points, ensuring minimum spacing
            sorted_indices = np.argsort(profile)[::-1]
            peak_indices = [sorted_indices[0]]

            for idx in sorted_indices[1:]:
                if all(abs(idx - existing) > 5 for existing in peak_indices):
                    peak_indices.append(idx)
                if len(peak_indices) == num_peaks:
                    break

            peak_indices = np.array(peak_indices)

        # Select top N by prominence
        prominences = properties.get("prominences", np.ones_like(peak_indices))
        if len(prominences) != len(peak_indices):
            # Fallback prominences
            prominences = np.array([profile[i] for i in peak_indices])

        top_indices = np.argsort(prominences)[-num_peaks:]

        return peak_indices[top_indices]

    def _get_fwhm_points(
        self,
        profile: np.ndarray,
        peak_idx: int,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Find left and right half-maximum points for a peak.

        Searches outward from peak_idx to find where profile crosses
        half of the peak's maximum value.
        """
        peak_value = profile[peak_idx]
        if peak_value <= 0:
            logger.debug(
                f"Peak at {peak_idx} has non-positive value {peak_value}"
            )
            return []

        half_max = peak_value / 2
        points = []

        # Find left edge
        left_idx = self._find_edge(profile, peak_idx, half_max, "left")
        if left_idx is not None:
            points.append(
                self._interpolate_coordinate(left_idx, x_coords, y_coords)
            )

        # Find right edge
        right_idx = self._find_edge(profile, peak_idx, half_max, "right")
        if right_idx is not None:
            points.append(
                self._interpolate_coordinate(right_idx, x_coords, y_coords)
            )

        return points

    def _find_edge(
        self,
        profile: np.ndarray,
        peak_idx: int,
        target_value: float,
        direction: str,
    ) -> float | None:
        """Find interpolated index where profile crosses target value.

        Searches from peak_idx outward until it finds where the profile
        crosses target_value, then performs linear interpolation for
        sub-pixel precision.
        """
        if direction == "left":
            search_range = range(peak_idx - 1, -1, -1)
            neighbor_offset = 1
        elif direction == "right":
            search_range = range(peak_idx + 1, len(profile))
            neighbor_offset = -1
        else:
            raise ValueError("direction must be 'left' or 'right'")

        for i in search_range:
            current = profile[i]
            neighbor = profile[i + neighbor_offset]

            # Check if target is between current and neighbor
            if (
                min(current, neighbor)
                <= target_value
                <= max(current, neighbor)
            ):
                # Avoid division by zero for flat regions
                if abs(neighbor - current) < 1e-10:
                    return float(i)

                # Linear interpolation
                fraction = (target_value - current) / (neighbor - current)
                return i + fraction * -neighbor_offset

        return None

    def _interpolate_coordinate(
        self,
        idx: float,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> tuple[float, float]:
        """Interpolate 2D coordinate at fractional index position.

        Uses bilinear interpolation between the nearest integer indices
        to compute precise (x, y) coordinates.
        """
        # Clamp index to valid range
        n_points = len(x_coords)
        idx = max(0.0, min(float(idx), n_points - 1))

        # Handle exact integer index
        if idx.is_integer():
            idx_int = int(idx)
            return float(x_coords[idx_int]), float(y_coords[idx_int])

        # Bilinear interpolation
        idx_int = int(np.floor(idx))
        idx_frac = idx - idx_int

        # Handle edge case at last point
        if idx_int >= n_points - 1:
            return float(x_coords[-1]), float(y_coords[-1])

        x = x_coords[idx_int] + idx_frac * (
            x_coords[idx_int + 1] - x_coords[idx_int]
        )
        y = y_coords[idx_int] + idx_frac * (
            y_coords[idx_int + 1] - y_coords[idx_int]
        )

        return float(x), float(y)

    def _plot_profile_debug(
        self,
        profile: np.ndarray,
        peak_indices: np.ndarray,
        intersection_points: list[tuple[float, float]],
        spoke_id: int,
        slice_no: int,
        object_mask: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> None:
        """Create debug plot for radial profile analysis.

        Visualizes the profile data, detected peaks, half-maximum intersection
        points, and object mask regions to help verify the intersection point
        detection algorithm.

        Args:
            profile: Smoothed and detrended 1D intensity profile
            peak_indices: Array of indices where peaks were detected
            intersection_points: List of (x, y) coordinate tuples for
                half-max points in image space
            spoke_id: Spoke number (0-9)
            slice_no: Slice number (8-11)
            object_mask: Binary mask array indicating object locations (size x 3)
            x_coords: X coordinates in image space for each profile point
            y_coords: Y coordinates in image space for each profile point

        """
        if not self.report:
            return

        # Convert intersection points from image coordinates back to profile indices
        # by finding the closest point in the profile coordinates
        intersection_indices = []
        for xi, yi in intersection_points:
            distances = np.sqrt((x_coords - xi) ** 2 + (y_coords - yi) ** 2)
            closest_idx = np.argmin(distances)
            intersection_indices.append(closest_idx)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))

        # Profile positions (0 to length-1)
        profile_positions = np.arange(len(profile))

        # Plot the smoothed profile
        ax.plot(
            profile_positions,
            profile,
            color="#1976D2",
            linewidth=1.5,
            label="Smoothed profile",
        )

        # Mark detected peaks
        if len(peak_indices) > 0:
            ax.scatter(
                peak_indices,
                profile[peak_indices],
                color="#D32F2F",
                marker="^",
                s=80,
                zorder=5,
                label="Peaks",
            )

        # Mark half-maximum intersection points
        if len(intersection_indices) > 0:
            # Ensure indices are within bounds
            valid_indices = [
                idx for idx in intersection_indices
                if 0 <= idx < len(profile)
            ]
            if valid_indices:
                ax.scatter(
                    valid_indices,
                    profile[np.array(valid_indices).astype(int)],
                    color="#388E3C",
                    marker="o",
                    s=60,
                    zorder=5,
                    label="Half-max points",
                )

        # Add vertical bands for object mask regions
        if object_mask is not None and object_mask.shape[1] > 0:
            for obj_idx in range(object_mask.shape[1]):
                obj_mask_col = object_mask[:, obj_idx]
                # Find contiguous regions where mask is True
                mask_start = None
                for i, is_masked in enumerate(obj_mask_col):
                    if is_masked and mask_start is None:
                        mask_start = i
                    elif not is_masked and mask_start is not None:
                        # End of masked region
                        color = self.OBJECT_RIBBON_COLORS[
                            obj_idx % len(self.OBJECT_RIBBON_COLORS)
                        ]
                        ax.axvspan(
                            mask_start,
                            i,
                            alpha=0.15,
                            color=color,
                            zorder=1,
                        )
                        mask_start = None
                # Handle case where mask extends to end
                if mask_start is not None:
                    color = self.OBJECT_RIBBON_COLORS[
                        obj_idx % len(self.OBJECT_RIBBON_COLORS)
                    ]
                    ax.axvspan(
                        mask_start,
                        len(obj_mask_col),
                        alpha=0.15,
                        color=color,
                        zorder=1,
                    )

        # Set axis limits with padding
        profile_range = np.ptp(profile)
        if profile_range > 0:
            ax.set_ylim(
                profile.min() - 0.1 * profile_range,
                profile.max() + 0.1 * profile_range,
            )

        # Labels and title
        ax.set_xlabel("Profile Position", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.set_title(
            f"LCOD Profile Debug - Slice {slice_no}, Spoke {spoke_id}",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        plt.tight_layout()

        # Save figure
        data_path = Path(self.dcm_list[0].filename).parent.name
        img_path = (
            Path(self.report_path)
            / f"{data_path}_profile_debug_slice{slice_no}_spoke{spoke_id}.png"
        )
        fig.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.report_files.append(img_path)
