"""Module containing task execution related functions."""

from __future__ import annotations

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Python imports
import time
from typing import ParamSpec

# Local imports
from hazenlib.types import Measurement, Result

P = ParamSpec("P")

def timed_execution(
    task_method: Callable[P, Result],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Result:
    """Execute task method and append timing metadata to result."""
    start: float = time.perf_counter()

    # Execute the actual task
    result: Result = task_method(*args, **kwargs)
    # Calculate and inject timing
    elapsed: float = time.perf_counter() - start
    result.add_measurement(
        Measurement(
            name="ExecutionMetadata",
            type="measured",
            description="analysis_duration",
            value=round(elapsed, 4),
            unit="s",
        ),
    )
    return result
