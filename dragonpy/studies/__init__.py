"""Higher-level analysis utilities built on top of the simulation core."""

from .capture import (
    CaptureStudyResult,
    capture_rate,
    captured_mask,
    run_capture_study,
    time_to_intercept,
)

__all__ = [
    "CaptureStudyResult",
    "capture_rate",
    "captured_mask",
    "run_capture_study",
    "time_to_intercept",
]
