"""
Terminal progress helpers using Rich with a pip-like layout.
"""

import sys
import time
from typing import Optional, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
)

PASTEL_GREEN = "#98F797"
PASTEL_RED = "#FF746C"
PASTEL_YELLOW = "#F4D58D"
GRAY = "#49484B"


def _format_rate(rate: float, unit: str) -> str:
    if rate <= 0:
        return f"-- {unit}/s"
    if rate >= 100:
        return f"{rate:.0f} {unit}/s"
    if rate >= 10:
        return f"{rate:.1f} {unit}/s"
    return f"{rate:.2f} {unit}/s"


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "0:00:00"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


class ProgressBar:
    """Small adapter around ``rich.progress.Progress`` used by plotting utilities."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        unit: str = "item",
        stream: Optional[TextIO] = None,
        min_interval: float = 0.1,
        label_width: int = 12,
    ) -> None:
        self.total = max(0, int(total))
        self.label = str(label)
        self.unit = unit
        self.unit_plural = unit if unit.endswith("s") else f"{unit}s"
        self.stream = stream or sys.stdout
        self.min_interval = max(0.0, float(min_interval))
        self.label_width = max(1, int(label_width))

        self.n = 0
        self._started = time.perf_counter()
        self._closed = False

        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._console = Console(
            file=self.stream,
            force_terminal=is_tty,
            soft_wrap=False,
        )
        self._progress = Progress(
            TextColumn(f"{{task.description:<{self.label_width}}}"),
            BarColumn(
                style=GRAY,
                complete_style=PASTEL_GREEN,
                finished_style=PASTEL_GREEN,
                pulse_style=GRAY,
            ),
            TextColumn("{task.fields[count_text]}", style=PASTEL_GREEN),
            TextColumn("{task.fields[rate_text]}", style=PASTEL_RED),
            TextColumn("{task.fields[eta_text]}", style=PASTEL_YELLOW),
            console=self._console,
            transient=False,
            refresh_per_second=max(1, int(round(1.0 / self.min_interval))) if self.min_interval > 0 else 10,
            disable=not is_tty,
        )
        self._task_id: Optional[int] = None

    def __enter__(self):
        if self._progress.disable:
            if self.total > 0:
                self.stream.write(f"{self.label}: {self._count_text()}\n")
                self.stream.flush()
            return self

        self._progress.start()
        total = self.total if self.total > 0 else None
        self._task_id = self._progress.add_task(
            self.label,
            total=total,
            completed=0,
            count_text=self._count_text(),
            rate_text=f"-- {self.unit_plural}/s",
            eta_text="--:--:--",
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(mark_complete=(exc_type is None))
        return False

    def update(self, delta: int = 1) -> None:
        self.set(self.n + int(delta))

    def set(self, value: int) -> None:
        if self._closed:
            return

        clamped = max(0, int(value))
        if self.total > 0:
            clamped = min(clamped, self.total)
        self.n = clamped

        if self._progress.disable:
            return
        if self._task_id is None:
            return

        elapsed = max(0.0, time.perf_counter() - self._started)
        rate = self.n / elapsed if elapsed > 0 else 0.0
        remaining = None
        if self.total > 0 and self.n < self.total and rate > 0:
            remaining = (self.total - self.n) / rate
        self._progress.update(
            self._task_id,
            completed=self.n,
            count_text=self._count_text(),
            rate_text=_format_rate(rate, self.unit_plural),
            eta_text=_format_eta(remaining),
        )

    def close(self, *, mark_complete: bool = True) -> None:
        if self._closed:
            return

        if mark_complete and self.total > 0:
            self.set(self.total)

        if self._progress.disable:
            if self.total > 0:
                elapsed = max(0.0, time.perf_counter() - self._started)
                rate = self.n / elapsed if elapsed > 0 else 0.0
                self.stream.write(
                    f"{self.label}: {self._count_text()} "
                    f"{_format_rate(rate, self.unit_plural)} "
                    f"{_format_eta(0.0 if self.n >= self.total else None)}\n"
                )
            else:
                self.stream.write(f"{self.label}: {self.n}\n")
            self.stream.flush()
        else:
            self._progress.stop()

        self._closed = True

    def _count_text(self) -> str:
        if self.total > 0:
            return f"{self.n}/{self.total} {self.unit_plural}"
        return f"{self.n} {self.unit_plural}"


def pip_progress(
    total: int,
    label: str,
    *,
    unit: str = "item",
    stream: Optional[TextIO] = None,
    min_interval: float = 0.1,
    label_width: int = 12,
) -> ProgressBar:
    return ProgressBar(
        total,
        label,
        unit=unit,
        stream=stream,
        min_interval=min_interval,
        label_width=label_width,
    )
