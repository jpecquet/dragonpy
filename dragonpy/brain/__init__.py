"""
Brain layer: the logic that turns sensor readings into muscle commands.

Contract (convention, not enforced):
  reads  - `dragonfly.sensors.*.reading` and `dragonfly.wings` (immutable spec).
  writes - `dragonfly.stroke_patterns` and `dragonfly.wing_frequency`.
  never  - reads body kinematic state, wing_states, or wing_phases directly.

A brain is stateful (one instance per dragonfly) and is called once per
slow tick with the elapsed slow-tick dt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..body.muscles import StrokePattern

if TYPE_CHECKING:
    from ..dragonfly import Dragonfly


class Brain(ABC):
    @abstractmethod
    def update(self, dragonfly: "Dragonfly", dt: float) -> None: ...


class NullBrain(Brain):
    """No-op brain. Stroke patterns and frequency remain as initialized."""

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        pass


class StaticBrain(Brain):
    """Installs a fixed set of patterns once, then no-ops.

    Useful for open-loop hover / forward flight runs, and as a base for
    simple closed-loop controllers that override `update`.
    """

    def __init__(
        self,
        patterns: list[StrokePattern],
        wing_frequency: float,
    ) -> None:
        self._patterns = patterns
        self._frequency = wing_frequency
        self._installed = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if self._installed:
            return
        for i, pat in enumerate(self._patterns):
            dragonfly.stroke_patterns[i] = pat
        dragonfly.wing_frequency = self._frequency
        self._installed = True
