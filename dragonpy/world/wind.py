"""
Wind field: a function of world-frame position and time returning a
world-frame velocity.

v1 ships a still-air field. Uniform / gusty / turbulent fields are easy to
drop in later by providing a different callable to `Environment`.
"""

import numpy as np


def still_air(position: np.ndarray, t: float) -> np.ndarray:
    return np.zeros(3)
