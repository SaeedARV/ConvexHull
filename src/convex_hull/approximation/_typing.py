from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
RngLike = Union[None, int, np.random.Generator]

class SupportsArgmaxDot(Protocol):
    def argmax_dot(self, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        ...

class SupportsHouseholder(Protocol):
    def apply_householder_to_rows(self, base_dirs: np.ndarray, v_house: np.ndarray) -> np.ndarray:
        ...

BackendCaps = Tuple[SupportsArgmaxDot, Optional[SupportsHouseholder]]
