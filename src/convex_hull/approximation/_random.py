from __future__ import annotations

from typing import Optional, Union

import numpy as np

RngLike = Union[None, int, np.random.Generator]


def get_rng(random_state: RngLike) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
