from typing import Callable

import numpy as np


def vote(player_values: np.ndarray) -> Callable:
    """Vote on objective"""
    if player_values.mean() > 0.5:
        return "egalitarian"
    else:
        return "utilitarian"
