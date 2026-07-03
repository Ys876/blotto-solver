"""Core game logic for Colonel Blotto."""

import itertools
import numpy as np


def get_all_allocations(n_troops: int, n_fields: int) -> list[tuple]:
    """Return all ordered ways to distribute n_troops across n_fields."""
    if n_fields == 1:
        return [(n_troops,)]
    result = []
    for first in range(n_troops + 1):
        for rest in get_all_allocations(n_troops - first, n_fields - 1):
            result.append((first,) + rest)
    return result


def compute_outcome(alloc_a: tuple, alloc_b: tuple) -> int:
    """Return 1 if A wins, -1 if B wins, 0 if tie."""
    wins_a = sum(a > b for a, b in zip(alloc_a, alloc_b))
    wins_b = sum(b > a for a, b in zip(alloc_a, alloc_b))
    if wins_a > wins_b:
        return 1
    elif wins_b > wins_a:
        return -1
    return 0


def build_payoff_matrix(allocations: list[tuple]) -> np.ndarray:
    """Build NxN payoff matrix where M[i][j] = outcome of strategy i vs j."""
    n = len(allocations)
    M = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(allocations):
        for j, b in enumerate(allocations):
            M[i, j] = compute_outcome(a, b)
    return M
