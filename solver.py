"""Nash equilibrium solver for Colonel Blotto via linear programming."""

import numpy as np
from scipy.optimize import linprog


def solve_blotto(payoff_matrix: np.ndarray, prob_threshold: float = 0.001):
    """
    Find the row player's Nash equilibrium mixed strategy.

    LP: maximize v s.t. M.T @ p >= v*1, sum(p) = 1, p >= 0
    Rewritten for linprog (minimization):
      variables: [p_0, ..., p_{n-1}, v]
      minimize -v
      s.t. for each j: -sum_i M[i,j]*p_i + v <= 0
           sum_i p_i = 1
           p_i >= 0, v unbounded
    """
    n = payoff_matrix.shape[0]

    # Objective: minimize -v  (last variable is v)
    c = np.zeros(n + 1)
    c[-1] = -1.0

    # Inequality constraints: -M[:,j] @ p + v <= 0  for each j
    # Shape: (n, n+1)
    A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # Equality: sum(p) = 1
    A_eq = np.ones((1, n + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])

    # Bounds: p_i in [0, 1], v unbounded
    bounds = [(0, 1)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    probs = result.x[:n]
    game_value = result.x[-1]

    # Normalize and zero out near-zero probabilities
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    probs[probs < prob_threshold] = 0.0
    probs /= probs.sum()

    return probs, game_value
