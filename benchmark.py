"""Benchmark LP-optimal mixed strategy against baselines."""

import numpy as np
from blotto import get_all_allocations, compute_outcome


def _uniform_alloc(n_troops: int, n_fields: int) -> tuple:
    base = n_troops // n_fields
    remainder = n_troops % n_fields
    alloc = [base] * n_fields
    for i in range(remainder):
        alloc[i] += 1
    return tuple(alloc)


def _greedy_alloc(n_troops: int, n_fields: int) -> tuple:
    alloc = [0] * n_fields
    alloc[0] = n_troops
    return tuple(alloc)


def baseline_strategy(name: str, allocations: list[tuple],
                      n_troops: int, n_fields: int) -> np.ndarray:
    """Return a probability distribution over allocations for a baseline."""
    n = len(allocations)
    alloc_set = {a: i for i, a in enumerate(allocations)}
    probs = np.zeros(n)

    if name == "uniform":
        idx = alloc_set.get(_uniform_alloc(n_troops, n_fields))
        if idx is not None:
            probs[idx] = 1.0
        else:
            probs[:] = 1.0 / n

    elif name == "greedy":
        idx = alloc_set.get(_greedy_alloc(n_troops, n_fields))
        if idx is not None:
            probs[idx] = 1.0
        else:
            probs[:] = 1.0 / n

    elif name == "random":
        probs[:] = 1.0 / n

    return probs


def simulate(lp_probs: np.ndarray, opponent_probs: np.ndarray,
             allocations: list[tuple], n_games: int = 10_000,
             rng: np.random.Generator = None) -> dict:
    """Simulate n_games and return win/loss/draw counts and win rate."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(allocations)
    lp_idx = rng.choice(n, size=n_games, p=lp_probs)
    opp_idx = rng.choice(n, size=n_games, p=opponent_probs)

    outcomes = np.array([
        compute_outcome(allocations[lp_idx[k]], allocations[opp_idx[k]])
        for k in range(n_games)
    ])

    wins = int((outcomes == 1).sum())
    losses = int((outcomes == -1).sum())
    draws = int((outcomes == 0).sum())
    win_rate = wins / n_games

    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def run_benchmarks(lp_probs: np.ndarray, allocations: list[tuple],
                   n_troops: int, n_fields: int,
                   n_games: int = 10_000) -> dict:
    """Run all benchmarks and return results dict."""
    rng = np.random.default_rng(42)
    results = {}
    for name in ("uniform", "greedy", "random"):
        opp_probs = baseline_strategy(name, allocations, n_troops, n_fields)
        results[name] = simulate(lp_probs, opp_probs, allocations, n_games, rng)
    return results


def print_results(results: dict):
    print("\nBenchmark Results (10,000 simulated games)")
    print("-" * 72)
    for name, r in results.items():
        n = r['wins'] + r['losses'] + r['draws']
        score = (r['wins'] + 0.5 * r['draws']) / n
        label = f"LP-Optimal vs {name.capitalize():<8}"
        print(f"{label}  W: {r['wins']:<5}  L: {r['losses']:<5}  "
              f"D: {r['draws']:<5}  Win rate: {r['win_rate']:.1%}  Score: {score:.1%}")
