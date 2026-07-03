"""CLI entry point for the Blotto solver."""

import argparse
import numpy as np
from blotto import get_all_allocations, build_payoff_matrix
from solver import solve_blotto
from benchmark import run_benchmarks, print_results
from visualize import plot_payoff_matrix, plot_mixed_strategy, plot_win_rates


def main():
    parser = argparse.ArgumentParser(description="Colonel Blotto Nash Equilibrium Solver")
    parser.add_argument("--troops", type=int, default=10, help="Number of troops (default: 10)")
    parser.add_argument("--fields", type=int, default=3, help="Number of battlefields (default: 3)")
    parser.add_argument("--games", type=int, default=10_000, help="Simulation games (default: 10000)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    print(f"\nColonel Blotto Solver — {args.troops} troops, {args.fields} fields")
    print("=" * 55)

    print("Building strategy space...", end=" ", flush=True)
    allocations = get_all_allocations(args.troops, args.fields)
    print(f"{len(allocations)} strategies found.")

    print("Building payoff matrix...", end=" ", flush=True)
    M = build_payoff_matrix(allocations)
    print("done.")

    print("Solving Nash equilibrium (LP)...", end=" ", flush=True)
    probs, game_value = solve_blotto(M)
    print("done.")

    print(f"\nGame value: {game_value:.4f}  (0 = perfectly symmetric)")
    print(f"\nTop strategies in Nash equilibrium mixed strategy:")
    top_idx = np.argsort(probs)[::-1]
    for rank, i in enumerate(top_idx[:10]):
        if probs[i] < 0.001:
            break
        print(f"  #{rank+1:2d}  {str(allocations[i]):<30}  p = {probs[i]:.4f}")

    print(f"\nRunning benchmarks ({args.games:,} games each)...")
    results = run_benchmarks(probs, allocations, args.troops, args.fields, n_games=args.games)
    print_results(results)

    if not args.no_plot:
        print("\nGenerating visualizations...")
        plot_payoff_matrix(M)
        plot_mixed_strategy(probs, allocations)
        plot_win_rates(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
