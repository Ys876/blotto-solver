"""CLI entry point for the Blotto solver."""

import argparse
import numpy as np
from blotto import get_all_allocations, build_payoff_matrix, compute_outcome
from solver import solve_blotto
from benchmark import run_benchmarks, print_results
from visualize import plot_payoff_matrix, plot_mixed_strategy, plot_win_rates


def play_interactive(allocations, probs, n_troops, n_fields):
    """Let the user challenge the Nash equilibrium strategy."""
    rng = np.random.default_rng()
    alloc_set = set(allocations)

    print(f"\n{'='*55}")
    print("  CHALLENGE THE NASH STRATEGY")
    print(f"{'='*55}")
    print(f"  Distribute {n_troops} troops across {n_fields} battlefields.")
    print("  The Nash strategy will draw from its mixed strategy each round.")
    print("  Type 'quit' to exit.\n")

    wins, losses, draws, rounds = 0, 0, 0, 0

    while True:
        print(f"  Round {rounds + 1}  |  Your score: {wins}W {losses}L {draws}D")
        raw = input(f"  Enter your allocation ({n_fields} numbers summing to {n_troops}): ").strip()

        if raw.lower() in ("quit", "q", "exit"):
            break

        try:
            parts = tuple(int(x) for x in raw.replace(",", " ").split())
        except ValueError:
            print("  Invalid input — enter integers separated by spaces.\n")
            continue

        if len(parts) != n_fields:
            print(f"  Need exactly {n_fields} numbers.\n")
            continue
        if sum(parts) != n_troops:
            print(f"  Numbers must sum to {n_troops} (yours sum to {sum(parts)}).\n")
            continue
        if any(x < 0 for x in parts):
            print("  No negative values allowed.\n")
            continue
        if parts not in alloc_set:
            print("  (Note: this is a valid allocation.)\n")

        # Nash draws from its mixed strategy
        nash_idx = rng.choice(len(allocations), p=probs)
        nash_alloc = allocations[nash_idx]
        outcome = compute_outcome(parts, nash_alloc)

        print(f"\n  You:  {parts}")
        print(f"  Nash: {nash_alloc}")

        field_results = []
        for i, (y, n) in enumerate(zip(parts, nash_alloc)):
            if y > n:
                field_results.append(f"F{i+1}✓")
            elif n > y:
                field_results.append(f"F{i+1}✗")
            else:
                field_results.append(f"F{i+1}~")
        print(f"  Fields: {' '.join(field_results)}")

        if outcome == 1:
            wins += 1
            print("  >>> YOU WIN this round!\n")
        elif outcome == -1:
            losses += 1
            print("  >>> Nash wins this round.\n")
        else:
            draws += 1
            print("  >>> Draw.\n")

        rounds += 1

    if rounds > 0:
        score = (wins + 0.5 * draws) / rounds
        print(f"\n  Final: {rounds} rounds — {wins}W {losses}L {draws}D  |  Score: {score:.1%}")
        print("  (Nash equilibrium guarantees ~50% score against any fixed strategy.)\n")


def _solve(n_troops, n_fields):
    print(f"\nColonel Blotto — {n_troops} troops, {n_fields} fields")
    print("=" * 55)
    print("Building strategy space...", end=" ", flush=True)
    allocations = get_all_allocations(n_troops, n_fields)
    print(f"{len(allocations)} strategies found.")
    print("Building payoff matrix...", end=" ", flush=True)
    M = build_payoff_matrix(allocations)
    print("done.")
    print("Solving Nash equilibrium (LP)...", end=" ", flush=True)
    probs, game_value = solve_blotto(M)
    print("done.")
    return allocations, M, probs, game_value


def main():
    parser = argparse.ArgumentParser(
        description="Colonel Blotto Nash Equilibrium Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # solve + benchmark + visualize (default 10t/3f)
  python main.py --troops 10 --fields 4 # larger game
  python main.py --play                 # challenge the Nash strategy interactively
  python main.py --play --troops 6 --fields 3
        """
    )
    parser.add_argument("--troops", type=int, default=10, help="Number of troops (default: 10)")
    parser.add_argument("--fields", type=int, default=3, help="Number of battlefields (default: 3)")
    parser.add_argument("--games", type=int, default=10_000, help="Simulation games (default: 10000)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--play", action="store_true", help="Interactive mode: challenge the Nash strategy")
    args = parser.parse_args()

    allocations, M, probs, game_value = _solve(args.troops, args.fields)

    print(f"\nGame value: {game_value:.4f}  (0 = perfectly symmetric)")
    print("\nTop strategies in Nash equilibrium mixed strategy:")
    top_idx = np.argsort(probs)[::-1]
    for rank, i in enumerate(top_idx[:10]):
        if probs[i] < 0.001:
            break
        print(f"  #{rank+1:2d}  {str(allocations[i]):<30}  p = {probs[i]:.4f}")

    if args.play:
        play_interactive(allocations, probs, args.troops, args.fields)
        return

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
