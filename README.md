# Blotto Solver — Mixed-Strategy Nash Equilibrium via LP

Colonel Blotto is a two-player zero-sum game where each player simultaneously distributes N troops across K battlefields. A player wins a battlefield if they send strictly more troops there; whoever wins more battlefields wins the game. Because both players choose simultaneously with no information about the opponent, the optimal strategy is a *mixed strategy* — a probability distribution over allocations rather than any single fixed deployment.

---

## Visualizations

### Payoff Matrix
Every cell shows the outcome (+1 win / 0 draw / −1 loss) when the row player's strategy meets the column player's strategy. The symmetric checkerboard pattern confirms no pure strategy dominates — you always need to mix.

![Payoff Matrix](payoff_matrix.png)

### Nash Equilibrium Mixed Strategy
The LP solver spreads probability across ~15 asymmetric allocations. No single allocation gets more than ~15% weight — that unpredictability is the whole point.

![Mixed Strategy](mixed_strategy.png)

### Win Rate vs Baselines
LP-optimal never loses to Greedy (68.3% wins, 31.7% draws). Against Uniform and Random the score is near 50% — exactly what Nash equilibrium theory predicts.

![Win Rates](win_rates.png)

---

## How the LP Solver Works

The Nash equilibrium is found by solving a minimax linear program. The row player wants to maximize their guaranteed minimum expected payoff `v` regardless of what the opponent does:

```
Maximize  v
Subject to:
  For each opponent strategy j:  Σ_i p_i · M[i,j] ≥ v
  Σ_i p_i = 1
  p_i ≥ 0
```

Where `M` is the payoff matrix (`M[i,j] = +1` if strategy i beats j, `-1` if it loses, `0` if tied) and `p` is the mixed strategy probability vector. This is rewritten as a minimization problem for `scipy.optimize.linprog` (HiGHS solver). The game value `v ≈ 0` confirms the game is perfectly symmetric.

---

## Setup

```bash
git clone https://github.com/Ys876/blotto-solver.git
cd blotto-solver
pip install -r requirements.txt
```

---

## Usage

```bash
# Solve + benchmark + generate all plots (default: 10 troops, 3 fields)
python main.py

# Try a larger game
python main.py --troops 10 --fields 4

# Challenge the Nash strategy yourself — interactive mode
python main.py --play

# Play with different parameters
python main.py --play --troops 6 --fields 3

# Skip plots
python main.py --no-plot
```

---

## Play Against the Nash Strategy

The `--play` flag lets you enter your own troop allocations and compete against the Nash mixed strategy in real time:

```
$ python main.py --play

Colonel Blotto — 10 troops, 3 fields
=======================================================
Solving Nash equilibrium (LP)... done.

=======================================================
  CHALLENGE THE NASH STRATEGY
=======================================================
  Distribute 10 troops across 3 battlefields.
  The Nash strategy will draw from its mixed strategy each round.
  Type 'quit' to exit.

  Round 1  |  Your score: 0W 0L 0D
  Enter your allocation (3 numbers summing to 10): 4 3 3

  You:  (4, 3, 3)
  Nash: (0, 4, 6)
  Fields: F1✓ F2✗ F3✗
  >>> Nash wins this round.

  Round 2  |  Your score: 0W 1L 0D
  Enter your allocation (3 numbers summing to 10): 5 5 0

  You:  (5, 5, 0)
  Nash: (6, 0, 4)
  Fields: F1✗ F2✓ F3✗
  >>> Nash wins this round.
```

No matter what fixed allocation you settle on, the Nash strategy will exploit it over time. That's the point — the equilibrium is *unexploitable*.

---

## Benchmark Results (10 troops, 3 fields, 10,000 simulated games)

```
LP-Optimal vs Uniform   W: 3136   L: 3084   D: 3780   Win rate: 31.4%  Score: 50.3%
LP-Optimal vs Greedy    W: 6831   L: 0      D: 3169   Win rate: 68.3%  Score: 84.2%
LP-Optimal vs Random    W: 3754   L: 2986   D: 3260   Win rate: 37.5%  Score: 53.8%
```

*Score = (W + 0.5·D) / N — the standard game-theoretic expected value treating draws as 0.5.*

The high draw rate vs Uniform is expected: Uniform plays a single fixed allocation that the Nash strategy sometimes ties and sometimes beats, but the key property holds — **W > L in every matchup**, which is exactly what Nash guarantees (expected payoff ≥ 0). Against Greedy, the LP strategy **never loses** (0 losses across 10,000 games).

---

## Computational Complexity

The strategy space grows combinatorially: `C(N+K-1, K-1)` allocations total.

| Troops | Fields | Strategies | Payoff Matrix |
|--------|--------|------------|---------------|
| 10 | 3 | 66 | 66×66 |
| 10 | 4 | 286 | 286×286 |
| 10 | 5 | 1,001 | 1,001×1,001 |
| 20 | 5 | 10,626 | 10,626×10,626 |

The LP scales roughly as O(n³) with strategy count. For (10, 5) it takes ~30s; beyond that, approximate methods (column generation, support enumeration) are needed.

---

## File Structure

```
blotto.py      — strategy enumeration, payoff computation, matrix builder
solver.py      — minimax LP via scipy linprog (HiGHS) — exact Nash equilibrium
benchmark.py   — 10,000-game simulation vs uniform / greedy / random baselines
visualize.py   — payoff heatmap, strategy bar chart, win-rate comparison
main.py        — CLI: solve / benchmark / visualize / interactive play mode
```
