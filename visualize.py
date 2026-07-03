"""Visualizations for Blotto solver results."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os


def plot_payoff_matrix(payoff_matrix: np.ndarray, save_path: str = "payoff_matrix.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(payoff_matrix, cmap="RdYlGn", center=0, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Outcome (+1 win / 0 draw / -1 loss)"})
    ax.set_title("Payoff Matrix — Row player (LP-Optimal) vs Column player", fontsize=13)
    ax.set_xlabel("Opponent Strategy Index")
    ax.set_ylabel("Row Player Strategy Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mixed_strategy(probs: np.ndarray, allocations: list[tuple],
                        save_path: str = "mixed_strategy.png", threshold: float = 0.01):
    mask = probs > threshold
    filtered_probs = probs[mask]
    filtered_allocs = [allocations[i] for i in np.where(mask)[0]]
    labels = [str(a) for a in filtered_allocs]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    bars = ax.bar(range(len(labels)), filtered_probs, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probability")
    ax.set_title("Nash Equilibrium Mixed Strategy (strategies with p > 1%)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for bar, p in zip(bars, filtered_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{p:.1%}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_win_rates(benchmark_results: dict, save_path: str = "win_rates.png"):
    names = list(benchmark_results.keys())
    win_rates = [benchmark_results[n]["win_rate"] for n in names]
    colors = ["#2ecc71", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([n.capitalize() for n in names], win_rates,
                  color=colors[:len(names)], edgecolor="white", width=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("LP-Optimal Strategy Win Rate vs Baselines", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
