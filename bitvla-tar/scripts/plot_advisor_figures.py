#!/usr/bin/env python3
"""Figures for advisor presentation: experiment roadmap + E3 progress snapshot."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report/figures"


def fig6_roadmap():
    """Two-track plan: track-1 (eval) then track-2 (train)."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Experiment roadmap (resource split)", fontsize=12, pad=12)

    # Track 1 row
    ax.text(0.1, 3.35, "Track 1 · 1 GPU · no new training", fontsize=10, fontweight="bold")
    steps1 = [
        ("E1", "code", 0.3, "done"),
        ("E2", "diag", 1.4, "done"),
        ("E3", "re-eval h=8", 2.5, "run"),
        ("E4", "h=4,2", 3.6, "queue"),
        ("E5", "TE/blend", 4.7, "queue"),
        ("E6–E7", "fail+/Plus", 5.9, "wait"),
    ]
    colors = {"done": "#2ca02c", "run": "#ff7f0e", "queue": "#aec7e8", "wait": "#cccccc"}
    for label, sub, x, st in steps1:
        rect = mpatches.FancyBboxPatch(
            (x, 2.85),
            0.95,
            0.55,
            boxstyle="round,pad=0.02",
            facecolor=colors[st],
            edgecolor="0.3",
        )
        ax.add_patch(rect)
        ax.text(x + 0.48, 3.25, label, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 0.48, 2.95, sub, ha="center", va="center", fontsize=7)

    ax.annotate(
        "",
        xy=(6.9, 3.12),
        xytext=(6.2, 3.12),
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    ax.text(7.05, 3.12, "informs which\nablations to run", fontsize=8, va="center")

    # Track 2 row
    ax.text(0.1, 1.85, "Track 2 · 4 GPUs · training", fontsize=10, fontweight="bold")
    steps2 = [
        ("A3", "λ=0 ctrl", 0.3, "must"),
        ("A1–A4", "TAR+seeds", 1.5, "must"),
        ("B", "2–3 variants", 3.0, "opt"),
        ("C", "λ sweep", 4.5, "opt"),
    ]
    for label, sub, x, _ in steps2:
        rect = mpatches.FancyBboxPatch(
            (x, 1.35),
            1.05,
            0.55,
            boxstyle="round,pad=0.02",
            facecolor="#1f77b4" if "must" in sub or label.startswith("A") else "#c6dbef",
            edgecolor="0.3",
            alpha=0.85 if label.startswith("A") else 0.6,
        )
        ax.add_patch(rect)
        ax.text(x + 0.52, 1.75, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white" if label.startswith("A3") or label.startswith("A1") else "black")
        ax.text(x + 0.52, 1.45, sub, ha="center", va="center", fontsize=7)

    ax.text(0.3, 0.55, "~2 days wall-clock per full 10k-step run on 4 GPUs  ·  ~9 days on 1 GPU (E8)", fontsize=8, color="0.35")

    legend_patches = [
        mpatches.Patch(facecolor=colors["done"], label="completed"),
        mpatches.Patch(facecolor=colors["run"], label="in progress"),
        mpatches.Patch(facecolor=colors["queue"], label="queued"),
        mpatches.Patch(facecolor=colors["wait"], label="not started"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    out = FIG / "fig6_experiment_roadmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig7_oct_results_bar():
    """Simple bar chart: success rate Oct 2026."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = ["Baseline\n(official)", "TAR\nλ=0.10"]
    vals = [83.6, 88.4]
    colors = ["tab:blue", "tab:red"]
    bars = ax.bar(names, vals, color=colors, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate on LIBERO-10 (%)")
    ax.set_title("Oct 2026 cloud reproduction (500 episodes)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=10)
    ax.axhline(83.6, color="0.85", ls="--", lw=0.8)
    ax.text(1.35, 84.2, "McNemar p=0.025", fontsize=8, color="0.4")
    fig.tight_layout()
    out = FIG / "fig7_success_oct2026.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig6_roadmap()
    fig7_oct_results_bar()