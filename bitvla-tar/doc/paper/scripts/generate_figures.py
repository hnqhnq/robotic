"""
Regenerate TAR paper figures from tabulated experiment data.

Data sources:
  - doc/20260504-实验报告.md (Table IV ablation, per-task S1)
  - doc/paper/tables/*.tex (main results, per-task success)

Usage:
  python scripts/generate_figures.py
  # outputs PDF + PNG to ../figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Data from experiment reports / paper tables ---

LAMBDAS_ABLATION = [0.0, 0.01, 0.05, 0.10]
SUCCESS = [86.4, 89.0, 90.8, 93.6]
S1 = [0.0154, 0.0111, 0.0104, 0.0099]
MAX_JUMP = [0.1613, 0.0844, 0.0794, 0.0761]
IB = [0.0477, 0.0421, 0.0451, 0.0460]
GRIPPER_SW = [0.572, 0.497, 0.439, 0.435]

# Fig.1 uses λ ∈ {0, 0.05, 0.10}
LAMBDAS_FIG1 = [0.0, 0.05, 0.10]
S1_FIG1 = [0.0154, 0.0104, 0.0099]
IB_FIG1 = [0.0477, 0.0451, 0.0460]
MAXJ_FIG1 = [0.1613, 0.0794, 0.0761]

TASK_LABELS = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
]
TASK_NAMES = [
    "Soup+Sauce", "Cheese+Butter", "Stove+Moka", "Bowl+Drawer",
    "Mugs+Plates", "Book+Caddy", "Mug+Pudding", "Soup+Cheese",
    "2 Moka", "Mug+Micro",
]

BASELINE_SUCCESS = [90, 80, 94, 96, 50, 88, 56, 82, 92, 80]
TAR005_SUCCESS = [94, 94, 94, 96, 60, 96, 82, 98, 84, 94]

# Per-task S1 from v4 report / paper Fig.4
PER_TASK_S1 = {
    0.00: [0.0177, 0.0200, 0.0154, 0.0136, 0.0164, 0.0106, 0.0148, 0.0206, 0.0103, 0.0149],
    0.01: [0.0122, 0.0145, 0.0110, 0.0111, 0.0119, 0.0081, 0.0101, 0.0136, 0.0083, 0.0105],
    0.05: [0.0110, 0.0134, 0.0103, 0.0100, 0.0117, 0.0075, 0.0095, 0.0131, 0.0076, 0.0102],
    0.10: [0.0111, 0.0136, 0.0100, 0.0094, 0.0111, 0.0067, 0.0094, 0.0119, 0.0068, 0.0089],
}

COL_BASELINE = "#d62728"
COL_TAR = "#1f77b4"
COL_LAMBDA = ["#888888", "#6baed6", "#2171b5"]


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  wrote {path}")


def _synthetic_chunk_trajectory(seed: int, boundary_jump: float) -> np.ndarray:
    """Illustrative 4-chunk action dim-1 trajectory (not raw eval logs)."""
    rng = np.random.default_rng(seed)
    chunks = []
    t = 0.0
    for _ in range(4):
        n = 8
        interior = np.linspace(t, t + 0.08, n)
        interior += rng.normal(0, 0.003, n)
        chunks.append(interior)
        t = interior[-1] + boundary_jump * (1 + 0.3 * rng.random())
    return np.concatenate(chunks)


def fig1_overview() -> None:
    fig = plt.figure(figsize=(10, 3.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    # (a) Schematic trajectories
    ax0 = fig.add_subplot(gs[0])
    steps = np.arange(32)
    base = _synthetic_chunk_trajectory(0, boundary_jump=0.06)
    tar = _synthetic_chunk_trajectory(1, boundary_jump=0.015)
    ax0.plot(steps, base, color=COL_BASELINE, lw=2, label=r"Baseline ($\lambda{=}0$)")
    ax0.plot(steps, tar, color=COL_TAR, lw=2, label=r"w/ TAR ($\lambda{=}0.05$)")
    for b in [8, 16, 24]:
        ax0.axvline(b - 0.5, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax0.set_xlabel("Timestep")
    ax0.set_ylabel("Action value (dim 1)")
    ax0.set_title("(a) Action trajectories across chunks")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_xlim(0, 31)
    ax0.text(3.5, ax0.get_ylim()[1] * 0.92, "Chunk 1", fontsize=7, ha="center")
    ax0.text(11.5, ax0.get_ylim()[1] * 0.92, "Chunk 2", fontsize=7, ha="center")

    # (b) Smoothness bars
    ax1 = fig.add_subplot(gs[1])
    metrics = ["S1\n(intra)", "IB\n(inter)", "MaxJump"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, (lam, c) in enumerate(zip(LAMBDAS_FIG1, COL_LAMBDA)):
        vals = [S1_FIG1[i], IB_FIG1[i], MAXJ_FIG1[i]]
        ax1.bar(x + (i - 1) * width, vals, width, label=f"$\\lambda_{{TAR}}={lam}$", color=c)
    baseline_vals = [S1_FIG1[0], IB_FIG1[0], MAXJ_FIG1[0]]
    tar_vals = [S1_FIG1[2], IB_FIG1[2], MAXJ_FIG1[2]]
    reductions = [(b - t) / b * 100 for b, t in zip(baseline_vals, tar_vals)]
    for j, pct in enumerate(reductions):
        if j == 1:
            pct = (baseline_vals[1] - tar_vals[1]) / baseline_vals[1] * 100
        ax1.text(j + width, max(tar_vals[j], baseline_vals[j]) * 1.05,
                 f"{pct:.0f}%", ha="center", fontsize=8, color=COL_TAR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Mean absolute variation")
    ax1.set_title("(b) Smoothness metrics")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_ylim(0, max(MAXJ_FIG1) * 1.25)

    fig.suptitle("Action chunking discontinuities; TAR reduces boundary-step velocities", fontsize=10, y=1.02)
    _save(fig, "fig1_overview")
    plt.close(fig)


def fig2_per_task() -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(TASK_LABELS))
    w = 0.38
    bars_b = ax.bar(x - w / 2, BASELINE_SUCCESS, w, label=r"Baseline ($\lambda{=}0$)", color=COL_BASELINE)
    bars_t = ax.bar(x + w / 2, TAR005_SUCCESS, w, label=r"TAR ($\lambda{=}0.05$)", color=COL_TAR)
    for i, (b, t) in enumerate(zip(BASELINE_SUCCESS, TAR005_SUCCESS)):
        delta = t - b
        if delta != 0:
            color = "#2ca02c" if delta > 0 else "#d62728"
            ax.annotate(f"{delta:+d}", xy=(x[i] + w / 2, t + 1), ha="center", fontsize=7, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=9)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Task Success Rates on LIBERO-10")
    ax.legend()
    ax.axhline(86.4, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(90.8, color=COL_TAR, ls="--", lw=0.8, alpha=0.5)
    _save(fig, "fig2_per_task")
    plt.close(fig)


def fig3_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.plot(LAMBDAS_ABLATION, SUCCESS, "o-", color=COL_TAR, lw=2, markersize=8)
    for lam, sr in zip(LAMBDAS_ABLATION, SUCCESS):
        ax.annotate(f"{sr:.1f}%", (lam, sr), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"$\lambda_{\mathrm{TAR}}$")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(a) Task Success Rate")
    ax.set_xticks(LAMBDAS_ABLATION)
    ax.set_ylim(84, 96)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    norm_s1 = np.array(S1) / S1[0]
    norm_mj = np.array(MAX_JUMP) / MAX_JUMP[0]
    norm_gs = np.array(GRIPPER_SW) / GRIPPER_SW[0]
    ax.plot(LAMBDAS_ABLATION, norm_s1, "s-", label="S1", lw=2)
    ax.plot(LAMBDAS_ABLATION, norm_mj, "^-", label="MaxJump", lw=2)
    ax.plot(LAMBDAS_ABLATION, norm_gs, "d-", label="GripperSw", lw=2)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\lambda_{\mathrm{TAR}}$")
    ax.set_ylabel("Normalized Value (Baseline = 1)")
    ax.set_title("(b) Smoothness Metrics")
    ax.set_xticks(LAMBDAS_ABLATION)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    fig.suptitle(r"TAR Ablation: Effect of $\lambda_{\mathrm{TAR}}$", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_ablation")
    plt.close(fig)


def fig4_per_task_s1() -> None:
    lambdas = [0.00, 0.01, 0.05, 0.10]
    data = np.array([PER_TASK_S1[lam] for lam in lambdas])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0.006, vmax=0.021)
    ax.set_xticks(np.arange(len(TASK_LABELS)))
    ax.set_xticklabels(TASK_LABELS)
    ax.set_yticks(np.arange(len(lambdas)))
    ax.set_yticklabels([f"{lam:.2f}" for lam in lambdas])
    ax.set_xlabel("Task")
    ax.set_ylabel(r"$\lambda_{\mathrm{TAR}}$")
    ax.set_title("Per-Task Intra-Chunk Smoothness (S1) — lower is smoother")
    for i in range(len(lambdas)):
        for j in range(len(TASK_LABELS)):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=7,
                    color="black" if data[i, j] > 0.012 else "white")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="S1")
    _save(fig, "fig4_per_task_s1")
    plt.close(fig)


def main() -> None:
    print(f"Generating figures -> {FIG_DIR}")
    fig1_overview()
    fig2_per_task()
    fig3_ablation()
    fig4_per_task_s1()
    print("Done.")


if __name__ == "__main__":
    main()
