#!/usr/bin/env python3
"""Generate figures for report/20261022-TAR方法说明与重构实验计划.md  (CPU only).

Real-data figures read from files; the concept figure is explicitly illustrative.
Needs /tmp/libero10_actions.pkl (python scripts/extract_demo_actions.py) and report/data/*.csv|json
(scripts/demo_reference_stats.py).
Usage:  python scripts/make_report_figures.py
"""
import csv
import glob
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C_DEMO, C_BASE, C_TAR = "#7f7f7f", "#d62728", "#1f77b4"


def load_smooth(pattern):
    p = glob.glob(str(ROOT / pattern))[0]
    return json.load(open(p))


# ---------------------------------------------------------------- Fig 1: concept (real GT chunk + illustrative jump)
def fig_concept():
    eps = pickle.load(open("/tmp/libero10_actions.pkl", "rb"))
    st = json.load(open(glob.glob(str(ROOT / "runs/*v4-tar01*10000_chkpt/dataset_statistics.json"))[0]))["libero_10_no_noops"]
    q01, q99 = np.array(st["action"]["q01"]), np.array(st["action"]["q99"])
    a = eps[0]["action"][60:76].copy()  # two consecutive 8-step chunks of a real demo (raw dataset space)
    a = np.clip(2 * (a[:, :6] - q01[:6]) / (q99[:6] - q01[:6]) - 1, -1, 1)
    dim = 2  # z translation, clearly visible motion
    gt = a[:, dim]
    c1, c2 = gt[:8], gt[8:]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    # left: one chunk with boundary velocities
    x = np.arange(8)
    ax[0].plot(x, c1, "o-", color=C_DEMO, label="one 8-step chunk (real demo, z-translation, normalized)")
    ax[0].annotate("", xy=(1, c1[1]), xytext=(0, c1[0]), arrowprops=dict(arrowstyle="->", color="k", lw=2))
    ax[0].annotate("", xy=(7, c1[7]), xytext=(6, c1[6]), arrowprops=dict(arrowstyle="->", color="k", lw=2))
    ax[0].text(0.15, c1[0] - 0.12, r"$v_{start}=\hat a_1-\hat a_0$", fontsize=9)
    ax[0].text(3.9, c1[7] + 0.1, r"$v_{end}=\hat a_{T-1}-\hat a_{T-2}$", fontsize=9)
    ax[0].set_xlabel("step inside chunk (t=0..7)")
    ax[0].set_ylabel("normalized action value")
    ax[0].set_title("(a) What TAR penalizes: first-step and last-step velocity")
    ax[0].legend(fontsize=7, loc="lower left")
    # right: illustrative boundary between chunks
    xs1, xs2 = np.arange(0, 8), np.arange(8, 16)
    jump = 0.25
    ax[1].plot(xs1, c1, "o-", color=C_DEMO, label="chunk k")
    ax[1].plot(xs2, c2 + jump, "s--", color=C_BASE, label="chunk k+1 (ILLUSTRATIVE: shifted by +0.25 to mimic a boundary jump)")
    ax[1].plot([7, 8], [c1[7], c2[0] + jump], color="k", lw=2)
    ax[1].text(5.0, (c1[7] + c2[0] + jump) / 2 - 0.12, "inter-boundary jump (IB)", fontsize=8)
    ax[1].set_xlabel("global step (chunks are executed open-loop, then re-planned)")
    ax[1].set_title("(b) Chunk boundary (IB) - illustrative")
    ax[1].legend(fontsize=7, loc="lower left")
    for a_ in ax:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_tar_concept.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- Fig 2: training curves
def fig_train():
    rows = list(csv.DictReader(open(ROOT / "report/data/v4tar01_train_curve.csv")))
    s = np.array([float(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    curr = np.array([float(r["curr_action_l1"]) for r in rows])
    nxt = np.array([float(r["next_actions_l1"]) for r in rows])
    tar = np.array([float(r["tar_loss"]) for r in rows])
    l1 = (curr + 7 * nxt) / 8
    k = 10
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(s, loss, alpha=0.25, color=C_TAR)
    ax[0].plot(s[k - 1:], sm(loss), color=C_TAR, label="total loss = L1 + 0.1 * TAR")
    ax[0].plot(s[k - 1:], sm(l1), color="g", lw=1.5, ls="--", label="L1 part only (nearly overlaps: TAR is ~1.4% of loss)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("loss (log)")
    ax[0].set_title("(a) lambda=0.10 training loss (v4-tar01)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(s, tar, alpha=0.25, color=C_BASE)
    ax[1].plot(s[k - 1:], sm(tar), color=C_BASE, label="TAR term (before x lambda)")
    ref = json.load(open(ROOT / "report/data/demo_reference_stats.json"))["gt_tar_norm"]
    ax[1].axhline(ref, color=C_DEMO, ls="--", label=f"same quantity on demo actions = {ref:.4f}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("training step")
    ax[1].set_title("(b) TAR term vs. the value on ground-truth demos")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_training_curves.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- Fig 3: smoothness demo vs baseline vs TAR
def fig_smooth():
    b = load_smooth("runs/eval_logs/baseline-long/smoothness_*.json")["aggregate"]
    t = load_smooth("runs/eval_logs/v4-tar01/smoothness_*.json")["aggregate"]
    d = json.load(open(ROOT / "report/data/demo_reference_stats.json"))["gt_real"]
    keys = [("s1", "S1 (within-chunk speed)"), ("s2", "S2 (within-chunk accel.)"), ("max_jump", "MaxJump"),
            ("inter_boundary", "IB (chunk boundary jump)"), ("gripper_switches", "Gripper switches / chunk")]
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.4))
    for a_, (k, name) in zip(ax, keys):
        vals = [d[k], b[k], t[k]]
        bars = a_.bar(["demo", "baseline", "TAR"], vals, color=[C_DEMO, C_BASE, C_TAR])
        for bar, v in zip(bars, vals):
            a_.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}" if v < 1 else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        a_.set_title(name, fontsize=9)
        a_.tick_params(axis="x", labelsize=8)
    plt.suptitle("Action-chunk smoothness in real action space (lower = smoother). demo = 379 human demos, eval-style chunks", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_smoothness_demo_base_tar.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- Fig 4: per-task success and S1
def fig_task():
    bj = load_smooth("runs/eval_logs/baseline-long/smoothness_*.json")["episodes"]
    tj = load_smooth("runs/eval_logs/v4-tar01/smoothness_*.json")["episodes"]
    sb = [sum(e["success"] for e in bj[i * 50:(i + 1) * 50]) / 50 * 100 for i in range(10)]
    st = [sum(e["success"] for e in tj[i * 50:(i + 1) * 50]) / 50 * 100 for i in range(10)]
    s1b = [np.mean([e["s1"] for e in bj[i * 50:(i + 1) * 50]]) for i in range(10)]
    s1t = [np.mean([e["s1"] for e in tj[i * 50:(i + 1) * 50]]) for i in range(10)]
    x = np.arange(10)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].bar(x - 0.2, sb, 0.4, color=C_BASE, label="baseline (83.6%)")
    ax[0].bar(x + 0.2, st, 0.4, color=C_TAR, label="TAR lambda=0.10 (88.4%)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"T{i + 1}" for i in x])
    ax[0].set_ylabel("success rate (%) / 50 trials")
    ax[0].set_title("(a) per-task success")
    ax[0].set_ylim(0, 118)
    ax[0].legend(fontsize=8, loc="upper center", ncol=2)
    ax[1].bar(x - 0.2, s1b, 0.4, color=C_BASE, label="baseline")
    ax[1].bar(x + 0.2, s1t, 0.4, color=C_TAR, label="TAR")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"T{i + 1}" for i in x])
    ax[1].set_ylabel("S1 (lower = smoother)")
    ax[1].set_title("(b) per-task S1")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_per_task.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_concept()
    fig_train()
    fig_smooth()
    fig_task()
    print("figures ->", OUT)
