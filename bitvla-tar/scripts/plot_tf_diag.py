#!/usr/bin/env python3
"""Figure for E2: per-position step-to-step change inside the predicted chunk (teacher forcing on demo frames).
Reads report/data/tf_diag_{base,tar}.json, writes report/figures/fig5_tf_diag_profile.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
base = json.load(open(ROOT / "report/data/tf_diag_base.json"))
tar = json.load(open(ROOT / "report/data/tf_diag_tar.json"))

x = list(range(1, 8))
labels = [f"{i - 1}→{i}" for i in x]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

ax = axes[0]
ax.plot(x, base["demo"]["step_diff_profile_norm_arm6"], "k--o", label="Demo")
ax.plot(x, base["pred"]["step_diff_profile_norm_arm6"], "-o", color="tab:blue", label="Baseline (official)")
ax.plot(x, tar["pred"]["step_diff_profile_norm_arm6"], "-o", color="tab:red", label="TAR λ=0.10")
ax.set_xticks(x, labels)
ax.set_xlabel("step pair inside the 8-step chunk")
ax.set_ylabel("mean |a(k+1) − a(k)|  (normalized, arm dims 0–5)")
ax.set_title("(a) Where inside the chunk the actions change")
ax.set_ylim(0, None)
ax.legend()

ax = axes[1]
names = ["Demo", "Baseline", "TAR"]
vals = [base["demo"]["tar_norm"], base["pred"]["tar_norm"], tar["pred"]["tar_norm"]]
l1 = [0.0, base["l1_norm"], tar["l1_norm"]]
pos = range(3)
ax.bar([p - 0.2 for p in pos], vals, width=0.4, color=["0.4", "tab:blue", "tab:red"], label="TAR term (start+end)")
ax.bar([p + 0.2 for p in pos], l1, width=0.4, color=["0.8", "lightsteelblue", "lightsalmon"],
       label="L1 error vs demo chunk")
for p, v, e in zip(pos, vals, l1):
    ax.text(p - 0.2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    if e > 0:
        ax.text(p + 0.2, e, f"{e:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(pos), names)
ax.set_title(f"(b) Teacher forcing on {tar['n_samples']} demo frames")
ax.legend(fontsize=8)

fig.tight_layout()
out = ROOT / "report/figures/fig5_tf_diag_profile.png"
fig.savefig(out, dpi=150)
print("wrote", out)
