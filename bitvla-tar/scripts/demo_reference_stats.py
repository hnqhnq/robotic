#!/usr/bin/env python3
"""Reference smoothness statistics of the human demonstrations (CPU only).

Needs /tmp/libero10_actions.pkl (scripts/extract_demo_actions.py) and a trained run dir for dataset_statistics.json.
Writes report/data/demo_reference_stats.json.

Two spaces (important - they are NOT comparable with each other):
  * normalized space  : what the TAR training loss sees (dims 0-5 -> [-1,1] with q01/q99; gripper in {0,1} unnormalized)
  * real action space : what the eval smoothness metrics see (training-space actions, i.e. gripper +1 open / 0 close)
Gripper follows libero_dataset_transform: raw -1(open)..1(close) -> clip(0,1) -> flip.
"""
import glob, json, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = 8
eps = pickle.load(open("/tmp/libero10_actions.pkl", "rb"))
st = json.load(open(glob.glob(str(ROOT / "runs/*v4-tar01*10000_chkpt/dataset_statistics.json"))[0]))["libero_10_no_noops"]
q01, q99, mask = (np.array(st["action"][k]) for k in ("q01", "q99", "mask"))


def train_space(a):
    a = a.copy(); a[:, -1] = 1 - np.clip(a[:, -1], 0, 1); return a


def norm(a):
    return np.where(mask, np.clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1), a)


tar_dims, s1n = [], []
for e in eps:  # (A) normalized space, sliding windows fully inside the episode (training-style)
    a = norm(train_space(e["action"]))
    for t in range(0, len(a) - T + 1):
        c = a[t:t + T]
        tar_dims.append(np.abs(c[1] - c[0]) + np.abs(c[-1] - c[-2]))
        s1n.append(np.abs(np.diff(c, axis=0)).mean())
tar_dims = np.array(tar_dims)

S1, S2, MJ, IB, GS = [], [], [], [], []
for e in eps:  # (B) real space, eval-style: non-overlapping 8-step chunks from t=0
    a = train_space(e["action"])
    chunks = [a[i:i + T] for i in range(0, len(a) - T + 1, T)]
    s1, s2, mj, gs, ib = [], [], [], [], []
    for c in chunks:
        d = np.diff(c, axis=0)
        s1.append(np.abs(d).mean()); s2.append(np.abs(np.diff(d, axis=0)).mean()); mj.append(np.abs(d).max())
        gs.append(np.abs(np.diff((c[:, -1] > 0).astype(int))).sum())
    for i in range(1, len(chunks)):
        ib.append(np.abs(chunks[i][0] - chunks[i - 1][-1]).mean())
    S1.append(np.mean(s1)); S2.append(np.mean(s2)); MJ.append(np.mean(mj)); GS.append(np.mean(gs)); IB.append(np.mean(ib))

out = dict(gt_tar_norm=float(tar_dims.mean()), gt_tar_norm_per_dim=tar_dims.mean(0).tolist(), gt_s1_norm=float(np.mean(s1n)),
           gt_real=dict(s1=float(np.mean(S1)), s2=float(np.mean(S2)), max_jump=float(np.mean(MJ)),
                        inter_boundary=float(np.mean(IB)), gripper_switches=float(np.mean(GS))),
           n_windows=int(len(tar_dims)), n_episodes=len(eps),
           note="gripper converted as in libero_dataset_transform (+1 open, 0 close); BOUNDS_Q99 on dims 0-5")
(ROOT / "report/data").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "report/data/demo_reference_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
