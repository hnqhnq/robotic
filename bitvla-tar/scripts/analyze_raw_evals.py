#!/usr/bin/env python3
"""Analyze per-episode raw eval data (runs/eval_logs/<label>/raw/*.npz, written by run_libero_eval_bitnet.py).

All action metrics are on the EXECUTED action sequence (unnormalized training-space actions before gripper
binarization), so they are comparable across execution modes (chunk / blend / te) and horizons.

  exec_s1          mean |a_{t+1} - a_t| over all steps and 7 dims
  exec_jerk_rms    RMS of the second difference a_{t+1} - 2 a_t + a_{t-1}, arm dims 0-5
  replan_jump      mean |a_t - a_{t-1}| (dims 0-5) at steps where a new chunk starts being executed
  interior_jump    same at all other steps;  replan/interior ratio ~ IB/S1 of the 20261022 report
  replan_jerk      SEAM-style: mean |2nd diff| (dims 0-5) centred on replanning steps;  interior_jerk elsewhere
  gripper_sw       gripper switches per 100 steps with threshold 0.5 (training space 1 = open, 0 = close)
  stall_frac       fraction of steps inside "stalls": windows of STALL_WIN steps in which the end effector moved
                   less than STALL_DIST metres
  max_stall        longest stall (steps)

Usage:
  python scripts/analyze_raw_evals.py e3-tar-h8 e3-base-h8 ...        # labels under runs/eval_logs/
  python scripts/analyze_raw_evals.py --out report/data/track1_raw_metrics.json e3-*
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs/eval_logs"
STALL_WIN, STALL_DIST = 20, 0.01


def stall_mask(pos):
    n = len(pos)
    m = np.zeros(n, dtype=bool)
    for s in range(0, max(n - STALL_WIN, 0)):
        if np.linalg.norm(pos[s + STALL_WIN] - pos[s]) < STALL_DIST:
            m[s:s + STALL_WIN + 1] = True
    return m


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def episode_metrics(d):
    a = d["exec_actions"]
    n = len(a)
    out = {"task_id": int(d["task_id"]), "episode_idx": int(d["episode_idx"]), "success": bool(d["success"]),
           "steps": n, "num_queries": int(len(d["query_steps"]))}
    if n < 3:
        return out
    d1 = np.abs(np.diff(a, axis=0))                     # d1[t-1] = |a_t - a_{t-1}|,  t = 1..n-1
    d2 = np.abs(a[2:] - 2 * a[1:-1] + a[:-2])           # d2[t-1] centred on t,       t = 1..n-2
    exec_start = np.zeros(n, dtype=bool)
    if str(d["execution_mode"]) == "te":
        exec_start[:] = True                            # every step is a re-plan
    else:
        exec_start[np.asarray(d["query_steps"], dtype=int)] = True
    replan = exec_start[1:]                             # aligned with d1
    out.update(
        exec_s1=float(d1.mean()),
        exec_jerk_rms=float(np.sqrt((d2[:, :6] ** 2).mean())),
        replan_jump=float(d1[replan, :6].mean()) if replan.any() else float("nan"),
        interior_jump=float(d1[~replan, :6].mean()) if (~replan).any() else float("nan"),
    )
    rj = exec_start[1:-1]
    out["replan_jerk"] = float(d2[rj, :6].mean()) if rj.any() else float("nan")
    out["interior_jerk"] = float(d2[~rj, :6].mean()) if (~rj).any() else float("nan")
    g = (a[:, 6] > 0.5).astype(int)
    out["gripper_sw_per100"] = float(np.abs(np.diff(g)).sum() * 100.0 / n)
    sm = stall_mask(d["eef_pos"].astype(np.float64))
    out["stall_frac"] = float(sm.mean())
    out["max_stall"] = int(longest_run(sm))
    return out


def summarize(eps):
    keys = ["exec_s1", "exec_jerk_rms", "replan_jump", "interior_jump", "replan_jerk", "interior_jerk",
            "gripper_sw_per100", "stall_frac", "max_stall", "steps", "num_queries"]
    s = {"n": len(eps), "successes": int(sum(e["success"] for e in eps))}
    s["success_rate"] = s["successes"] / max(s["n"], 1)
    for k in keys:
        v = [e[k] for e in eps if k in e and np.isfinite(e[k])]
        s[k] = float(np.mean(v)) if v else float("nan")
    if np.isfinite(s["replan_jump"]) and s["interior_jump"] > 0:
        s["replan_over_interior_jump"] = s["replan_jump"] / s["interior_jump"]
    for tag, sel in (("succ", True), ("fail", False)):
        sub = [e for e in eps if e["success"] == sel and "stall_frac" in e]
        if sub:
            s[f"stall_frac_{tag}"] = float(np.mean([e["stall_frac"] for e in sub]))
            s[f"exec_s1_{tag}"] = float(np.mean([e["exec_s1"] for e in sub]))
    per_task = {}
    for e in eps:
        per_task.setdefault(e["task_id"], []).append(e["success"])
    s["per_task_success"] = {int(k): float(np.mean(v)) for k, v in sorted(per_task.items())}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = {}
    for label in args.labels:
        files = sorted(glob.glob(str(LOGS / label / "raw" / "*.npz")))
        if not files:
            print(f"[skip] {label}: no npz")
            continue
        eps = [episode_metrics(np.load(f)) for f in files]
        result[label] = {"summary": summarize(eps), "episodes": eps}
        s = result[label]["summary"]
        print(f"{label:22s} n={s['n']:3d} SR={s['success_rate']*100:5.1f}%  exec_s1={s['exec_s1']:.4f} "
              f"jerk={s['exec_jerk_rms']:.4f} replan/int={s.get('replan_over_interior_jump', float('nan')):.2f} "
              f"gripSw/100={s['gripper_sw_per100']:.2f} stall={s['stall_frac']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
