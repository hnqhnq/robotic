#!/usr/bin/env python3
"""E6: rule-based failure mode tags on saved eval npz (track 1).

Uses the same stall definition as analyze_raw_evals.py. Primary tags (failed episodes only):
  timeout      ran until step budget (~LIBERO-10 520)
  stall_heavy  stall_frac >= STALL_HEAVY (default 0.35)
  gripper      gripper_sw_per100 >= GRIP_THR and not stall_heavy
  other        remaining failures

Paired view: episodes with matching task_id + episode_idx across two labels.

Usage:
  python scripts/classify_failures_e6.py --pair e3-base-h8 e3-tar-h8
  python scripts/classify_failures_e6.py --labels e3-base-h8 e3-tar-h8 e4-base-h4 e4-tar-h4 \\
      --out report/data/e6_failure_modes.json --md report/data/e6_failure_modes.md
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs/eval_logs"
STALL_WIN, STALL_DIST = 20, 0.01
LIBERO10_MAX_STEPS = 520
TIMEOUT_MIN_STEPS = 500
STALL_HEAVY = 0.35
GRIP_THR = 8.0  # switches per 100 steps


def stall_mask(pos: np.ndarray) -> np.ndarray:
    n = len(pos)
    m = np.zeros(n, dtype=bool)
    for s in range(0, max(n - STALL_WIN, 0)):
        if np.linalg.norm(pos[s + STALL_WIN] - pos[s]) < STALL_DIST:
            m[s : s + STALL_WIN + 1] = True
    return m


def episode_row(path: Path) -> dict:
    d = np.load(path)
    a = d["exec_actions"]
    n = len(a)
    g = (a[:, 6] > 0.5).astype(int) if n else np.zeros(0, dtype=int)
    grip_sw = float(np.abs(np.diff(g)).sum() * 100.0 / n) if n > 1 else 0.0
    sm = stall_mask(d["eef_pos"].astype(np.float64)) if n else np.zeros(0, dtype=bool)
    stall_frac = float(sm.mean()) if n else 0.0
    # longest_run inline
    best = cur = 0
    for v in sm:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    max_stall = best
    ok = bool(d["success"])
    tag = "success"
    if not ok:
        if n >= TIMEOUT_MIN_STEPS:
            tag = "timeout_stall" if stall_frac >= STALL_HEAVY else "timeout"
        elif stall_frac >= STALL_HEAVY:
            tag = "stall_heavy"
        elif grip_sw >= GRIP_THR:
            tag = "gripper"
        else:
            tag = "other"
    return {
        "file": str(path.name),
        "task_id": int(d["task_id"]),
        "episode_idx": int(d["episode_idx"]),
        "success": ok,
        "tag": tag,
        "steps": int(n),
        "stall_frac": stall_frac,
        "max_stall": int(max_stall),
        "gripper_sw_per100": grip_sw,
    }


def load_label(label: str) -> dict[tuple[int, int], dict]:
    files = sorted(glob.glob(str(LOGS / label / "raw" / "*.npz")))
    out = {}
    for f in files:
        row = episode_row(Path(f))
        out[(row["task_id"], row["episode_idx"])] = row
    return out


def summarize_failures(rows: list[dict]) -> dict:
    fails = [r for r in rows if not r["success"]]
    c = Counter(r["tag"] for r in fails)
    return {
        "n_episodes": len(rows),
        "n_fail": len(fails),
        "tags": dict(c),
        "tag_frac_of_fail": {k: v / max(len(fails), 1) for k, v in c.items()},
        "mean_stall_on_fail": float(np.mean([r["stall_frac"] for r in fails])) if fails else 0.0,
    }


def pair_table(base_label: str, tar_label: str) -> dict:
    b = load_label(base_label)
    t = load_label(tar_label)
    keys = sorted(set(b) & set(t))
    rows = []
    for k in keys:
        rows.append({"key": k, "base": b[k], "tar": t[k]})
    cross = Counter()
    for r in rows:
        if r["base"]["success"] and r["tar"]["success"]:
            cross["both_ok"] += 1
        elif not r["base"]["success"] and r["tar"]["success"]:
            cross["tar_only"] += 1
        elif r["base"]["success"] and not r["tar"]["success"]:
            cross["base_only"] += 1
        else:
            cross["both_fail"] += 1
            cross[f"fail_{r['base']['tag']}|{r['tar']['tag']}"] += 1
    return {
        "base": base_label,
        "tar": tar_label,
        "paired_n": len(keys),
        "outcome": dict(cross),
        "base_fail_summary": summarize_failures([r["base"] for r in rows]),
        "tar_fail_summary": summarize_failures([r["tar"] for r in rows]),
    }


def render_md(payload: dict) -> str:
    lines = ["# E6 失败模式（自动生成）", ""]
    for label, block in payload.get("per_label", {}).items():
        s = block["summary"]
        lines.append(f"## {label}")
        lines.append(f"- episodes: {s['n_episodes']}, fail: {s['n_fail']}")
        lines.append(f"- fail tags: {s['tags']}")
        lines.append(f"- mean stall (fail): {s['mean_stall_on_fail']:.3f}")
        lines.append("")
    for pair in payload.get("pairs", []):
        lines.append(f"## 配对 {pair['base']} vs {pair['tar']}")
        lines.append(f"- paired n: {pair['paired_n']}")
        lines.append(f"- outcomes: {pair['outcome']}")
        lines.append(f"- baseline fail tags: {pair['base_fail_summary']['tags']}")
        lines.append(f"- TAR fail tags: {pair['tar_fail_summary']['tags']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=["e3-base-h8", "e3-tar-h8"])
    ap.add_argument("--pair", nargs=2, action="append", metavar=("BASE", "TAR"))
    ap.add_argument("--out", default="report/data/e6_failure_modes.json")
    ap.add_argument("--md", default="report/data/e6_failure_modes.md")
    args = ap.parse_args()
    pairs = args.pair or [["e3-base-h8", "e3-tar-h8"], ["e4-base-h4", "e4-tar-h4"], ["e4-base-h2", "e4-tar-h2"]]

    payload = {"per_label": {}, "pairs": []}
    for label in args.labels:
        epmap = load_label(label)
        rows = list(epmap.values())
        payload["per_label"][label] = {"summary": summarize_failures(rows)}
        print(f"{label}: n={len(rows)} fails={payload['per_label'][label]['summary']['n_fail']} "
              f"tags={payload['per_label'][label]['summary']['tags']}")

    for bl, tl in pairs:
        try:
            pt = pair_table(bl, tl)
            payload["pairs"].append(pt)
            print(f"pair {bl} vs {tl}: both_fail={pt['outcome'].get('both_fail', 0)} "
                  f"tar fail tags={pt['tar_fail_summary']['tags']}")
        except Exception as e:
            print(f"[warn] pair {bl}/{tl}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)
    if args.md:
        Path(args.md).write_text(render_md(payload), encoding="utf-8")
        print("wrote", args.md)


if __name__ == "__main__":
    main()