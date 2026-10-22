#!/usr/bin/env python3
"""Statistics for two LIBERO eval logs (pure stdlib, CPU only).

Parses per-episode success from EVAL-*.txt logs, then reports:
  * Wilson 95% CI for each model
  * Paired comparison (same task + same episode index => same fixed initial state):
    exact McNemar test, paired bootstrap CI for the success-rate difference
  * Unpaired two-proportion z-test (for reference)
  * Per-task table

Usage:
  python scripts/eval_stats.py BASELINE_LOG_OR_GLOB TAR_LOG_OR_GLOB   # globs merge sharded logs
  python scripts/eval_stats.py --default      # uses the 2026-10 v4-tar01 / baseline-long logs
  python scripts/eval_stats.py --summary 'runs/eval_logs/LABEL/EVAL-*.txt'   # one model, merges shards
"""
import argparse
import glob
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "runs/eval_logs/baseline-long/EVAL-*--baseline-long.txt"
DEFAULT_TAR = "runs/eval_logs/v4-tar01/EVAL-*14_01_22--v4-tar01.txt"


def parse_logs(pattern_or_path):
    """Parse one or several (sharded) EVAL logs; returns ({(task_idx, episode_idx): success}, [task names]).

    Task index comes from the "Task id: N" line (new logs). Old logs have no such line, so we fall back to
    order of appearance within the file. Later files override earlier ones for duplicate keys.
    """
    files = sorted(glob.glob(str(pattern_or_path))) or [str(pattern_or_path)]
    results, names = {}, {}
    for path in files:
        cur_name, cur_task, cur_ep, appear = None, None, None, []
        for line in Path(path).read_text(errors="ignore").splitlines():
            if line.startswith("Task:"):
                cur_name = line[5:].strip()
                if not appear or appear[-1] != cur_name:
                    appear.append(cur_name)
                cur_task = len(appear) - 1  # fallback; overridden by "Task id:" if present
            m = re.match(r"Task id: (\d+)", line)
            if m:
                cur_task = int(m.group(1))
            m = re.match(r"Starting episode (\d+)", line)
            if m:
                cur_ep = int(m.group(1))
            m = re.match(r"Success: (True|False)", line)
            if m and cur_task is not None and cur_ep is not None:
                results[(cur_task, cur_ep)] = m.group(1) == "True"
                names[cur_task] = cur_name
    return results, [names[k] for k in sorted(names)], files


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided(b, c):
    """Exact two-sided McNemar p-value = binomial test with p=0.5 on b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?")
    ap.add_argument("tar", nargs="?")
    ap.add_argument("--default", action="store_true")
    ap.add_argument("--summary", metavar="LOG_GLOB", help="single-model summary (merges shard logs); no comparison")
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    if args.summary:
        r, names, files = parse_logs(args.summary)
        n = len(r)
        k = sum(r.values())
        lo, hi = wilson(k, n) if n else (0, 0)
        print(f"files merged: {len(files)}   episodes: {n}")
        print(f"success: {k}/{n} = {k / max(n, 1):.1%}   Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
        for t in sorted({key[0] for key in r}):
            e = [v for key, v in r.items() if key[0] == t]
            print(f"  task {t}: {sum(e)}/{len(e)} = {sum(e) / len(e):.0%}")
        return 0
    if args.default:
        args.base = str(ROOT / DEFAULT_BASE)
        args.tar = str(ROOT / DEFAULT_TAR)
    if not (args.base and args.tar):
        ap.error("need BASE and TAR logs or --default")

    rb, tb, fb = parse_logs(args.base)
    rt, tt, ft = parse_logs(args.tar)
    keys = sorted(set(rb) & set(rt))
    n = len(keys)
    print(f"baseline logs: {fb}\n  episodes parsed: {len(rb)}")
    print(f"tar logs:      {ft}\n  episodes parsed: {len(rt)}")
    print(f"paired episodes: {n}")
    if tb != tt:
        print("WARNING: task order differs between logs")

    kb = sum(rb[k] for k in keys)
    kt = sum(rt[k] for k in keys)
    lb, ub = wilson(kb, n)
    lt, ut = wilson(kt, n)
    print("\n== Per-model success (Wilson 95% CI) ==")
    print(f"baseline: {kb}/{n} = {kb/n:.1%}  CI [{lb:.1%}, {ub:.1%}]")
    print(f"TAR     : {kt}/{n} = {kt/n:.1%}  CI [{lt:.1%}, {ut:.1%}]")

    both = sum(rb[k] and rt[k] for k in keys)
    only_t = sum((not rb[k]) and rt[k] for k in keys)  # TAR wins
    only_b = sum(rb[k] and (not rt[k]) for k in keys)  # baseline wins
    neither = n - both - only_t - only_b
    print("\n== Paired contingency (same task + same init state) ==")
    print(f"both success={both}  TAR only={only_t}  baseline only={only_b}  both fail={neither}")
    p_mc = binom_two_sided(only_t, only_b)
    print(f"exact McNemar two-sided p = {p_mc:.4f}")

    diff = (kt - kb) / n
    # paired bootstrap, resampling episodes (stratified by task to keep 50/task)
    rng = random.Random(0)
    by_task = {}
    for k in keys:
        by_task.setdefault(k[0], []).append(int(rt[k]) - int(rb[k]))
    ds = []
    for _ in range(args.boot):
        tot = 0
        for v in by_task.values():
            tot += sum(rng.choices(v, k=len(v)))
        ds.append(tot / n)
    ds.sort()
    lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]
    print(f"diff = {diff:+.1%}   paired bootstrap 95% CI [{lo:+.1%}, {hi:+.1%}]")

    pp = (kb + kt) / (2 * n)
    se = math.sqrt(pp * (1 - pp) * 2 / n)
    z = diff / se
    print(f"(reference) unpaired two-proportion z = {z:.2f}, one-sided p = {norm_sf(z):.4f}")

    print("\n== Per task (n=50 each) ==")
    print(f"{'#':>2} {'base':>6} {'tar':>6} {'delta':>7}  TAR-only/base-only  task")
    for t in sorted(by_task):
        ks = [k for k in keys if k[0] == t]
        a = sum(rb[k] for k in ks)
        b = sum(rt[k] for k in ks)
        w = sum((not rb[k]) and rt[k] for k in ks)
        l = sum(rb[k] and (not rt[k]) for k in ks)
        print(f"{t+1:>2} {a/len(ks):>6.0%} {b/len(ks):>6.0%} {(b-a)/len(ks):>+7.0%}  {w:>8}/{l:<9}  {tb[t][:50]}")

    print("\n== Reading guide ==")
    print("* A paired CI excluding 0 / McNemar p<0.05 => the same-environment TAR>baseline gap")
    print("  is unlikely to be eval sampling noise. It does NOT capture training-seed variance.")


if __name__ == "__main__":
    sys.exit(main())
