#!/usr/bin/env bash
# E9 helper: scan LaTeX/paper for known inconsistent claims (no GPU).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/report/data/e9_paper_audit_$(date +%Y%m%d).md"
PAPER="$ROOT/doc/paper"
mkdir -p report/data
{
  echo "# E9 文稿口径扫描 $(date -Iseconds)"
  echo ""
  echo "Patterns to fix manually (§1.4 I1–I6):"
  echo ""
  for pat in '首个' '消除' '边界跳变' 'first to identify' 'eliminating boundary'; do
    echo "## pattern: $pat"
    rg -n "$pat" "$PAPER" 2>/dev/null || echo "(no match in report/paper)"
    echo ""
  done
  echo "## E5 numbers reminder (paste into tables)"
  echo "- chunk h8 paired: +4.8 pp (E3)"
  echo "- blend500 paired: +4.2 pp p=0.033"
  echo "- TE screen: base 68.0% vs tar 72.7% (150 ep, tasks 4/6/8)"
} >"$OUT"
echo "wrote $OUT"