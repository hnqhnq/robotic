#!/usr/bin/env bash
# Extract figures from the finalized PDF, then force-rebuild main.pdf.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/extract_figures_from_pdf.py
# -g: force rebuild (latexmk may skip when only figures/*.png changed)
latexmk -pdf -g -interaction=nonstopmode main.tex

echo ""
echo "[build_paper] Done: $ROOT/main.pdf"
echo "[build_paper] If the viewer still shows old figures, close the PDF tab and reopen."
