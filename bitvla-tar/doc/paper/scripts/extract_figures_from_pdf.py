#!/usr/bin/env python3
"""Extract paper figures from the finalized PDF (Option B).

Crops Fig.1–4 from ``doc/TAR-NingqiuHe.pdf`` at 200 DPI and writes PNGs to
``doc/paper/figures/``. Coordinates are tuned for the IEEE two-column layout at
that resolution (page size 1700×2200 px).

Usage:
    python scripts/extract_figures_from_pdf.py
    python scripts/extract_figures_from_pdf.py --pdf ../TAR-NingqiuHe.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

# (page_number, (left, top, right, bottom)) at 200 DPI
CROPS: dict[str, tuple[int, tuple[int, int, int, int]]] = {
    "fig1_overview": (1, (850, 535, 1650, 900)),
    "fig2_per_task": (5, (95, 95, 872, 478)),
    "fig3_ablation": (5, (885, 95, 1695, 475)),
    "fig4_per_task_s1": (5, (848, 640, 1690, 938)),
}

DPI = 200


def render_pages(pdf: Path, out_dir: Path, pages: set[int]) -> dict[int, Path]:
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found; install poppler-utils.")

    lo, hi = min(pages), max(pages)
    prefix = out_dir / "page"
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(DPI),
            "-f",
            str(lo),
            "-l",
            str(hi),
            str(pdf),
            str(prefix),
        ],
        check=True,
    )

    rendered: dict[int, Path] = {}
    for page in pages:
        path = out_dir / f"page-{page}.png"
        if not path.exists():
            sys.exit(f"Expected rendered page missing: {path}")
        rendered[page] = path
    return rendered


def extract(pdf: Path, figures_dir: Path) -> None:
    pages = {page for page, _ in CROPS.values()}
    figures_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rendered = render_pages(pdf, tmp_dir, pages)

        for name, (page, box) in CROPS.items():
            out = figures_dir / f"{name}.png"
            Image.open(rendered[page]).crop(box).save(out)
            print(f"wrote {out}  crop={box}  size={Image.open(out).size}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    paper_dir = script_dir.parent
    default_pdf = paper_dir.parent / "TAR-NingqiuHe.pdf"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=default_pdf,
        help=f"Source PDF (default: {default_pdf})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paper_dir / "figures",
        help="Output directory for PNG figures",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    extract(args.pdf.resolve(), args.out.resolve())
    print("\nNext: latexmk -pdf -g main.tex   (or: bash scripts/build_paper.sh)")


if __name__ == "__main__":
    main()
