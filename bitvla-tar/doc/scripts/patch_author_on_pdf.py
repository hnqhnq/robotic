#!/usr/bin/env python3
"""Patch author / affiliation / email on the finalized TAR PDF.

Keeps layout, figures, and page breaks identical to the original; only the
title-block author lines are white-out + rewritten (centered, same as原稿).

Usage:
    python doc/scripts/patch_author_on_pdf.py
    python doc/scripts/patch_author_on_pdf.py --output doc/TAR-NingqiuHe-MPU.pdf
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore  # pymupdf < 1.24

# Match原稿 title-block bbox (points, page width 612)
AUTHOR_BLOCK = fitz.Rect(84, 143, 528, 186)
FONT_SIZE = 10.96  # NimbusRomNo9L-Regu body text in原稿
SUPERSCRIPT_SIZE = 7.67  # affiliation markers ¹, *
SUPERSCRIPT_RAISE = 3.96  # baseline offset (原稿: 155.86 vs 151.90)

# (text, fontsize, is_superscript) — line 1 matches原稿: authors + affiliation
AUTHOR_LINE1: list[tuple[str, float, bool]] = [
    ("Ningqiu He", FONT_SIZE, False),
    ("1", SUPERSCRIPT_SIZE, True),
    (" and ", FONT_SIZE, False),
    ("Chi Kin Lam", FONT_SIZE, False),
    ("1,*", SUPERSCRIPT_SIZE, True),
    (" ", FONT_SIZE, False),
    ("1", SUPERSCRIPT_SIZE, True),
    ("Macao Polytechnic University, Macao, China", FONT_SIZE, False),
]
AUTHOR_LINE2 = "Email: hnq0824@gmail.com; cklamsta@mpu.edu.mo"
AUTHOR_LINE3: list[tuple[str, float, bool]] = [
    ("*", SUPERSCRIPT_SIZE, True),
    ("Corresponding author (Supervisor)", FONT_SIZE, False),
]
# Baselines copied from原稿 page 1 (tiro font)
BASELINES = [158.5, 171.9, 185.2]


def _text_width(font: fitz.Font, text: str, size: float) -> float:
    return font.text_length(text, fontsize=size)


def _insert_centered_plain(page: fitz.Page, font: fitz.Font, text: str, baseline_y: float) -> None:
    x = (page.rect.width - _text_width(font, text, FONT_SIZE)) / 2
    tw = fitz.TextWriter(page.rect, color=(0, 0, 0))
    tw.append((x, baseline_y), text, font=font, fontsize=FONT_SIZE)
    tw.write_text(page)


def _insert_centered_segments(
    page: fitz.Page,
    font: fitz.Font,
    segments: list[tuple[str, float, bool]],
    baseline_y: float,
) -> None:
    total = sum(_text_width(font, t, s) for t, s, _ in segments)
    x = (page.rect.width - total) / 2
    tw = fitz.TextWriter(page.rect, color=(0, 0, 0))
    for text, size, is_super in segments:
        y = baseline_y - SUPERSCRIPT_RAISE if is_super else baseline_y
        tw.append((x, y), text, font=font, fontsize=size)
        x += _text_width(font, text, size)
    tw.write_text(page)


def patch(src: Path, dst: Path) -> None:
    doc = fitz.open(src)
    page = doc[0]
    font = fitz.Font("tiro")  # Times-like, matches IEEE PDF

    page.add_redact_annot(AUTHOR_BLOCK, fill=(1, 1, 1))
    page.apply_redactions()

    _insert_centered_segments(page, font, AUTHOR_LINE1, BASELINES[0])
    _insert_centered_plain(page, font, AUTHOR_LINE2, BASELINES[1])
    _insert_centered_segments(page, font, AUTHOR_LINE3, BASELINES[2])

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    doc.close()
    print(f"Wrote {dst}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_src = root / "TAR-NingqiuHe.pdf"
    default_dst = root / "TAR-NingqiuHe-MPU.pdf"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_src)
    parser.add_argument("--output", type=Path, default=default_dst)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input (creates .bak backup first)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input PDF not found: {args.input}")

    if args.in_place:
        backup = args.input.with_suffix(".pdf.bak")
        shutil.copy2(args.input, backup)
        patch(args.input, args.input)
        print(f"Backup: {backup}")
    else:
        patch(args.input, args.output)


if __name__ == "__main__":
    main()
