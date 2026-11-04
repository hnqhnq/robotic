# Assignment 2 — Conference Poster (BitVLA-TAR)

**Option A:** Doctoral research — TAR on BitVLA + LIBERO-10.

## Deliverables

| File | Description |
|------|-------------|
| [`poster.html`](poster.html) | A0 **landscape** poster (1189 × 841 mm), English, body text ≥ 24 pt |
| [`poster_teaser.md`](poster_teaser.md) | ~2-minute flash talk script (poster advertisement) |
| [`poster.pdf`](poster.pdf) | PDF export (generate with commands below if not present) |

## View / print the poster

1. Open `poster.html` in Chrome or Edge.
2. **Print** → Destination: **Save as PDF**.
3. Paper size: **A0**, Orientation: **Landscape**, Margins: **None**, Background graphics: **On**.

### Export PDF (recommended)

Headless Chrome’s `--print-to-pdf` defaults to **US Letter** and will split/crop the poster. Use the CDP script instead (single **A0 landscape** page):

```bash
cd /home/src/robotic/bitvla-tar/assignment2/out
python3 export_poster_pdf.py
```

This writes both `poster.pdf` and `zh/poster.pdf`.

### Manual print from browser

If you print from Chrome/Edge yourself: paper **A0**, orientation **Landscape**, margins **None**, background graphics **On**. Do not use the default Letter size.

## Design notes (assignment checklist)

- **Size:** A0 landscape — stated in poster header.
- **Language:** English; written for VLA/robotics specialists and adjacent fields.
- **Hierarchy:** Title → Motivation → Method → Results → Takeaways → Contact.
- **Typography:** CSS uses 24–28 pt body, 44–72 pt headings (readable at ~1 m / ~1.5–2 m).

## Source material

Content drawn from `bitvla-tar/report/` (导师汇报, v4 ablation, method note) and `doc/paper/`.

## Chinese reference (查阅)

Synced Chinese copies: [`zh/README.md`](zh/README.md), [`zh/poster.html`](zh/poster.html), [`zh/poster_teaser.md`](zh/poster_teaser.md), [`zh/poster.pdf`](zh/poster.pdf). **Submit the English poster for the course.**
