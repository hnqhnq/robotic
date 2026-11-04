# 作业 2 — 会议海报（BitVLA-TAR）中文查阅版

> **说明：** 课程提交仍以英文版为准（上级目录 `../poster.html`、`../poster.pdf`、`../poster_teaser.md`）。本目录为与英文内容同步的中文对照，便于汇报前查阅与演练。

**方案 A：** 博士研究 — 在 BitVLA + LIBERO-10 上的 TAR 工作。

## 文件清单

| 文件 | 说明 |
|------|------|
| [`poster.html`](poster.html) | A0 **横版**海报中文版（1189 × 841 mm），正文 ≥ 24 pt |
| [`poster_teaser.md`](poster_teaser.md) | 约 2 分钟快闪推介稿（中文口播版 + 英文对照索引） |
| [`poster.pdf`](poster.pdf) | 由 HTML 导出的 PDF（可选重新生成，见下） |

## 查看 / 打印

1. 用 Chrome 或 Edge 打开 `poster.html`。
2. **打印** → 目标：**另存为 PDF**。
3. 纸张：**A0**，方向：**横向**，边距：**无**，背景图形：**开启**。

### 导出 PDF（推荐）

勿用 `google-chrome --print-to-pdf`（默认为 Letter，会裁切/分页）。请在 **`out/` 目录**运行：

```bash
cd /home/src/robotic/bitvla-tar/assignment2/out
python3 export_poster_pdf.py
```

会同时生成英文与中文 `poster.pdf`（单页 A0 横版）。

## 与作业要求对照

- **尺寸：** A0 横版（海报页眉已标注）。
- **语言：** 本目录为中文；**正式交作业请用英文版**（`assignment2.md` 要求英文海报）。
- **层次：** 动机 → 方法 → 实验设置 → 主结果 → 结论与后续。
- **字体：** 正文 24 pt 起，标题更大；中文使用系统黑体/微软雅黑等 sans-serif。

## 内容来源

`bitvla-tar/report/`（导师汇报、v4 消融、方法说明）与 `doc/paper/`。
