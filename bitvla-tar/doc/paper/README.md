# TAR 论文 LaTeX 源码

本目录由定稿 PDF [`../TAR-NingqiuHe.pdf`](../TAR-NingqiuHe.pdf) 反推重建，便于后续修改文字/数据。

> **汇报 / 投稿若需与定稿版式完全一致**：请直接用定稿 PDF，仅改作者信息：
>
> ```bash
> pip install pymupdf
> python ../scripts/patch_author_on_pdf.py
> # 输出 ../TAR-NingqiuHe-MPU.pdf
> ```
>
> LaTeX 重建版（`main.pdf`）图表内容一致，但页码与浮动体位置会与定稿不同，**不必强求一致**。

## 目录结构

```text
paper/
├── main.tex
├── sections/ / tables/ / figures/
└── scripts/
    ├── extract_figures_from_pdf.py   # 从定稿 PDF 裁剪插图
    ├── build_paper.sh
    └── generate_figures.py           # 备选：从实验数据重绘
```

## 作者信息

| 角色 | 英文 | 邮箱 |
|------|------|------|
| 第一作者 | Ningqiu He | hnq0824@gmail.com |
| 导师 / 通讯作者 | Chi Kin Lam | cklamsta@mpu.edu.mo |
| 单位 | Macao Polytechnic University | — |

- **定稿 PDF 改作者**：`doc/scripts/patch_author_on_pdf.py`
- **LaTeX 改作者**：编辑 `main.tex` 的 `\author{...}` 后重新编译

## 编译 LaTeX

```bash
cd doc/paper
bash scripts/build_paper.sh
# 或：python scripts/extract_figures_from_pdf.py && latexmk -pdf -g main.tex
```

改 `figures/*.png` 后须加 `-g` 强制重编译，否则可能仍显示旧图。

## 插图（方案 B）

```bash
python scripts/extract_figures_from_pdf.py
```

从 `../TAR-NingqiuHe.pdf` 裁剪 Fig.1–4；坐标见 `extract_figures_from_pdf.py` 中的 `CROPS`。

## 相关文档

| 内容 | 路径 |
|------|------|
| 定稿 PDF | `../TAR-NingqiuHe.pdf` |
| 定稿 + MPU 作者 | `../TAR-NingqiuHe-MPU.pdf`（运行 patch 脚本后） |
| 复现指南 | `../REPRODUCE.md` |
| 汇报说明 | `../汇报说明-数据与模型缺失.md` |
