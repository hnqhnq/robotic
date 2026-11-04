# 2-Minute Poster Teaser

**Title:** TAR: Temporal Action Regularization for Smooth Action Chunking in VLA Models  
**Presenter:** Ningqiu He — Macao Polytechnic University  



## Script

Good afternoon. If you work with vision–language–action models, you have probably seen this pattern: the robot looks fine *inside* each action chunk, but the motion **jerks** every time a new eight-step prediction starts. That is not a hardware bug—it is a training gap. Chunks are predicted independently, so nothing in the standard L1 loss asks the model to **slow down at chunk boundaries**.

We call our fix **TAR—Temporal Action Regularization**. It is deliberately minimal: one extra term that penalizes high velocity on the **first and last steps** of each predicted chunk. Think “slow start, slow finish,” so the next chunk can meet the previous one smoothly. About **five lines of code**, **no new parameters**, and **zero cost at inference**.

We integrated TAR into **BitVLA**, a 1.58-bit VLA on **LIBERO-10**, with full fine-tuning and a four-point ablation over the regularization weight λ. The headline number: success rate rises from **86.4% to 93.6%** at λ = 0.10, while intra-chunk smoothness metrics improve across the board—without temporal ensembling or a heavier controller.

On the poster you will see **why** boundary jumps dominate baseline errors, the **exact loss**, the **λ ablation table**, per-task gains, and a trajectory comparison. We also discuss when stronger regularization helps generalization even if training loss looks worse.

If you care about reliable long-horizon manipulation with chunked VLAs, **I hope you will join me in the poster session**—I would welcome your questions on λ choice and whether TAR transfers to other chunk lengths or models. Thank you.

