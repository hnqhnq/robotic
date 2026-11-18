#!/usr/bin/env python3
"""E2: teacher-forcing diagnostic (no simulator).

Feeds demonstration frames (both cameras + 8-d state + instruction) of libero_10_no_noops through the same inference
path as the LIBERO eval, and compares the predicted 8-step chunk with the demo chunk at that frame.

Spaces (same conventions as scripts/demo_reference_stats.py):
  * normalized : dims 0-5 -> [-1,1] with q01/q99 of the v4-tar01 run (shared by both models), gripper in [0,1]
                 -> this is the space of the TAR training loss
  * real       : unnormalized training-space actions (gripper 1 = open, 0 = close)

Usage (GPU, from repo root, bitvla env):
  python scripts/teacher_forcing_diag.py --model tar  --stride 20
  python scripts/teacher_forcing_diag.py --model base --stride 20
Outputs report/data/tf_diag_<model>.npz and .json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
OFT = ROOT / "openvla-oft"
sys.path.insert(0, str(OFT))
sys.path.insert(0, str(OFT / "bitvla"))
os.chdir(OFT)

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

from experiments.robot.libero.run_libero_eval_bitnet import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import get_action
from bitvla.constants import (
    BITNET_ACTION_TOKEN_BEGIN_IDX,
    BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    BITNET_IGNORE_INDEX,
    BITNET_PROPRIO_PAD_IDX,
    BITNET_STOP_INDEX,
)

CKPTS = {
    "tar": "/root/autodl-tmp/robotic-data/runs/bitvla-bf16+libero_10_no_noops+b16+lr-0.0004--image_aug--v4-tar01"
           "--image_aug+tar-0.1--10000_chkpt/lora_adapter",
    "base": "/root/robotic/bitvla-tar/checkpoints/ft-bitvla-libero-long",
}
T = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CKPTS), required=True)
    ap.add_argument("--stride", type=int, default=20, help="sample every k-th frame of every demo")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    st = json.load(open(glob.glob(str(ROOT / "runs/*v4-tar01*10000_chkpt/dataset_statistics.json"))[0]))
    st = st["libero_10_no_noops"]["action"]
    q01, q99, mask = np.array(st["q01"]), np.array(st["q99"]), np.array(st["mask"])

    def norm(a):
        return np.where(mask, np.clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1), a)

    def train_space(a):
        a = a.copy()
        a[:, -1] = 1 - np.clip(a[:, -1], 0, 1)
        return a

    cfg = GenerateConfig(
        model_family="bitnet",
        pretrained_checkpoint=CKPTS[args.model],
        task_suite_name="libero_10",
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        center_crop=True,
        use_wandb=False,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )

    builder = tfds.builder_from_directory(str(ROOT / "data/modified_libero_rlds/libero_10_no_noops/1.0.0"))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    preds, gts, ep_ids, steps = [], [], [], []
    t0 = time.time()
    for ep_i, ep in enumerate(ds):
        frames = list(ep["steps"])
        actions = train_space(np.stack([f["action"].numpy() for f in frames]))
        instr = frames[0]["language_instruction"].numpy().decode()
        for t in range(0, len(frames) - T + 1, args.stride):
            ob = frames[t]["observation"]
            observation = {
                "full_image": resize_image_for_policy(ob["image"].numpy(), 224),
                "wrist_image": resize_image_for_policy(ob["wrist_image"].numpy(), 224),
                "state": ob["state"].numpy().astype(np.float64),
            }
            chunk = np.asarray(get_action(cfg, model, observation, instr, processor=processor,
                                          action_head=action_head, proprio_projector=proprio_projector,
                                          noisy_action_projector=noisy_action_projector, use_film=False))
            preds.append(chunk)
            gts.append(actions[t:t + T])
            ep_ids.append(ep_i)
            steps.append(t)
            if args.max_samples and len(preds) >= args.max_samples:
                break
        print(f"ep {ep_i}: {len(preds)} samples, {time.time() - t0:.0f}s", flush=True)
        if args.max_samples and len(preds) >= args.max_samples:
            break

    P, G = np.stack(preds), np.stack(gts)          # (N, 8, 7) real space
    Pn, Gn = norm(P), norm(G)

    def stats(x):
        d = np.abs(np.diff(x, axis=1))             # (N, 7, 7): |a_{k+1} - a_k|
        tar = d[:, 0] + d[:, -1]                   # (N, 7)
        return {
            "tar_norm": float(tar.mean()),
            "tar_norm_per_dim": tar.mean(0).tolist(),
            "tar_norm_arm6": float(tar[:, :6].mean() * 6 / 7),
            "tar_norm_gripper": float(tar[:, 6].mean() / 7),
            "gripper_share_of_tar": float(tar[:, 6].mean() / tar.sum(1).mean()),
            "v_start_norm": float(d[:, 0].mean()),
            "v_end_norm": float(d[:, -1].mean()),
            "step_diff_profile_norm": d.mean(axis=(0, 2)).tolist(),        # 7 positions in the chunk
            "step_diff_profile_norm_arm6": d[:, :, :6].mean(axis=(0, 2)).tolist(),
            "s1_norm": float(d.mean()),
        }

    out = {
        "model": args.model,
        "checkpoint": CKPTS[args.model],
        "n_samples": int(len(P)),
        "stride": args.stride,
        "pred": stats(Pn),
        "demo": stats(Gn),
        "pred_s1_real": float(np.abs(np.diff(P, axis=1)).mean()),
        "demo_s1_real": float(np.abs(np.diff(G, axis=1)).mean()),
        "l1_norm": float(np.abs(Pn - Gn).mean()),
        "l1_norm_per_dim": np.abs(Pn - Gn).mean(axis=(0, 1)).tolist(),
        "l1_norm_per_step": np.abs(Pn - Gn).mean(axis=(0, 2)).tolist(),
        "seconds": time.time() - t0,
    }
    (ROOT / "report/data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / f"report/data/tf_diag_{args.model}.npz", pred=P, demo=G,
                        episode=np.array(ep_ids), step=np.array(steps))
    json.dump(out, open(ROOT / f"report/data/tf_diag_{args.model}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
