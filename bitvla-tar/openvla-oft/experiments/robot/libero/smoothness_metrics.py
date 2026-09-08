"""
Action smoothness metrics for TAR evaluation (paper Section III-B).

Metrics computed on predicted action chunks during LIBERO rollouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeSmoothness:
    chunks: List[np.ndarray] = field(default_factory=list)
    success: bool = False


class SmoothnessTracker:
    """Accumulates per-episode action chunks and computes aggregate smoothness metrics."""

    def __init__(self) -> None:
        self.episodes: List[EpisodeSmoothness] = []
        self._current = EpisodeSmoothness()

    def start_episode(self) -> None:
        self._current = EpisodeSmoothness()

    def add_chunk(self, chunk: np.ndarray) -> None:
        self._current.chunks.append(np.asarray(chunk, dtype=np.float64))

    def end_episode(self, success: bool) -> None:
        self._current.success = bool(success)
        self.episodes.append(self._current)

    @staticmethod
    def chunk_s1(chunk: np.ndarray) -> float:
        """Mean intra-chunk step-to-step absolute variation."""
        if chunk.shape[0] < 2:
            return 0.0
        return float(np.abs(np.diff(chunk, axis=0)).mean())

    @staticmethod
    def chunk_s2(chunk: np.ndarray) -> float:
        """Mean second-order variation within a chunk."""
        if chunk.shape[0] < 3:
            return 0.0
        first_order = np.diff(chunk, axis=0)
        return float(np.abs(np.diff(first_order, axis=0)).mean())

    @staticmethod
    def chunk_max_jump(chunk: np.ndarray) -> float:
        """Peak step-to-step absolute velocity within a chunk."""
        if chunk.shape[0] < 2:
            return 0.0
        return float(np.abs(np.diff(chunk, axis=0)).max())

    @staticmethod
    def inter_boundary_jump(prev_last: np.ndarray, next_first: np.ndarray) -> float:
        """Absolute jump between consecutive chunks."""
        return float(np.abs(next_first - prev_last).mean())

    @staticmethod
    def chunk_gripper_switches(chunk: np.ndarray, threshold: float = 0.0) -> int:
        """Count gripper open/close transitions within a chunk (last action dim)."""
        if chunk.shape[0] < 2:
            return 0
        gripper = (chunk[:, -1] > threshold).astype(np.int32)
        return int(np.sum(np.abs(np.diff(gripper)) > 0))

    def _episode_metrics(self, episode: EpisodeSmoothness) -> Dict[str, float]:
        chunks = episode.chunks
        if not chunks:
            return {}

        s1_vals = [self.chunk_s1(c) for c in chunks]
        s2_vals = [self.chunk_s2(c) for c in chunks]
        max_jump_vals = [self.chunk_max_jump(c) for c in chunks]
        gripper_vals = [self.chunk_gripper_switches(c) for c in chunks]

        ib_vals: List[float] = []
        for i in range(1, len(chunks)):
            ib_vals.append(self.inter_boundary_jump(chunks[i - 1][-1], chunks[i][0]))

        metrics = {
            "s1": float(np.mean(s1_vals)),
            "s2": float(np.mean(s2_vals)),
            "max_jump": float(np.mean(max_jump_vals)),
            "gripper_switches": float(np.mean(gripper_vals)),
        }
        if ib_vals:
            metrics["inter_boundary"] = float(np.mean(ib_vals))
        return metrics

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate metrics across all recorded episodes."""
        if not self.episodes:
            return {}

        per_episode = [self._episode_metrics(ep) for ep in self.episodes]
        per_episode = [m for m in per_episode if m]
        if not per_episode:
            return {}

        keys = per_episode[0].keys()
        overall = {k: float(np.mean([m[k] for m in per_episode if k in m])) for k in keys}

        success_eps = [
            self._episode_metrics(ep) for ep in self.episodes if ep.success and ep.chunks
        ]
        fail_eps = [
            self._episode_metrics(ep) for ep in self.episodes if not ep.success and ep.chunks
        ]

        if success_eps:
            overall["success_s1"] = float(np.mean([m["s1"] for m in success_eps]))
        if fail_eps:
            overall["fail_s1"] = float(np.mean([m["s1"] for m in fail_eps]))

        overall["num_episodes"] = len(self.episodes)
        overall["num_successes"] = sum(int(ep.success) for ep in self.episodes)
        return overall

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregate": self.aggregate(),
            "episodes": [
                {
                    "success": ep.success,
                    "num_chunks": len(ep.chunks),
                    **self._episode_metrics(ep),
                }
                for ep in self.episodes
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
