from __future__ import annotations
import math
from typing import Iterable, Tuple, List
import numpy as np

MP_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

LIMBS: List[Tuple[str, str]] = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
LIMB_IDXS = np.array([(MP_LANDMARKS[a], MP_LANDMARKS[b]) for a, b in LIMBS])

# ---------------------------------------------------------------------------
# Helper functions（已改为 2D）
# ---------------------------------------------------------------------------
def _unit_vectors(points: np.ndarray, idx_pairs: np.ndarray) -> np.ndarray:
    """返回单位向量 (M, 2)"""
    vecs = points[idx_pairs[:, 1]] - points[idx_pairs[:, 0]]  # (M, 2)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms

def _angle_diffs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u,v 形状 (M,2)，返回对应向量夹角 (M,)（弧度）"""
    cos = np.sum(u * v, axis=1).clip(-1.0, 1.0)
    return np.arccos(cos)

def pose_similarity_2d(
    landmarks_a: Iterable[Iterable[float]],
    landmarks_b: Iterable[Iterable[float]],
    limb_weights: List[float] | None = None,
    aggregate: str = "mean",
) -> float:

    a = np.asarray(landmarks_a, dtype=np.float32)
    b = np.asarray(landmarks_b, dtype=np.float32)

    if a.shape != (33, 2) or b.shape != (33, 2):
        raise ValueError("Each landmark set must have shape (33, 2)")

    u = _unit_vectors(a, LIMB_IDXS)
    v = _unit_vectors(b, LIMB_IDXS)

    angles = _angle_diffs(u, v)          # ∈ [0, π]
    sims = 1.0 - angles / math.pi        # 归一化到 [0,1]

    if limb_weights is not None:
        limb_weights = np.asarray(limb_weights, dtype=np.float32)
        if limb_weights.shape != (len(LIMB_IDXS),):
            raise ValueError(f"limb_weights must have length {len(LIMB_IDXS)}")
        sims *= limb_weights
        normaliser = limb_weights.sum()
    else:
        normaliser = len(LIMB_IDXS)

    if aggregate == "mean":
        score = sims.sum() / normaliser
    elif aggregate == "median":
        score = float(np.median(sims))
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    return float(score)

# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 生成两组随机 (33,2) 数据做演示
    score = pose_similarity_2d(lm1, lm2)
    print(f"Similarity (arms & legs, 2‑D): {score:.3f}")


