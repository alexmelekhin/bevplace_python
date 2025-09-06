from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _svd_icp(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rigid transform between 2D point sets using SVD (no scale).

    Args:
        src: [N,2]
        dst: [N,2]
    Returns:
        R: [2,2], t: [2]
    """
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    mu_src = src.mean(axis=0, keepdims=True)
    mu_dst = dst.mean(axis=0, keepdims=True)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = (mu_dst.T - R @ mu_src.T).ravel()
    return R.astype(np.float32), t.astype(np.float32)


def estimate_rigid_transform_ransac(
    pts_q: np.ndarray,
    pts_m: np.ndarray,
    pixel_thresh: float = 1.0,
    max_iters: int = 2000,
    confidence: float = 0.99,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Estimate 2D rigid transform with RANSAC.

    Args:
        pts_q: [N,2] points in query
        pts_m: [N,2] corresponding points in matched map image
        pixel_thresh: inlier threshold in pixels
        max_iters: number of random hypotheses
        confidence: not used directly (placeholder for future adaptive RANSAC)
    Returns:
        T: [3,3] SE(2) matrix; report: dict with inliers, iterations
    """
    pts_q = np.asarray(pts_q, dtype=np.float32)
    pts_m = np.asarray(pts_m, dtype=np.float32)
    N = min(pts_q.shape[0], pts_m.shape[0])
    if N < 2:
        return np.eye(3, dtype=np.float32), {"inliers": 0, "iters": 0}

    best_inliers = 0
    best_R = np.eye(2, dtype=np.float32)
    best_t = np.zeros(2, dtype=np.float32)

    rng = np.random.default_rng()
    for it in range(max_iters):
        idx = rng.choice(N, size=2, replace=False)
        R, t = _svd_icp(pts_q[idx], pts_m[idx])
        pred = (pts_q @ R.T) + t
        err = np.linalg.norm(pred - pts_m, axis=1)
        inliers_mask = err <= pixel_thresh
        num_in = int(inliers_mask.sum())
        if num_in > best_inliers:
            best_inliers = num_in
            best_R, best_t = R, t

    # Refine with all inliers from the best model (recompute mask)
    pred = (pts_q @ best_R.T) + best_t
    err = np.linalg.norm(pred - pts_m, axis=1)
    inliers_mask = err <= pixel_thresh
    if inliers_mask.sum() >= 2:
        best_R, best_t = _svd_icp(pts_q[inliers_mask], pts_m[inliers_mask])

    T = np.eye(3, dtype=np.float32)
    T[:2, :2] = best_R
    T[:2, 2] = best_t
    report = {"inliers": int(best_inliers), "iters": int(max_iters)}
    return T, report
