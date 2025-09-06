from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Match = Tuple[int, int, float]


def _l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, C], b: [M, C] -> [N, M]
    aa = (a**2).sum(axis=1, keepdims=True)
    bb = (b**2).sum(axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(aa + bb - 2 * ab, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def match_descriptors(
    desc_q: np.ndarray,
    desc_m: np.ndarray,
    ratio: float = 0.8,
    mutual: bool = True,
    max_matches: Optional[int] = None,
) -> List[Match]:
    """Match descriptors using L2 with ratio test and optional mutual check.

    Args:
        desc_q: [N, C]
        desc_m: [M, C]
        ratio: Lowe's ratio threshold.
        mutual: if True, require mutual nearest neighbors.
        max_matches: optional cap on number of returned matches (best-first).
    Returns:
        List of (i, j, distance) matches.
    """
    if desc_q.size == 0 or desc_m.size == 0:
        return []
    d = _l2_dist(desc_q.astype(np.float32), desc_m.astype(np.float32))  # [N, M]
    # For each query, find best and second-best
    idx_sorted = np.argsort(d, axis=1)
    best = idx_sorted[:, 0]
    second = idx_sorted[:, 1] if d.shape[1] > 1 else idx_sorted[:, 0]
    d_best = d[np.arange(d.shape[0]), best]
    d_second = d[np.arange(d.shape[0]), second] + 1e-12
    keep = d_best / d_second <= ratio

    candidates: List[Match] = [
        (int(i), int(j), float(db)) for i, j, db, k in zip(np.arange(d.shape[0]), best, d_best, keep) if k
    ]

    if mutual and candidates:
        # Build reverse nearest neighbor map
        d_mq = d.T  # [M, N]
        m_best = np.argmin(d_mq, axis=1)
        candidates = [m for m in candidates if m_best[m[1]] == m[0]]

    # Sort by distance ascending and cap
    candidates.sort(key=lambda t: t[2])
    if max_matches is not None:
        candidates = candidates[:max_matches]
    return candidates
