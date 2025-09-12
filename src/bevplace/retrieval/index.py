from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import faiss
import numpy as np

from bevplace.core.types import Pose2D


@dataclass
class RetrievalHit:
    id: int
    distance: float


class BEVIndex:
    """Descriptor index with optional PCA reduction and FAISS Flat L2 search.

    API:
      - fit_pca(X)
      - add(X, poses)
      - search(Q, k)
      - save(dir) / load(dir)
    """

    def __init__(self, pca_dim: Optional[int] = None) -> None:
        self.pca_dim = pca_dim
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None  # [pca_dim, d]
        self._index: Optional[faiss.IndexFlatL2] = None
        self._ids: List[int] = []
        self._poses: List[Optional[Pose2D]] = []
        self._next_id: int = 0

    @property
    def dim(self) -> Optional[int]:
        if self._index is None:
            return None
        return self._index.d

    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        if self.pca_dim is None:
            return X.astype(np.float32, copy=False)
        if self._pca_components is None or self._pca_mean is None:
            raise RuntimeError("PCA not fitted; call fit_pca first or set pca_dim=None")
        Xc = (X - self._pca_mean).astype(np.float32, copy=False)
        return Xc @ self._pca_components.T  # [n, pca_dim]

    def fit_pca(self, X: np.ndarray) -> None:
        if self.pca_dim is None:
            return
        X = np.asarray(X, dtype=np.float32)
        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean
        # Economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.pca_dim, Vt.shape[0])
        self._pca_components = Vt[:k, :].copy().astype(np.float32)
        self._pca_mean = mean.astype(np.float32)

    def add(self, X: np.ndarray, poses: Optional[Iterable[Optional[Pose2D]]]) -> None:
        X = np.asarray(X, dtype=np.float32)
        Xp = self._apply_pca(X)
        d = Xp.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatL2(d)
        assert self._index.d == d, "Descriptor dimension mismatch"
        self._index.add(Xp)
        n = X.shape[0]
        if poses is None:
            poses = [None] * n
        for p in poses:
            self._ids.append(self._next_id)
            self._poses.append(p)
            self._next_id += 1

    def search(self, Q: np.ndarray, k: int = 1) -> List[List[RetrievalHit]]:
        if self._index is None:
            raise RuntimeError("Index is empty")
        Q = np.asarray(Q, dtype=np.float32)
        Qp = self._apply_pca(Q)
        distances, indices = self._index.search(Qp, k)
        results: List[List[RetrievalHit]] = []
        for row_dist, row_idx in zip(distances, indices):
            hits = [RetrievalHit(id=int(idx), distance=float(dist)) for dist, idx in zip(row_dist, row_idx)]
            results.append(hits)
        return results

    def save(self, out_dir: Path | str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            raise RuntimeError("Cannot save empty index")
        faiss.write_index(self._index, str(out / "faiss.index"))
        if self.pca_dim is not None and self._pca_components is not None and self._pca_mean is not None:
            np.savez(out / "pca.npz", components=self._pca_components, mean=self._pca_mean)
        meta = {
            "pca_dim": self.pca_dim,
            "num_items": len(self._ids),
        }
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: Path | str) -> "BEVIndex":
        src = Path(in_dir)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        idx = cls(pca_dim=meta.get("pca_dim"))
        idx._index = faiss.read_index(str(src / "faiss.index"))
        pca_path = src / "pca.npz"
        if pca_path.exists():
            data = np.load(pca_path)
            idx._pca_components = data["components"].astype(np.float32)
            idx._pca_mean = data["mean"].astype(np.float32)
        # Recreate ids
        n = meta.get("num_items", idx._index.ntotal)
        idx._ids = list(range(n))
        idx._next_id = n
        idx._poses = [None] * n
        return idx
