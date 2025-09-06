from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bevplace.core.types import Pose2D
from bevplace.retrieval.index import BEVIndex


@pytest.mark.parametrize("use_pca", [False, True])
def test_index_add_and_search(use_pca: bool):
    rng = np.random.default_rng(42)
    d = 128
    n = 100
    descs = rng.normal(size=(n, d)).astype(np.float32)
    poses = [Pose2D(x_m=float(i), y_m=0.0, yaw_rad=0.0) for i in range(n)]

    index = BEVIndex(pca_dim=16 if use_pca else None)
    if use_pca:
        index.fit_pca(descs)
    index.add(descs, poses=poses)

    # Query a few existing vectors; nearest should be itself (distance ~ 0)
    q = descs[:5]
    hits = index.search(q, k=1)
    assert len(hits) == 5
    for i, hit_list in enumerate(hits):
        assert hit_list[0].id == i
        assert hit_list[0].distance <= 1e-5


def test_index_save_and_load(tmp_path: Path):
    rng = np.random.default_rng(0)
    d = 64
    n = 50
    descs = rng.normal(size=(n, d)).astype(np.float32)

    idx_dir = tmp_path / "db"
    index = BEVIndex(pca_dim=16)
    index.fit_pca(descs)
    index.add(descs, poses=None)
    index.save(idx_dir)

    # Basic artifacts exist
    assert (idx_dir / "faiss.index").exists()
    assert (idx_dir / "pca.npz").exists()
    assert (idx_dir / "meta.json").exists()

    # Reload and search
    loaded = BEVIndex.load(idx_dir)
    q = descs[:3]
    hits = loaded.search(q, k=1)
    for i, hit_list in enumerate(hits):
        assert hit_list[0].id == i
        assert hit_list[0].distance <= 1e-5
