from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bevplace.cli.index_build import build_index
from bevplace.cli.localize import run_localize


def _make_bin(path: Path, n: int = 1000) -> None:
    pts = np.zeros((n, 4), dtype=np.float32)
    pts[:, 0:3] = np.random.randn(n, 3).astype(np.float32) * 10.0
    path.write_bytes(pts.tobytes())


def test_index_build_outputs(tmp_path: Path):
    # Create small dataset of BIN files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(5):
        _make_bin(data_dir / f"{i:06d}.bin", n=200)

    db_dir = tmp_path / "db"
    build_index(map_dir=data_dir, out_dir=db_dir, ext="bin", pca_dim=16, quiet=True)

    # Check artifacts
    assert (db_dir / "faiss.index").exists()
    assert (db_dir / "pca.npz").exists()
    assert (db_dir / "items.jsonl").exists()
    assert (db_dir / "bev_params.json").exists()
    meta = json.loads((db_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_items"] == 5


def test_index_build_store_locals(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        _make_bin(data_dir / f"{i:06d}.bin", n=150)

    db_dir = tmp_path / "db"
    build_index(map_dir=data_dir, out_dir=db_dir, ext="bin", pca_dim=8, quiet=True, store_locals=True)

    locals_dir = db_dir / "locals"
    assert locals_dir.exists()
    # Expect 3 locals npz files
    files = sorted(locals_dir.glob("*.npz"))
    assert len(files) == 3
    # Validate one file structure
    arr = np.load(files[0])["data"]
    assert arr.dtype == np.float32
    assert arr.ndim == 3  # [C,H,W]


def test_localize_retrieval_only(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(4):
        _make_bin(data_dir / f"{i:06d}.bin", n=180)

    db_dir = tmp_path / "db"
    build_index(map_dir=data_dir, out_dir=db_dir, ext="bin", pca_dim=8, quiet=True)

    # Prepare args namespace for retrieval only
    class Args:
        bin = str(data_dir / "000002.bin")
        pcd = None
        db = str(db_dir)
        map_dir = None
        estimate_pose = False
        topk = 3
        device = "cpu"
        D = 40.0
        g = 0.4
        out_json = None
        quiet = True

    run_localize(Args)
    # If no exception, consider pass; retrieval may be arbitrary due to random weights


def test_localize_pose_with_locals(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        _make_bin(data_dir / f"{i:06d}.bin", n=160)

    db_dir = tmp_path / "db"
    build_index(map_dir=data_dir, out_dir=db_dir, ext="bin", pca_dim=8, quiet=True, store_locals=True)

    out_json = tmp_path / "res.json"

    class Args:
        pass

    args = Args()
    args.bin = str(data_dir / "000001.bin")
    args.pcd = None
    args.db = str(db_dir)
    args.map_dir = None  # not needed when locals stored
    args.estimate_pose = True
    args.topk = 1
    args.device = "cpu"
    args.D = 40.0
    args.g = 0.4
    args.out_json = str(out_json)
    args.quiet = True

    run_localize(args)
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "matched_id" in data and "topk" in data and "distances" in data
    assert "pose_relative" in data and isinstance(data["pose_relative"], list)
