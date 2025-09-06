from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np

from bevplace.core.types import BEVParams


def discover_files(root: Path, exts: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*.{ext}"))
    files = sorted([p for p in files if p.is_file()])
    return files


def load_bin_kitti(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.float32)
    if data.size % 4 == 0:
        pts = data.reshape(-1, 4)[:, :3]
    elif data.size % 3 == 0:
        pts = data.reshape(-1, 3)
    else:
        raise ValueError(f"{path} has unexpected BIN size: {data.size}")
    return pts


def load_pcd_open3d(path: Path) -> np.ndarray:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("open3d is required for .pcd reading") from exc
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


def read_items(db_dir: Path) -> List[Dict]:
    items_path = db_dir / "items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"items.jsonl not found in {db_dir}")
    items: List[Dict] = []
    with items_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_bev_params(db_dir: Path, fallback: BEVParams) -> BEVParams:
    p = db_dir / "bev_params.json"
    if not p.exists():
        return fallback
    data = json.loads(p.read_text(encoding="utf-8"))
    return BEVParams(D=float(data.get("D", fallback.D)), g=float(data.get("g", fallback.g)))
