from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from bevplace import REIN
from bevplace.core.types import BEVParams
from bevplace.preprocess.bev import bev_density_image_torch
from bevplace.retrieval import BEVIndex
from bevplace.cli.common import discover_files, load_bin_kitti, load_pcd_open3d


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Build BEVPlace index from a directory of point clouds")
    parser.add_argument("--map-dir", required=True, help="Directory with .pcd/.bin files")
    parser.add_argument("--out", required=True, help="Output directory for index")
    parser.add_argument("--ext", choices=["pcd", "bin"], default=None, help="Restrict to a single extension")
    parser.add_argument("--poses", default=None, help="Optional poses file (csv/json/jsonl)")
    parser.add_argument("--pca-dim", type=int, default=512, help="PCA target dimension; 0 disables PCA")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--D", type=float, default=40.0, help="BEV half-size in meters")
    parser.add_argument("--g", type=float, default=0.4, help="BEV grid size in meters per pixel")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pca_dim = None if args.pca_dim == 0 else args.pca_dim

    build_index(
        map_dir=args.map_dir,
        out_dir=args.out,
        ext=args.ext,
        poses_path=args.poses,
        pca_dim=pca_dim,
        device=device,
        D=args.D,
        g=args.g,
        quiet=args.quiet,
    )


def _parse_poses(path: Path, map_dir: Path) -> Dict[str, Tuple[float, float, float]]:
    poses: Dict[str, Tuple[float, float, float]] = {}
    if not path.exists():
        return poses
    if path.suffix.lower() in {".jsonl"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rel = str(Path(obj["file"]))
                pose = tuple(float(x) for x in obj["pose"])  # type: ignore
                poses[rel] = (pose[0], pose[1], pose[2])
    elif path.suffix.lower() in {".json"}:
        obj = json.loads(path.read_text(encoding="utf-8"))
        for item in obj.get("items", []):
            rel = str(Path(item["file"]))
            pose = tuple(float(x) for x in item["pose"])  # type: ignore
            poses[rel] = (pose[0], pose[1], pose[2])
    elif path.suffix.lower() in {".csv"}:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = {h.lower() for h in reader.fieldnames or []}
            use_deg = "yaw_deg" in headers and "yaw_rad" not in headers
            for row in reader:
                rel = str(Path(row["file"]))
                x = float(row["x_m"])
                y = float(row["y_m"])
                if use_deg:
                    yaw = float(row["yaw_deg"]) * np.pi / 180.0
                else:
                    yaw = float(row.get("yaw_rad", 0.0))
                poses[rel] = (x, y, yaw)
    else:
        raise ValueError(f"Unsupported pose file format: {path}")
    return poses


def build_index(
    map_dir: str | Path,
    out_dir: str | Path,
    ext: Optional[str] = None,
    poses_path: Optional[str | Path] = None,
    pca_dim: Optional[int] = 512,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    D: float = 40.0,
    g: float = 0.4,
    quiet: bool = False,
) -> None:
    t0 = time.perf_counter()
    map_path = Path(map_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exts = [ext] if ext else ["pcd", "bin"]
    files = discover_files(map_path, exts)
    if not files:
        raise FileNotFoundError(f"No files with extensions {exts} found in {map_path}")

    poses: Dict[str, Tuple[float, float, float]] = {}
    if poses_path:
        poses = _parse_poses(Path(poses_path), map_path)

    params = BEVParams(D=D, g=g)
    model = REIN().to(device).eval()

    descriptors: List[np.ndarray] = []
    items: List[Tuple[str, Optional[Tuple[float, float, float]]]] = []

    if not quiet:
        print(
            f"Building index: files={len(files)} device={device} pca_dim={pca_dim or 'off'} D={D} g={g}",
            flush=True,
        )

    total = len(files)
    for idx, p in enumerate(files, start=1):
        # Load point cloud
        if p.suffix.lower() == ".bin":
            pts = load_bin_kitti(p)
        elif p.suffix.lower() == ".pcd":
            pts = load_pcd_open3d(p)
        else:
            continue
        if pts.size == 0:
            continue

        # To tensor -> BEV -> REIN
        pts_t = torch.from_numpy(pts).to(device)
        bev = bev_density_image_torch(pts_t, params)
        with torch.no_grad():
            _, _, desc = model(bev.unsqueeze(0))
        descriptors.append(desc.detach().cpu().numpy())

        rel = str(p.relative_to(map_path))
        pose = poses.get(rel)
        items.append((rel, pose))

        if not quiet:
            width = 28
            filled = int(width * idx / total)
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r[{bar}] {idx}/{total} {rel[:60]}", end="", flush=True)

    if not descriptors:
        raise RuntimeError("No descriptors were extracted; aborting")

    if not quiet:
        print()  # newline after progress bar

    X = np.vstack(descriptors)
    index = BEVIndex(pca_dim=pca_dim)
    if pca_dim is not None and pca_dim > 0:
        if not quiet:
            print("Fitting PCA...", flush=True)
        index.fit_pca(X)
    index.add(X, poses=None)

    # Save index
    if not quiet:
        print("Saving index...", flush=True)
    index.save(out_path)

    # Write items.jsonl
    with (out_path / "items.jsonl").open("w", encoding="utf-8") as f:
        for i, (rel, pose) in enumerate(items):
            rec = {"id": i, "file": rel, "pose": pose}
            f.write(json.dumps(rec) + "\n")

    # Write bev_params
    (out_path / "bev_params.json").write_text(
        json.dumps({"D": params.D, "g": params.g, "clamp_per_cell": params.clamp_per_cell}),
        encoding="utf-8",
    )

    # Write meta
    meta = {
        "version": "0.1.0",
        "pca_dim": pca_dim,
        "num_items": len(items),
        "descriptor_dim": int(index.dim or X.shape[1]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_ms": (time.perf_counter() - t0) * 1000.0,
    }
    (out_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if not quiet:
        dur_ms = meta["build_ms"]
        print(
            f"Done. items={meta['num_items']} dim={meta['descriptor_dim']} pca={pca_dim or 'off'} in {dur_ms:.1f} ms",
            flush=True,
        )
