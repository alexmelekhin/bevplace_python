from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import torch

from bevplace import REIN
from bevplace.core.types import BEVParams
from bevplace.preprocess.bev import bev_density_image_torch
from bevplace.retrieval import BEVIndex
from bevplace.cli.common import discover_files, load_bin_kitti, load_pcd_open3d
from bevplace.models.weights import ensure_default_weights


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Build BEVPlace index from a directory of point clouds")
    parser.add_argument("--map-dir", required=True, help="Directory with .pcd/.bin files")
    parser.add_argument("--out", required=True, help="Output directory for index")
    parser.add_argument("--ext", choices=["pcd", "bin"], default=None, help="Restrict to a single extension")
    parser.add_argument(
        "--poses",
        default=None,
        help="Optional poses file (csv/json/jsonl/tum). For TUM, poses are auto-synced by nearest timestamp.",
    )
    parser.add_argument("--pca-dim", type=int, default=0, help="PCA target dimension; 0 disables PCA")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--D", type=float, default=40.0, help="BEV half-size in meters")
    parser.add_argument("--g", type=float, default=0.4, help="BEV grid size in meters per pixel")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--store-locals", action="store_true", help="Store REM local features in index dir")
    parser.add_argument("--weights", default=None, help="Path to checkpoint to load (overrides default)")
    parser.add_argument(
        "--no-init-from-data", action="store_true", help="Do not init NetVLAD from data if weights missing"
    )

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
        store_locals=args.store_locals,
        weights_path=args.weights,
        no_init_from_data=args.no_init_from_data,
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


def _parse_tum_poses(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a TUM poses file.

    Supports two common formats per line (whitespace-separated):
    - t x y z qx qy qz qw (8 columns, quaternion)
    - t x y z roll pitch yaw (7 columns, radians)

    Returns:
        A tuple (times, xy_yaw) where times is a float numpy array [N], and
        xy_yaw is a float numpy array [N,3] with columns [x, y, yaw].
    """
    times: List[float] = []
    xy_yaw_list: List[Tuple[float, float, float]] = []

    if not path.exists():
        return np.asarray(times, dtype=float), np.asarray(xy_yaw_list, dtype=float)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                if len(parts) >= 8:
                    t, x, y, _z, qx, qy, qz, qw = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                                    float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                    # Yaw from quaternion (Z rotation)
                    # Reference: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                    num = 2.0 * (qw * qz + qx * qy)
                    den = 1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw = float(np.arctan2(num, den))
                elif len(parts) >= 7:
                    t, x, y, _z, _roll, _pitch, yaw = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                                       float(parts[4]), float(parts[5]), float(parts[6]))
                else:
                    continue
            except ValueError:
                continue

            times.append(t)
            xy_yaw_list.append((x, y, yaw))

    if not times:
        return np.asarray([], dtype=float), np.asarray([], dtype=float).reshape(0, 3)

    times_arr = np.asarray(times, dtype=float)
    xy_yaw_arr = np.asarray(xy_yaw_list, dtype=float)

    # Ensure sorted by time
    order = np.argsort(times_arr)
    return times_arr[order], xy_yaw_arr[order]


def _extract_timestamp_from_filename(p: Path) -> Optional[float]:
    """Extract a floating-point timestamp from a file name.

    Tries to parse the stem directly; if that fails, searches for the first
    numeric substring that looks like a float or integer.

    Args:
        p: Path to the file.

    Returns:
        Parsed timestamp as float if found, otherwise None.
    """
    stem = p.stem
    try:
        return float(stem)
    except ValueError:
        pass

    # Find first numeric token (float or int)
    match = re.search(r"[-+]?\d*\.\d+|\d+", stem)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _nearest_index(sorted_times: np.ndarray, t: float) -> int:
    """Find index of nearest time in a sorted 1D array.

    Args:
        sorted_times: 1D numpy array of monotonically increasing times.
        t: Query timestamp.

    Returns:
        Index of the nearest time value.
    """
    if sorted_times.size == 0:
        return 0
    idx = int(np.searchsorted(sorted_times, t))
    if idx == 0:
        return 0
    if idx >= sorted_times.size:
        return int(sorted_times.size - 1)
    before = idx - 1
    after = idx
    if abs(sorted_times[after] - t) < abs(sorted_times[before] - t):
        return after
    return before


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
    store_locals: bool = False,
    weights_path: Optional[str] = None,
    no_init_from_data: bool = False,
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
    tum_times: Optional[np.ndarray] = None
    tum_xyyaw: Optional[np.ndarray] = None
    if poses_path:
        _pp = Path(poses_path)
        if _pp.suffix.lower() in {".tum"}:
            tum_times, tum_xyyaw = _parse_tum_poses(_pp)
            if not quiet and (tum_times is None or tum_times.size == 0):
                print("Warning: TUM poses file parsed but contains no entries", flush=True)
        else:
            poses = _parse_poses(_pp, map_path)

    params = BEVParams(D=D, g=g)
    model = REIN().to(device).eval()
    # Attempt to load weights (explicit path or default); fallback to data-init if allowed
    loaded = False
    if weights_path:
        try:
            from bevplace.models.weights import load_state_dict_into_rein
            from pathlib import Path as _P

            load_state_dict_into_rein(model, _P(weights_path))
            loaded = True
            if not quiet:
                print("Loaded weights from --weights", flush=True)
        except Exception as _:
            if not quiet:
                print("Warning: failed to load --weights; will try default or data-init", flush=True)
    if not loaded:
        loaded = ensure_default_weights(model, quiet=quiet)
    if not loaded and no_init_from_data and not quiet:
        print("Note: --no-init-from-data set; using random NetVLAD (results may be poor)", flush=True)

    descriptors: List[np.ndarray] = []
    items: List[Tuple[str, Optional[Tuple[float, float, float]]]] = []

    if not quiet:
        print(
            f"Building index: files={len(files)} device={device} pca_dim={pca_dim or 'off'} D={D} g={g}",
            flush=True,
        )

    total = len(files)
    locals_dir = out_path / "locals"
    if store_locals:
        locals_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
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
            _, rem_map, desc = model(bev.unsqueeze(0))
            descriptors.append(desc.detach().cpu().numpy())

            # Optionally persist locals now to avoid second pass
            if store_locals:
                arr = rem_map.detach().cpu().squeeze(0).numpy()  # [C,H,W]
                np.savez(locals_dir / f"{idx - 1:06d}.npz", data=arr)

            rel = str(p.relative_to(map_path))
            pose = poses.get(rel)
            if pose is None and tum_times is not None and tum_xyyaw is not None and tum_times.size > 0:
                ts = _extract_timestamp_from_filename(p)
                if ts is not None:
                    j = _nearest_index(tum_times, float(ts))
                    x, y, yaw = tum_xyyaw[j]
                    pose = (float(x), float(y), float(yaw))
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
    if (pca_dim is not None and pca_dim > 0) and not quiet:
        print("Fitting PCA...", flush=True)
    if pca_dim is not None and pca_dim > 0:
        index.fit_pca(X)
    index.add(X, poses=None)

    # Locals are already stored during the main pass when requested

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
