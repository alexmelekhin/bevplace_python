from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bevplace import REIN
from bevplace.core.types import BEVParams
from bevplace.pipeline.localizer import BEVLocalizer
from bevplace.preprocess.bev import bev_density_image_torch
from bevplace.retrieval import BEVIndex
from bevplace.cli.common import (
    load_bin_kitti,
    load_pcd_open3d,
    read_items,
    read_bev_params,
    read_locals,
)


def run_localize(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    db_dir = Path(args.db)
    index = BEVIndex.load(db_dir)

    # Resolve BEV params (prefer DB defaults if available)
    params = read_bev_params(db_dir, BEVParams(D=args.D, g=args.g))

    # Load query point cloud
    if args.pcd is None and args.bin is None:
        raise ValueError("Provide exactly one of --pcd or --bin")
    if args.pcd is not None and args.bin is not None:
        raise ValueError("Provide exactly one of --pcd or --bin, not both")
    if args.pcd:
        pts_np = load_pcd_open3d(Path(args.pcd))
    else:
        pts_np = load_bin_kitti(Path(args.bin))

    # Build model
    model = REIN().to(device).eval()

    # Optional reference provider
    ref_provider = None
    if args.estimate_pose:
        # Prefer stored locals (no need for map-dir)
        locals_dir = db_dir / "locals"
        if locals_dir.exists():
            def provider_locals(match_id: int):
                rem = read_locals(db_dir, match_id)  # [C,H,W] float32
                return torch.from_numpy(rem).unsqueeze(0).to(device)

            ref_provider = provider_locals
        else:
            # Fallback: require map-dir to compute rem_map on the fly
            if not args.map_dir:
                raise ValueError("--map-dir is required for --estimate-pose")
            map_dir = Path(args.map_dir)
            items = read_items(db_dir)

            def provider_fn(match_id: int):
                rel = items[match_id]["file"]
                path = map_dir / rel
                if path.suffix.lower() == ".pcd":
                    pts_m = load_pcd_open3d(path)
                else:
                    pts_m = load_bin_kitti(path)
                pts_m_t = torch.from_numpy(pts_m).to(device)
                bev_m = bev_density_image_torch(pts_m_t, params)
                with torch.no_grad():
                    _, rem_map_m, _ = model(bev_m.unsqueeze(0))
                return bev_m, rem_map_m

            ref_provider = provider_fn

    # Localize via orchestrator
    localizer = BEVLocalizer(
        model=model, index=index, bev_params=params, device=device, reference_provider=ref_provider
    )
    pts_t = torch.from_numpy(pts_np).to(device)
    if not args.quiet:
        print("Running localization...", flush=True)
    result = localizer.localize(pts_t, k=args.topk)

    # Print summary
    if not args.quiet:
        print(f"matched_id: {result.matched_id}")
        print(f"topk: {list(result.topk)}")
        print(f"distances: {[round(d, 3) for d in result.distances]}")
        if args.estimate_pose:
            tx, ty = float(result.pose_relative[0, 2]), float(result.pose_relative[1, 2])
            print(
                f"pose: tx={tx:.2f}px ty={ty:.2f}px inliers={result.inliers_ratio:.2f} matches={result.num_matches}",
                flush=True,
            )
        print(
            f"timings(ms): bev={result.timings.get('bev_ms', 0):.1f} rein={result.timings.get('rein_ms', 0):.1f} retrieval={result.timings.get('retrieval_ms', 0):.1f} pose={result.timings.get('pose_ms', 0):.1f}",
            flush=True,
        )

    # Save JSON if requested
    if args.out_json:
        out = {
            "matched_id": int(result.matched_id),
            "topk": list(result.topk),
            "distances": list(result.distances),
            "pose_relative": result.pose_relative.tolist(),
            "pose_global": (
                result.pose_global.tolist() if isinstance(result.pose_global, np.ndarray) else None
            ),
            "inliers_ratio": float(result.inliers_ratio),
            "num_matches": int(result.num_matches),
            "timings_ms": {
                "bev": float(result.timings.get("bev_ms", 0.0)),
                "rein": float(result.timings.get("rein_ms", 0.0)),
                "retrieval": float(result.timings.get("retrieval_ms", 0.0)),
                "pose": float(result.timings.get("pose_ms", 0.0)),
            },
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
