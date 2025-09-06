from __future__ import annotations

import argparse

from bevplace.cli.index_build import build_index
from bevplace.cli.localize import run_localize


def _add_index_subparser(subparsers: argparse._SubParsersAction) -> None:
    p_index = subparsers.add_parser("index", help="Index-related commands")
    sp = p_index.add_subparsers(dest="index_cmd", required=True)

    p_build = sp.add_parser("build", help="Build a retrieval index from a directory of point clouds")
    p_build.add_argument("--map-dir", required=True, help="Directory with .pcd/.bin files")
    p_build.add_argument("--out", required=True, help="Output directory for index")
    p_build.add_argument("--ext", choices=["pcd", "bin"], default=None, help="Restrict to a single extension")
    p_build.add_argument("--poses", default=None, help="Optional poses file (csv/json/jsonl)")
    p_build.add_argument("--pca-dim", type=int, default=512, help="PCA target dimension; 0 disables PCA")
    p_build.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p_build.add_argument("--D", type=float, default=40.0, help="BEV half-size in meters")
    p_build.add_argument("--g", type=float, default=0.4, help="BEV grid size in meters per pixel")
    p_build.add_argument("--store-locals", action="store_true", help="Store REM local features in index dir")

    def run_build(args: argparse.Namespace) -> None:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            store_locals=args.store_locals,
        )

    p_build.set_defaults(func=run_build)

    # localize subcommand under root CLI (sibling to index)
    p_local = subparsers.add_parser("localize", help="Localize a single point cloud against an index")
    g_in = p_local.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--pcd", default=None, help="Query PCD file")
    g_in.add_argument("--bin", default=None, help="Query KITTI BIN file")
    p_local.add_argument("--db", required=True, help="Path to index directory")
    p_local.add_argument("--map-dir", default=None, help="Path to original map directory (for pose)")
    p_local.add_argument("--estimate-pose", action="store_true", help="Estimate relative pose on BEV")
    p_local.add_argument("--topk", type=int, default=5, help="Retrieve top-K candidates (print all)")
    p_local.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p_local.add_argument("--D", type=float, default=40.0, help="BEV half-size in meters")
    p_local.add_argument("--g", type=float, default=0.4, help="BEV grid size in meters per pixel")
    p_local.add_argument("--out-json", default=None, help="Write result JSON to this path")
    p_local.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    p_local.set_defaults(func=run_localize)


def main() -> None:  # pragma: no cover - simple CLI wiring
    parser = argparse.ArgumentParser(prog="bevplace", description="BEVPlace command line")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _add_index_subparser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
