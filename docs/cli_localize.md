# bevplace localize — Localize a single LiDAR frame against a BEVPlace index

This command runs the end-to-end BEVPlace++ inference for a single point cloud:
- LiDAR → BEV density image
- REIN descriptor extraction and retrieval against an existing index
- (Optional) relative pose estimation on BEV using REM-feature matching + RANSAC

## Quick start

```bash
# Retrieval only
uv run bevplace localize \
  --pcd data/frame_000123.pcd \
  --db db/map

# Retrieval + pose estimation (requires the original map directory)
uv run bevplace localize \
  --bin data_kitti/000123.bin \
  --db db/seq08 \
  --map-dir data_kitti \
  --estimate-pose
```

## Synopsis

```bash
bevplace localize \
  (--pcd <FILE.pcd> | --bin <FILE.bin>) \
  --db <DB_DIR> \
  [--map-dir <DIR>] \
  [--estimate-pose] \
  [--topk 5] \
  [--device cpu|cuda] \
  [--D 40.0] \
  [--g 0.4] \
  [--out-json result.json] \
  [-q|--quiet]
```

- Exactly one of `--pcd` or `--bin` must be provided.
- `--db` points to an index directory created by `bevplace index build`.
- `--map-dir` is required only when `--estimate-pose` is used so we can load the matched reference cloud identified in `items.jsonl`.
  - If the index was built with `--store-locals`, `--map-dir` is not required: the tool will load stored REM locals from the index directory.

## Options

- **--pcd FILE**: Input PCD point cloud (Open3D-readable).
- **--bin FILE**: Input KITTI .bin point cloud (float32 Nx4 or Nx3).
- **--db DIR**: Path to the index directory produced by `bevplace index build`.
- **--map-dir DIR**: Root directory containing the map point clouds referenced in the index `items.jsonl` (used for pose estimation).
- **--estimate-pose**: If set, runs BEV keypoint matching + RANSAC to estimate a 2D rigid transform in BEV pixels, returned as SE(2). Requires `--map-dir`.
- **--topk K**: Retrieve top-K candidates (default: 5). The CLI prints all K; pose is computed from top-1.
- **--device**: Device for inference (default: auto).
- **--D**, **--g**: BEV generation parameters. If the DB contains `bev_params.json`, those values are used; otherwise these flags are used. They should match the parameters used for indexing for best results.
- **--out-json FILE**: Save the localization result as JSON.
- **-q**, **--quiet**: Suppress progress output.

## Inputs

- PCD: loaded via Open3D (requires `open3d`).
- BIN: KITTI format float32 Nx4 (XYZI) or Nx3 (XYZ); intensity is ignored.

## Outputs

The CLI prints a compact human-readable summary and (optionally) writes a JSON with the following fields:

```json
{
  "matched_id": 123,
  "topk": [123, 122, 87, 41, 3],
  "distances": [0.00, 0.31, 0.47, 0.53, 0.61],
  "pose_relative": [[1,0,tx],[0,1,ty],[0,0,1]],
  "pose_global": null,
  "inliers_ratio": 0.93,
  "num_matches": 542,
  "timings_ms": {"bev": 4.5, "rein": 15.2, "retrieval": 0.5, "pose": 9.8}
}
```

Notes:
- `pose_relative` is an SE(2) transform estimated on BEV; to convert pixel translations to meters multiply by grid size `g`.
- `pose_global` stays `null` unless you post-process with the stored map pose for the matched item (compose externally if needed).

## Behavior and notes

- If `--estimate-pose` is set, the tool reads the matched item’s relative path from `items.jsonl` in the DB and loads that cloud from `--map-dir` to compute REM features for matching.
  - If the index contains stored locals (`db/locals/{id}.npz`), they are used directly and the map directory is not needed.
- If `--estimate-pose` is not set, the command performs retrieval only.
- `--D` and `--g` should match the parameters used to build the index to avoid descriptor distribution shift.
- Progress: a small progress message is printed; use `--quiet` to suppress.
- The tool attempts to load official BEVPlace2 weights by default for consistency with the published model; if unavailable, it proceeds with random init (results may vary).

## Examples

- Retrieval only on CPU:
```bash
uv run bevplace localize --pcd sample.pcd --db db/map --device cpu
```

- Retrieve+pose with top-10 candidates (pose uses top-1):
```bash
uv run bevplace localize --bin 000314.bin --db db/seq08 --map-dir data_kitti --estimate-pose --topk 10
```

## Troubleshooting

- “DB missing items.jsonl”: ensure you built the index via `bevplace index build` and didn’t remove files.
- “Cannot load reference for pose estimation”: check `--map-dir` points to the exact directory containing the files referenced in `items.jsonl`.
- “CUDA OOM”: try `--device cpu` or reduce `--D` and/or increase `--g`.
