# bevplace index build — Build a BEVPlace index from a directory of point clouds

This command builds a retrieval index (optionally with PCA) from a map directory containing LiDAR point clouds.
Supports PCD (Open3D) and KITTI-style BIN files. Poses are optional.

## Quick start

```bash
# Minimal (no poses) — auto GPU if available
uv run bevplace index build \
  --map-dir data/pcd \
  --out db/map \
  --ext pcd

# With poses (CSV) and quiet mode
uv run bevplace index build \
  --map-dir data/kitti/seq08 \
  --out db/seq08 \
  --ext bin \
  --poses poses.csv \
  -q
```

## Synopsis

```bash
bevplace index build \
  --map-dir <DIR> \
  --out <DB_DIR> \
  [--ext pcd|bin] \
  [--poses <POSES_FILE>] \
  [--pca-dim 0] \
  [--device cpu|cuda] \
  [--D 40.0] \
  [--g 0.4] \
  [--store-locals] \
  [--weights <CKPT>] \
  [--no-init-from-data] \
  [-q|--quiet]
```

- **--map-dir**: Directory containing point clouds.
- **--out**: Output directory to save the index.
- **--ext**: File extension type. Auto-detected if omitted (pcd and bin files discovered).
- **--poses**: Optional poses file (CSV, JSONL, or JSON). See “Pose file formats”.
- **--pca-dim**: PCA target dimension (set to 0 or omit to disable). Default: 512.
- **--pca-dim**: PCA target dimension (set to 0 or omit to disable). Default: 0 (disabled).
- **--device**: Device for feature extraction. Default: auto (cuda if available).
- **--D**, **--g**: BEV generation parameters (half-size in meters, grid size in meters).
- **--store-locals**: Also store per-item REM local feature maps (float32) to speed up pose estimation later.
- **--weights**: Load model weights from a checkpoint path (overrides default official weights URL).
- **--no-init-from-data**: If weights are unavailable, do not attempt any data-based initialization; proceed with random NetVLAD.
- **-q**, **--quiet**: Suppress progress output.

## Inputs

- PCD: loaded via Open3D; uses XYZ fields.
- BIN: KITTI format float32 Nx4; columns are (x, y, z, intensity). Intensity is ignored.
- Files are discovered recursively under --map-dir and processed in sorted order for deterministic IDs.
  - Note: For PCD support, `open3d` must be installed.

## Pose file formats (optional)

If provided, poses are stored alongside items; if omitted, poses are set to null (retrieval-only index).

- CSV (header required):
  - Columns: `file,x_m,y_m,yaw_rad` or `file,x_m,y_m,yaw_deg` (deg is auto-converted to radians)
  - Example:
    ```csv
    file,x_m,y_m,yaw_rad
    frames/000001.pcd,12.34,5.67,1.5707963
    frames/000002.pcd,13.01,5.70,1.5533430
    ```
- JSON Lines (JSONL): one object per line
  - Each line: `{ "file": "frames/000001.pcd", "pose": [x, y, yaw_rad] }`
- JSON (array in object):
  - `{ "items": [{"file":"frames/000001.bin","pose":[x,y,yaw_rad]}, ...] }`

Notes:
- Paths in pose files are matched relative to `--map-dir` (case-sensitive).
- Unmatched pose entries are warned and skipped.
- Clouds without a pose are accepted (pose will be null).

## Outputs (on-disk layout)

The index directory is self-describing and versioned:

- `<DB_DIR>/faiss.index` — FAISS Flat L2 index of descriptors (possibly PCA-projected)
- `<DB_DIR>/pca.npz` — PCA `components` and `mean` (saved only when PCA is used)
- `<DB_DIR>/items.jsonl` — One JSON per line with stable `id`, relative `file`, and optional `pose`
  - Example line: `{ "id": 0, "file": "frames/000001.pcd", "pose": [12.34, 5.67, 1.57] }`
- `<DB_DIR>/bev_params.json` — BEV parameters used for generation: `{ "D": 40.0, "g": 0.4, "clamp_per_cell": 10 }`
- `<DB_DIR>/meta.json` — Build metadata, e.g.:
  ```json
  {
    "version": "0.1.0",
    "pca_dim": 512,
    "num_items": 1234,
    "descriptor_dim": 8192,
    "created_utc": "2025-09-06T12:34:56Z"
  }
  ```
- If `--store-locals` is set:
  - `<DB_DIR>/locals/{id}.npz` — per-item REM local feature map stored as float32 array `data` with shape `[C,H,W]`.
  - This enables pose estimation at inference without access to the original map directory.

IDs are stable and correspond to the sorted order of discovered files.

## Examples

- Minimal (no poses):
  ```bash
  bevplace index build \
    --map-dir data/kitti/seq08 \
    --out db/seq08 \
    --ext bin \
    --pca-dim 512
  ```

- With PCD map and CSV poses:
  ```bash
  bevplace index build \
    --map-dir data/map_pcd \
    --out db/map \
    --ext pcd \
    --poses poses.csv \
    --D 40.0 --g 0.4
  ```

- Using JSONL poses and auto-detecting file extensions:
  ```bash
  bevplace index build \
    --map-dir data/mixed \
    --out db/mixed \
    --poses poses.jsonl
  ```

## Behavior and notes

- The command extracts BEV images, runs the REIN network to obtain global descriptors, then builds the index.
- By default, the tool attempts to load the official BEVPlace2 pretrained weights for REIN. If download/load fails, it falls back to random init.
- If PCA is enabled, PCA is fit on all descriptors (or a future `--pca-sample N` when available), and descriptors are projected before FAISS indexing.
- Unreadable/empty files are skipped with a warning; the build continues.
- Deterministic execution: discovery order is sorted; seeds fixed for any sampling.
- With `--store-locals`, REM local features are written during the main processing pass; no extra pass is needed.
- Progress: a compact progress bar and minimal status messages are printed. Use `--quiet` to disable all output.
- Performance tips:
  - Set `--device cuda` to utilize GPU for feature extraction.
  - Adjust `--pca-dim` to balance recall and search speed (512 is a good default).
  - `--D` and `--g` affect BEV size (and throughput). With upstream-compatible indexing: D=40.0, g=0.4 → 201×201.

## Downstream usage

- The resulting index works with `BEVLocalizer`:
  - Use `items.jsonl` to provide a reference provider (id → file path) for pose estimation.
  - Use stored poses where available for downstream global pose composition.

## Troubleshooting

- “No files found”: check `--map-dir` and `--ext`; ensure `.pcd` or `.bin` files exist.
- “Pose entries not matched”: ensure `file` paths in the pose file are relative to `--map-dir` and correctly cased.
- “CUDA OOM”: try `--device cpu` or reduce `--pca-dim` or BEV size (`--D`, `--g`).
