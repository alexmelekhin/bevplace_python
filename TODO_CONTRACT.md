## BEVPlace++ Inference Package — TODO Contract

### Scope
- **Goal**: Production-ready Python package that takes LiDAR point clouds and outputs a global 3-DoF pose with diagnostics, implementing BEVPlace++ pipeline (BEV density image → REIN → retrieval → 2D pose via RANSAC).
- **Constraints**: Modern Python (>=3.10), `uv` for packaging, `ruff` for lint/format, device-agnostic PyTorch (CPU/GPU), clean public APIs, minimal external assumptions.

### High-level Architecture (agreed)
- `bevplace/core`: types, geometry(SE(2)), config (Pydantic)
- `bevplace/io`: lidar adapters (numpy/Open3D/ROS2), storage
- `bevplace/preprocess`: BEV density image generation
- `bevplace/models`: REIN (REM + NetVLAD), weights management
- `bevplace/retrieval`: FAISS + PCA-512 index (add/search/save/load)
- `bevplace/pose`: keypoints, sampling, matching, 2D rigid RANSAC
- `bevplace/pipeline`: `BEVLocalizer` orchestrator
- `bevplace/utils`: logging, timing, visualization
- `bevplace/cli`: `index`, `localize`, optional `serve`

### Development Process Rules (agreed)
- **TDD**: follow pragmatic TDD — write a minimal set of simple unit tests first, then implement code; expand tests alongside features and fixes.
- **Test all code**: add unit tests covering public APIs and critical edge cases.
- **Formatting and linting**: always run `uv run ruff format` and then `uv run ruff check` before committing/marking tasks complete.

### Public APIs (contract)
- BEV generation:
  - `bevplace.preprocess.bev.bev_density_image_torch(points: torch.Tensor, params: BEVParams) -> torch.Tensor`
  - Returns: `torch.Tensor[1,H,W]` (float32 in [0,1]) on same device as `points`.
- REIN model:
  - `bevplace.models.rein.REIN(weights: str|Path|None, device: torch.device|str|None) -> nn.Module`
  - Forward: `(rem_for_vlad, rem_for_kp, global_desc)`.
- Retrieval:
  - `BEVIndex.add(descs: np.ndarray, poses: list[Pose2D], meta: dict|None)`
  - `BEVIndex.search(query_desc: np.ndarray, k:int=1) -> list[RetrievalHit]`
  - `save(dir) / load(dir)` (versioned artifacts: FAISS index, PCA, poses/meta, bev_params).
- Pose estimation:
  - `estimate_relative_pose(bev_q, bev_m, rem_q, rem_m, params) -> (Transform2D, RansacReport)`
- End-to-end:
  - `BEVLocalizer.localize(point_cloud: np.ndarray|torch.Tensor, index: BEVIndex) -> LocalizationResult`

### Torch+CUDA BEV Density Image — Canonical Specification (agreed)
- Function: `bev_density_image_torch(points, params)` where:
  - `points`: `torch.Tensor[N,3]` (x,y,z) on CPU or CUDA; dtype float32/float64 accepted; processed on the same device.
  - `params`: `BEVParams(D: float = 40.0, g: float = 0.4, clamp_per_cell: int = 10)`.
- Grid:
  - `H = W = int(round(2*D/g))` (paper default → 200x200 when D=40, g=0.4).
  - Axis convention (fixed): x→rows (downwards), y→cols (rightwards).
    - `row = floor(( D - x) / g)` maps x=+D to row 0 (top), x=−D to row H−1 (bottom).
    - `col = floor(( y + D) / g)` maps y=−D to col 0 (left), y=+D to col W−1 (right).
- Filtering:
  - Keep points with `|x|<=D`, `|y|<=D`, `|z|<=D` (symmetric crop for robustness; z used only to remove outliers).
- Binning:
  - Compute per-cell counts using vectorized indexing; counts clipped at `clamp_per_cell`.
- Normalization:
  - Density image `I(u,v) = min(N_g, clamp_per_cell) / clamp_per_cell` → float32 in `[0,1]`.
  - Edge case: if all counts are zero, return zeros.
- Output:
  - Tensor shape: `[1, H, W]` (channel-first) on same device as input.
  - Deterministic given the same inputs and params.
- Debug helper (optional): `bev_to_uint8(I: torch.Tensor) -> torch.Tensor[H,W]` scaling to `[0,255]` for visualization.
- Performance notes:
  - No host↔device copies; works entirely on device of `points`.
  - Large-N robustness via `torch.bincount` on flat indices (as in `mmpr_bevplace`), without Python loops.

### Retrieval Index (contract)
- PCA to 512-D with saved components/mean; FAISS index type configurable (Flat/HNSW/IVF). Versioned artifacts and reproducible save/load.
- Incremental `add()` and deferred `train()` (when index type requires training).

### Pose Estimation (contract)
- FAST keypoints (OpenCV) with configurable threshold; mutual check + ratio test; max matches cap.
- RANSAC (SE(2)): configurable inlier threshold (pixels or meters via `g`), max iters, confidence; return best transform + report (inliers, residual stats).

### Diagnostics and Outputs
- `LocalizationResult`: `pose_global`, `pose_relative`, `matched_id`, `topk`, `distances`, `inliers_ratio`, `num_matches`, `timings`, `descriptor_q`, optional debug images.

### CLI (contract)
- `bevplace index build --map <dir|bag|pcd> --out <db_dir>`
- `bevplace localize --pcd <path> --db <db_dir>`
- `bevplace serve --db <db_dir>` (optional; HTTP/gRPC/ROS2 adapters later)

### Deliverables — TODOs
- [x] Implement torch-based BEV density image with CUDA support and fixed D,g.
- [x] Port REM/NetVLAD/REIN from `mmpr_bevplace`; device-agnostic, typed.
- [ ] Build retrieval index (PCA-512 + FAISS); save/load; add/search APIs.
- [ ] Implement keypoints, descriptor sampling, matching with thresholds.
- [ ] Implement 2D rigid RANSAC with reports and parameters.
- [ ] Create `BEVLocalizer` orchestrator returning `LocalizationResult` with diagnostics.
- [ ] Add CLI commands: `index` build/localize; optional `serve`.
- [ ] Introduce Pydantic-based configs and seed control.
- [ ] Add logging/profiling/visualization utilities.
- [ ] Weights management: load/convert pre-trained REIN; document format.
- [ ] QA: invariance tests, pose composition checks, performance smoke tests.
- [ ] Packaging with `uv` and `ruff`; CI quality gate.
- [ ] Optional adapters: ROS2/gRPC for online serving.

### Acceptance criteria
- End-to-end inference on sample LiDAR returns valid `LocalizationResult` with stable timings (>20 Hz desktop or close) and non-empty diagnostics.
- Descriptor invariance sanity check across rotations passes within tolerance.
- Pose composition on synthetic transforms is correct (within thresholds set in config).
- Index save/load round-trips with identical search results (within float tolerance).
