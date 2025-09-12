# BEVPlace++ method packaged

## Installation

Requires Python 3.10+.

Install from GitHub (core):

```bash
pip install "bevplace @ git+https://github.com/alexmelekhin/bevplace_python.git"
```

Install with extras (recommended for runtime):

```bash
pip install "bevplace[torch,faiss,open3d] @ git+https://github.com/alexmelekhin/bevplace_python.git"
```

Notes:
- These extras correspond to optional dependencies defined in `pyproject.toml` (`[project.optional-dependencies]`).
- Depending on your platform/GPU, you may want a specific PyTorch build (CPU or CUDA). Example (CUDA 12.1):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- If you prefer manual installs, you can install `torch`, `faiss-cpu` (or `faiss-gpu`), and `open3d` yourself before installing `bevplace`.

## Usage

### CLI

Two entrypoints:

- Build index (PCA off by default):
```bash
python -m bevplace.cli index build --map-dir /path/to/pcds_or_bins --out /path/to/index \
  --D 40.0 --g 0.4 --store-locals
```

- Localize a single query:
```bash
python -m bevplace.cli localize --db /path/to/index --bin /path/to/query.bin \
  --estimate-pose --pose-thresh-m 0.5 --pose-kps fast
```

Notes:
- BEV images are computed on-the-fly with upstream-compatible settings (voxel ds=g, per-cell cap=10, per-image max normalization).
- Weights are auto-downloaded from the official repo; if not available, you can pass `--weights PATH` during index build to use a local checkpoint.
- If you did not build with `--store-locals`, provide `--map-dir` to `localize` so pose can be computed on-the-fly from the original map files.

### API

Simple programmatic API for integration (e.g., a ROS node):

```python
from pathlib import Path
import numpy as np
import torch

from bevplace import REIN
from bevplace.core.types import BEVParams
from bevplace.models.weights import ensure_default_weights
from bevplace.retrieval import BEVIndex
from bevplace.pipeline.localizer import BEVLocalizer
from bevplace.cli.common import read_bev_params, read_locals

# 1) Load pre-computed index and BEV params
db_dir = Path("/path/to/index")
index = BEVIndex.load(db_dir)
params = read_bev_params(db_dir, BEVParams(D=40.0, g=0.4))

# 2) Build model and load official weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = REIN().to(device).eval()
ensure_default_weights(model, quiet=True)

# 3) Optional: reference provider for pose (expects locals stored during indexing)
def provider(match_id: int):
    rem = read_locals(db_dir, match_id)  # [C,H,W] float32
    return torch.from_numpy(rem).unsqueeze(0).to(device)

localizer = BEVLocalizer(
    model=model,
    index=index,
    bev_params=params,
    device=device,
    reference_provider=provider,  # omit or set None to skip pose estimation
)

# 4) In your ROS callback (Nx3 float32 point cloud):
def handle_point_cloud(points_xyz: np.ndarray):
    result = localizer.localize(points_xyz, k=1)
    # result.pose_relative: 3x3 SE(2) (pixel units); convert to meters via g if needed
    # result.matched_id: retrieved map id
    return result
```

Notes:
- For best pose results, build the index with `--store-locals` so the provider can read precomputed REM features.
- BEV generation is done on-the-fly inside `BEVLocalizer` and matches the upstream implementation (D=40, g=0.4 by default).

#### On-the-fly pose provider (no stored locals)

If you don't store locals in the index, you can compute them when needed by supplying a provider that loads the matched map cloud and runs the model once:

```python
from pathlib import Path
import torch
from bevplace.cli.common import read_items, load_bin_kitti, load_pcd_open3d
from bevplace.preprocess.bev import bev_density_image_torch

items = read_items(db_dir)  # list of {"id": int, "file": str, ...}

def provider(match_id: int):
    rel = items[match_id]["file"]
    path = Path(map_dir) / rel  # map_dir is the folder used for indexing
    if path.suffix.lower() == ".pcd":
        pts = load_pcd_open3d(path)
    else:
        pts = load_bin_kitti(path)
    bev_m = bev_density_image_torch(torch.from_numpy(pts).to(device), params)
    with torch.no_grad():
        _, rem_map_m, _ = model(bev_m.unsqueeze(0))
    return rem_map_m  # or (bev_m, rem_map_m)
```

## Acknowledgments

This package draws on ideas and assets from the original BEVPlace/BEVPlace++ project by Luo et al.

- Original repository: [BEVPlace2 (official)](https://github.com/zjuluolun/BEVPlace2)

Please cite the original papers if you use this work in academic contexts:

```
@ARTICLE{luo2024bevplaceplus,
      journal={IEEE Transactions on Robotics (T-RO)}, 
      title={BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles}, 
      author={Lun Luo and Si-Yuan Cao and Xiaorui Li and Jintao Xu and Rui Ai and Zhu Yu and Xieyuanli Chen},
      volume={41},
      number={},
      pages={4479-4498},
      year={2025},
}
```

```
@INPROCEEDINGS{luo2023bevplace,
      author={Luo, Lun and Zheng, Shuhang and Li, Yixuan and Fan, Yongzhi and Yu, Beinan and Cao, Si-Yuan and Li, Junwei and Shen, Hui-Liang},
      booktitle={2023 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
      title={BEVPlace: Learning LiDAR-based Place Recognition using Bird’s Eye View Images}, 
      year={2023},
      pages={8666-8675},
      doi={10.1109/ICCV51070.2023.00799}
}
```
