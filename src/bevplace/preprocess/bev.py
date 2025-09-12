from __future__ import annotations

from typing import Tuple

import torch

from bevplace.core.types import BEVParams


def _compute_grid_indices(
    points_xy: torch.Tensor, params: BEVParams
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute row/col indices and grid size for BEV image (upstream-compatible).

    Mapping (matching BEVPlace2 generator):
      - Bounds: x,y in [-D, D]
      - Indexing:
          x_max_ind = floor(D/g); y_max_ind = floor(D/g)
          col = x_max_ind - floor(y/g)
          row = y_max_ind - floor(x/g)
      - Shape:
          H = floor(D/g) - floor(-D/g) + 1 (same for W)
    """
    D = params.D
    g = params.g

    # Crop to square window and keep inside bounds (|x|<=D, |y|<=D)
    mask = (points_xy[:, 0].abs() <= D) & (points_xy[:, 1].abs() <= D)
    pts = points_xy[mask]

    # Compute grid size using floor-based indexing
    import math

    x_min_ind = math.floor(-D / g)
    x_max_ind = math.floor(D / g)
    y_min_ind = math.floor(-D / g)
    y_max_ind = math.floor(D / g)
    H = (y_max_ind - y_min_ind) + 1
    W = (x_max_ind - x_min_ind) + 1
    if H <= 0 or W <= 0:
        return pts.new_empty((0,), dtype=torch.long), pts.new_empty((0,), dtype=torch.long), 0, 0

    # Upstream axis convention
    cols_raw = torch.floor(pts[:, 1] / g)
    rows_raw = torch.floor(pts[:, 0] / g)
    cols = (x_max_ind - cols_raw).to(dtype=torch.long)
    rows = (y_max_ind - rows_raw).to(dtype=torch.long)

    rows = torch.clamp(rows, 0, H - 1)
    cols = torch.clamp(cols, 0, W - 1)
    return rows, cols, H, W


def bev_density_image_torch(points: torch.Tensor, params: BEVParams) -> torch.Tensor:
    """Compute BEV density image (upstream-compatible) on the same device as input points.

    Steps:
      1) Crop |x|,|y|,|z| <= D
      2) Voxel downsample with voxel_size (default g) in 3D
      3) Map to pixel grid with upstream indexing
      4) Count per-cell, cap increments at ``increment_cap``
      5) Intensity shaping: multiply by 10, clamp to 255, per-image max normalize to 255
      6) Return float32 image scaled by 1/256 (to match upstream reader)
    """
    if points.ndim != 2 or points.shape[-1] < 2:
        raise ValueError("points must be [N,3] or [N,>=2]")

    device = points.device
    D = float(params.D)
    g = float(params.g)
    vs = float(params.voxel_size) if params.voxel_size is not None else g

    # Determine output grid size using same rule as upstream
    import math

    x_min_ind = math.floor(-D / g)
    x_max_ind = math.floor(D / g)
    y_min_ind = math.floor(-D / g)
    y_max_ind = math.floor(D / g)
    H = (y_max_ind - y_min_ind) + 1
    W = (x_max_ind - x_min_ind) + 1
    if H <= 0 or W <= 0:
        return torch.zeros((1, 0, 0), device=device, dtype=torch.float32)

    # Filter to bounds (|x|,|y|,|z| <= D)
    if points.shape[1] >= 3:
        mask = (points[:, 0].abs() <= D) & (points[:, 1].abs() <= D) & (points[:, 2].abs() <= D)
        pts = points[mask]
    else:
        mask = (points[:, 0].abs() <= D) & (points[:, 1].abs() <= D)
        # Synthesize z=0 for voxelization
        z = torch.zeros((points.shape[0], 1), device=device, dtype=points.dtype)
        pts = torch.cat([points, z], dim=1)[mask]

    if pts.numel() == 0:
        return torch.zeros((1, H, W), device=device, dtype=torch.float32)

    # 3D voxel downsample using integer grid and first-occurrence selection
    vx = torch.floor(pts[:, 0] / vs).to(torch.int64)
    vy = torch.floor(pts[:, 1] / vs).to(torch.int64)
    vz = torch.floor(pts[:, 2] / vs).to(torch.int64)
    V = torch.stack([vx, vy, vz], dim=1)
    # Unique voxel keys with stable first index per voxel
    uniq, inv = torch.unique(V, dim=0, return_inverse=True)
    order = torch.argsort(inv)
    inv_sorted = inv[order]
    is_first = torch.ones_like(inv_sorted, dtype=torch.bool)
    is_first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first_indices = order[is_first]

    pts_xy = pts[first_indices, :2]

    # Map to grid indices
    rows, cols, _, _ = _compute_grid_indices(pts_xy, params)
    if rows.numel() == 0:
        return torch.zeros((1, H, W), device=device, dtype=torch.float32)

    # Count hits per cell
    flat_idx = rows * W + cols
    counts = torch.bincount(flat_idx, minlength=H * W)
    # Cap increments per cell to increment_cap (default 10)
    cap = int(params.increment_cap)
    counts = torch.clamp(counts, max=cap)

    # Intensity shaping to mimic upstream PNG pipeline
    img = counts.to(torch.float32) * 10.0
    img = torch.clamp(img, max=255.0)

    if params.per_image_max_normalize:
        max_val = torch.max(img)
        if max_val > 0:
            img = img * (255.0 / max_val)
        else:
            img = img * 0.0

    # Final scaling to match cv2.imread(...)/256 in upstream
    img = img.view(H, W) / 256.0
    return img.unsqueeze(0).to(device=device, dtype=torch.float32)
