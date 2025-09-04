from __future__ import annotations

from typing import Tuple

import torch

from bevplace.core.types import BEVParams


def _compute_grid_indices(
    points_xy: torch.Tensor, params: BEVParams
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute row/col indices and grid size for BEV density image.

    Args:
        points_xy: Tensor [N, 2] of (x,y) on any device, float.
        params: BEV parameters.
    Returns:
        rows: Tensor [M] of row indices (valid subset).
        cols: Tensor [M] of col indices (valid subset).
        H: image height (== W).
        W: image width (== H).
    """
    D = params.D
    g = params.g

    # Crop to square window and keep inside bounds (|x|<=D, |y|<=D)
    mask = (points_xy[:, 0].abs() <= D) & (points_xy[:, 1].abs() <= D)
    pts = points_xy[mask]

    H = W = int(round(2 * D / g))
    if H <= 0:
        return pts.new_empty((0,), dtype=torch.long), pts.new_empty((0,), dtype=torch.long), 0, 0

    # Axis convention:
    # row = floor(( D - x) / g), col = floor(( y + D) / g)
    # Include boundary points by clamping to [0, H-1] and [0, W-1].
    rows_raw = torch.floor((D - pts[:, 0]) / g).to(dtype=torch.long)
    cols_raw = torch.floor((pts[:, 1] + D) / g).to(dtype=torch.long)

    rows = torch.clamp(rows_raw, 0, H - 1)
    cols = torch.clamp(cols_raw, 0, W - 1)
    return rows, cols, H, W


def bev_density_image_torch(points: torch.Tensor, params: BEVParams) -> torch.Tensor:
    """Compute BEV density image on the same device as input points.

    Args:
        points: Tensor [N,3] (x,y,z) on CPU or CUDA.
        params: BEVParams with D, g, clamp_per_cell.
    Returns:
        Tensor [1,H,W] float32 in [0,1] on same device.
    """
    if points.ndim != 2 or points.shape[-1] < 2:
        raise ValueError("points must be [N,3] or [N,>=2]")

    device = points.device
    # Determine output grid regardless of valid points
    H = W = int(round(2 * params.D / params.g))
    if H <= 0:
        return torch.zeros((1, 0, 0), device=device, dtype=torch.float32)

    # Optional z-filtering per contract
    if points.shape[1] >= 3:
        zmask = points[:, 2].abs() <= params.D
        pts_xy = points[zmask, :2]
    else:
        pts_xy = points[:, :2]

    rows, cols, _, _ = _compute_grid_indices(pts_xy, params)
    if rows.numel() == 0:
        return torch.zeros((1, H, W), device=device, dtype=torch.float32)

    # Flat binning: idx = row*W + col
    flat_idx = rows * W + cols
    bincount = torch.bincount(flat_idx, minlength=H * W)

    # Clip counts per cell to clamp_per_cell
    clamp = int(params.clamp_per_cell)
    bincount = torch.clamp(bincount, max=clamp)

    # Normalize to [0,1] by clamp
    if clamp > 0:
        bev = bincount.to(torch.float32) / float(clamp)
    else:
        bev = bincount.to(torch.float32)

    bev = bev.view(H, W)
    return bev.unsqueeze(0).to(device=device, dtype=torch.float32)
