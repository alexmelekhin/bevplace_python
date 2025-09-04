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
    # row = floor(( D - x) / g) in [0, H-1], col = floor(( y + D) / g) in [0, W-1]
    rows = torch.floor((D - pts[:, 0]) / g).to(dtype=torch.long)
    cols = torch.floor((pts[:, 1] + D) / g).to(dtype=torch.long)

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    return rows[valid], cols[valid], H, W


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
    dtype = points.dtype
    pts_xy = points[:, :2]

    rows, cols, H, W = _compute_grid_indices(pts_xy, params)
    if H == 0 or rows.numel() == 0:
        return torch.zeros((1, 0, 0), device=device, dtype=torch.float32)

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
