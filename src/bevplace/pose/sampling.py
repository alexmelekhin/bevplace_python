from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sample_descriptors(rem_map: torch.Tensor, keypoints_rc: np.ndarray | torch.Tensor) -> np.ndarray:
    """Sample L2-normalized descriptors from REM feature map at integer pixel keypoints.

    Args:
        rem_map: Tensor [1, C, H, W] on CPU/GPU.
        keypoints_rc: array-like [N, 2] of (row, col) in pixel coords.
    Returns:
        np.ndarray [N, C] float32, L2-normalized.
    """
    if rem_map.ndim != 4 or rem_map.shape[0] != 1:
        raise ValueError("rem_map must be [1, C, H, W]")
    H, W = rem_map.shape[2], rem_map.shape[3]

    if isinstance(keypoints_rc, torch.Tensor):
        kps = keypoints_rc.detach().cpu().to(torch.long).numpy()
    else:
        kps = np.asarray(keypoints_rc, dtype=np.int64)

    # Clamp to image bounds
    kps[:, 0] = np.clip(kps[:, 0], 0, H - 1)
    kps[:, 1] = np.clip(kps[:, 1], 0, W - 1)

    # Convert to normalized grid coords in [-1,1]: x=col, y=row
    xs = (kps[:, 1] / (W - 1)) * 2.0 - 1.0
    ys = (kps[:, 0] / (H - 1)) * 2.0 - 1.0
    grid = torch.from_numpy(np.stack([xs, ys], axis=1)).to(rem_map.device, dtype=rem_map.dtype)
    grid = grid.view(1, -1, 1, 2)  # [1, N, 1, 2]

    # Grid sample expects [N,H,W,2]; use align_corners=True to match affine grids
    samples = F.grid_sample(rem_map, grid, mode="bilinear", align_corners=True)
    # samples: [1, C, N, 1]
    desc = samples[0, :, :, 0].transpose(0, 1)  # [N, C]
    desc = F.normalize(desc, dim=1).detach().cpu().numpy().astype(np.float32)
    return desc
