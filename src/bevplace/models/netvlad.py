from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation.

    Args:
        num_clusters: Number of VLAD clusters K.
        dim: Local descriptor dimensionality C.
    """

    def __init__(self, num_clusters: int = 64, dim: int = 128) -> None:
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.dim = int(dim)
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=1, bias=False)
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        self.alpha: Optional[float] = None

    @torch.no_grad()
    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize centroids and the assignment conv from pre-computed clusters.

        Args:
            clsts: (K, C) cluster centers.
            traindescs: (M, C) sample descriptors used to compute alpha.
        """
        clsts = np.asarray(clsts)
        traindescs = np.asarray(traindescs)
        clsts_assign = clsts / (np.linalg.norm(clsts, axis=1, keepdims=True) + 1e-12)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # descending
        self.alpha = float((-np.log(0.01) / np.mean(dots[0, :] - dots[1, :] + 1e-12)))

        self.centroids.data.copy_(torch.from_numpy(clsts))
        weight = torch.from_numpy((self.alpha * clsts_assign).astype(np.float32)).unsqueeze(2).unsqueeze(3)
        self.conv.weight.data.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute VLAD descriptor from local feature map.

        Args:
            x: Tensor [B, C, H, W]
        Returns:
            Tensor [B, K*C] L2-normalized.
        """
        B, C = x.shape[:2]
        x_flat = x.view(B, C, -1)

        soft_assign = self.conv(x).view(B, self.num_clusters, -1)  # [B, K, HW]
        soft_assign = F.softmax(soft_assign, dim=1)

        vlad = torch.zeros((B, self.num_clusters, C), dtype=x.dtype, device=x.device)  # [B, K, C]
        for k in range(self.num_clusters):
            residual = x_flat.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[k : k + 1, :].expand(
                x_flat.size(-1), -1
            ).permute(1, 0).unsqueeze(0)  # [B, C, HW]
            residual = residual * soft_assign[:, k : k + 1, :].unsqueeze(2)
            vlad[:, k : k + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(B, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
