from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class REM(nn.Module):
    """Rotation-Equivariant Module based on ResNet-34 up to conv3_x.

    Args:
        from_scratch: If True, do not load ImageNet pretrained weights.
        rotations: Number of rotations (N_R) for equivariance pooling.
    """

    def __init__(self, from_scratch: bool = False, rotations: int = 8) -> None:
        super().__init__()
        weights = None if from_scratch else ResNet34_Weights.DEFAULT
        encoder = models.resnet34(weights=weights)
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)
        self.rotations = int(rotations)
        self.register_buffer(
            "angles",
            -torch.arange(0, 359.00001, 360.0 / self.rotations) / 180.0 * torch.pi,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor [B, 1, H, W]
        Returns:
            out1: Tensor [B, C, H/4, W/4] (for NetVLAD)
            out2: Tensor [B, C, H, W] (for keypoints)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        equ_features = []
        for ang in self.angles:
            # input warp
            aff = torch.zeros(B, 2, 3, device=device, dtype=dtype)
            aff[:, 0, 0] = torch.cos(-ang)
            aff[:, 0, 1] = torch.sin(-ang)
            aff[:, 1, 0] = -torch.sin(-ang)
            aff[:, 1, 1] = torch.cos(-ang)
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True).type_as(x)
            warped = F.grid_sample(x, grid, align_corners=True, mode="bicubic")
            if warped.shape[1] == 1:
                warped = warped.repeat(1, 3, 1, 1)

            # cnn backbone feature
            out = self.encoder(warped)
            if len(equ_features) == 0:
                im1_init_size = out.size()

            # rotate back
            aff = torch.zeros(B, 2, 3, device=device, dtype=dtype)
            aff[:, 0, 0] = torch.cos(ang)
            aff[:, 0, 1] = torch.sin(ang)
            aff[:, 1, 0] = -torch.sin(ang)
            aff[:, 1, 1] = torch.cos(ang)
            grid = F.affine_grid(aff, torch.Size(im1_init_size), align_corners=True).type_as(x)
            out = F.grid_sample(out, grid, align_corners=True, mode="bicubic")
            equ_features.append(out.unsqueeze(-1))

        equ = torch.cat(equ_features, dim=-1)  # [B, C_feat, H', W', R]
        equ = torch.max(equ, dim=-1, keepdim=False)[0]

        # Identity affine
        aff_id = (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        # upsample for NetVLAD to H/4, W/4 of input
        C_feat = equ.shape[1]
        grid = F.affine_grid(aff_id, torch.Size((B, C_feat, H // 4, W // 4)), align_corners=True).type_as(x)
        out1 = F.grid_sample(equ, grid, align_corners=True, mode="bicubic")
        out1 = F.normalize(out1, dim=1)

        # upsample for keypoints to H, W of input
        grid = F.affine_grid(aff_id, torch.Size((B, C_feat, H, W)), align_corners=True).type_as(x)
        out2 = F.grid_sample(equ, grid, align_corners=True, mode="bicubic")
        out2 = F.normalize(out2, dim=1)
        return out1, out2
