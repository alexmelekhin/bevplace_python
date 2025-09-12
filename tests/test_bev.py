from __future__ import annotations

import numpy as np
import pytest
import torch

from bevplace.core.types import BEVParams
from bevplace.preprocess.bev import bev_density_image_torch


def test_bev_size_and_bounds():
    params = BEVParams(D=40.0, g=0.4)
    # create points at corners and center
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [39.9, 0.0, 0.0],
            [-39.9, 0.0, 0.0],
            [0.0, 39.9, 0.0],
            [0.0, -39.9, 0.0],
        ],
        dtype=np.float32,
    )
    img = bev_density_image_torch(torch.from_numpy(pts), params)
    assert img.shape[0] == 1
    H, W = img.shape[-2], img.shape[-1]
    # For D=40, g=0.4: H=W = floor(40/0.4) - floor(-40/0.4) + 1 = 100 - (-100) + 1 = 201
    assert H == 201 and W == 201
    assert torch.isfinite(img).all()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bev_shape_and_device(device):
    params = BEVParams(D=40.0, g=0.4)
    H = W = 201
    pts = torch.randn(1000, 3, device=device) * 10.0
    out = bev_density_image_torch(pts, params)
    assert out.shape == (1, H, W)
    assert out.device.type == device
    assert out.dtype == torch.float32


def test_bev_empty_returns_zeros_cpu():
    params = BEVParams(D=40.0, g=0.4)
    H = W = 201
    pts = torch.empty(0, 3)
    out = bev_density_image_torch(pts, params)
    assert out.shape == (1, H, W)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("cap", [1, 5, 10])
def test_bev_value_range(cap):
    params = BEVParams(D=2.0, g=1.0, increment_cap=cap)
    # Place multiple points in same cell to trigger cap
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [0.05, 0.05, 0.0],
            [0.02, -0.02, 0.0],
            [0.01, 0.01, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [0.05, 0.05, 0.0],
            [0.02, -0.02, 0.0],
            [0.01, 0.01, 0.0],
        ]
    )
    out = bev_density_image_torch(pts, params)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0 + 1e-6


def test_bev_z_filtering():
    params = BEVParams(D=5.0, g=1.0)
    # Two points same XY; one with |z|>D should be dropped
    pts = torch.tensor(
        [
            [0.0, 0.0, 10.0],  # filtered out by z
            [0.0, 0.0, 0.0],  # kept
        ]
    )
    out = bev_density_image_torch(pts, params)
    assert out.max().item() > 0.0


def test_grid_bounds():
    params = BEVParams(D=2.0, g=1.0)
    H = W = 5
    # Points exactly at bounds should map inside
    pts = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
        ]
    )
    out = bev_density_image_torch(pts, params)
    assert out.shape == (1, H, W)
    assert out.max().item() > 0.0
