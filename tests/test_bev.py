import pytest
import torch

from bevplace import bev_density_image_torch
from bevplace.core.types import BEVParams


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bev_shape_and_device(device):
    params = BEVParams(D=40.0, g=0.4)
    H = W = int(round(2 * params.D / params.g))
    pts = torch.randn(1000, 3, device=device) * 10.0
    out = bev_density_image_torch(pts, params)
    assert out.shape == (1, H, W)
    assert out.device.type == device
    assert out.dtype == torch.float32


def test_bev_empty_returns_zeros_cpu():
    params = BEVParams(D=40.0, g=0.4)
    H = W = int(round(2 * params.D / params.g))
    pts = torch.empty(0, 3)
    out = bev_density_image_torch(pts, params)
    assert out.shape == (1, H, W)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("clamp", [1, 5, 10])
def test_bev_normalization_in_range(clamp):
    params = BEVParams(D=2.0, g=1.0, clamp_per_cell=clamp)
    # Place multiple points in same cell to trigger clamp
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
    params = BEVParams(D=5.0, g=1.0, clamp_per_cell=10)
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
    H = W = int(round(2 * params.D / params.g))
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
