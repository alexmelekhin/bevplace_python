import pytest
import torch

from bevplace.models.netvlad import NetVLAD
from bevplace.models.rem import REM
from bevplace.models.rein import REIN


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_rem_forward_shapes(device):
    B, H, W = 2, 200, 200
    x = torch.rand(B, 1, H, W, device=device)
    rem = REM(from_scratch=True).to(device)
    out1, out2 = rem(x)
    assert out1.shape[0] == B and out2.shape[0] == B
    # channels should be 128 (ResNet34 conv3_x output)
    assert out1.shape[1] == out2.shape[1] == 128
    assert out2.shape[2:] == (H, W)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_netvlad_forward(device):
    B, C, H4, W4 = 2, 128, 50, 50
    feats = torch.rand(B, C, H4, W4, device=device)
    vlad = NetVLAD().to(device)
    desc = vlad(feats)
    assert desc.shape == (B, 64 * 128)
    assert desc.device.type == device


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_rein_forward(device):
    B, H, W = 2, 200, 200
    x = torch.rand(B, 1, H, W, device=device)
    model = REIN().to(device)
    out1, out2, desc = model(x)
    assert out1.shape[0] == out2.shape[0] == B
    assert out2.shape[2:] == (H, W)
    assert desc.shape == (B, 64 * 128)
