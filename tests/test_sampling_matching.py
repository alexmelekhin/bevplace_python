import numpy as np
import pytest
import torch

from bevplace.pose.sampling import sample_descriptors
from bevplace.pose.matching import match_descriptors


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_sample_descriptors_shape_and_norm(device):
    C, H, W = 16, 64, 64
    rem_map = torch.randn(1, C, H, W, device=device)
    # 20 random keypoints inside the image
    kps = np.stack(
        [
            np.random.randint(0, H, size=(20,)),
            np.random.randint(0, W, size=(20,)),
        ],
        axis=1,
    ).astype(np.int64)

    desc = sample_descriptors(rem_map, kps)
    assert desc.shape == (20, C)
    # Check near unit norm
    norms = np.linalg.norm(desc, axis=1)
    assert np.all(norms > 0.0)
    assert np.allclose(norms.mean(), 1.0, atol=1e-1)


def test_match_descriptors_identity_and_limits():
    rng = np.random.default_rng(0)
    N, C = 50, 32
    base = rng.normal(size=(N, C)).astype(np.float32)
    other = base + rng.normal(scale=1e-4, size=(N, C)).astype(np.float32)

    matches = match_descriptors(base, other, ratio=0.9, mutual=True, max_matches=None)
    # Expect one-to-one with same indices due to tiny noise
    assert len(matches) == N
    for i, j, d in matches:
        assert i == j
        assert d >= 0.0

    # Enforce max_matches cap
    matches_cap = match_descriptors(base, other, ratio=0.9, mutual=True, max_matches=10)
    assert len(matches_cap) == 10
