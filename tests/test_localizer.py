import numpy as np
import torch

from bevplace import REIN
from bevplace.core.types import BEVParams
from bevplace.pipeline.localizer import BEVLocalizer
from bevplace.preprocess.bev import bev_density_image_torch
from bevplace.retrieval import BEVIndex


def test_localizer_retrieval_only_cpu():
    device = "cpu"
    # Build query bev and descriptor
    pts = torch.randn(100000, 3, device=device) * 20.0
    params = BEVParams(D=40.0, g=0.4)
    bev = bev_density_image_torch(pts, params)

    model = REIN().to(device).eval()
    with torch.no_grad():
        _, _, q_desc = model(bev.unsqueeze(0))

    # Build small DB around q_desc
    base = q_desc.detach().cpu().numpy()
    DB = base + np.random.default_rng(0).normal(scale=1e-3, size=base.shape)
    DB = np.vstack([DB for _ in range(5)])
    index = BEVIndex(pca_dim=16)
    index.fit_pca(DB)
    index.add(DB, poses=None)

    localizer = BEVLocalizer(model=model, index=index, bev_params=params, device=device)
    result = localizer.localize(pts)
    assert result.matched_id == 0
    assert result.topk[0] == 0
    assert result.descriptor_q is not None


def test_localizer_with_pose_estimation_cpu():
    device = "cpu"
    params = BEVParams(D=20.0, g=0.5)

    # Create a deterministic BEV and corresponding features
    pts = torch.randn(80000, 3, device=device) * 10.0
    bev = bev_density_image_torch(pts, params)

    model = REIN().to(device).eval()
    with torch.no_grad():
        out1, rem_map_q, q_desc = model(bev.unsqueeze(0))

    base = q_desc.detach().cpu().numpy()
    DB = base.copy()

    index = BEVIndex(pca_dim=8)
    index.fit_pca(DB)
    index.add(DB, poses=None)

    # Reference provider returns the same rem map as the match
    def provider_fn(_id: int):
        return bev, rem_map_q

    localizer = BEVLocalizer(
        model=model,
        index=index,
        bev_params=params,
        device=device,
        reference_provider=provider_fn,
    )
    result = localizer.localize(pts)
    assert result.matched_id == 0
    assert result.num_matches > 0
    assert result.inliers_ratio >= 0.9
    # pose relative should be close to identity
    T = np.array(result.pose_relative)
    assert np.allclose(T, np.eye(3), atol=1e-1)
