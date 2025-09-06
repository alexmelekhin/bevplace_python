from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from bevplace.core.types import BEVParams
from bevplace.pose.matching import match_descriptors
from bevplace.pose.ransac import estimate_rigid_transform_ransac
from bevplace.pose.sampling import sample_descriptors
from bevplace.preprocess.bev import bev_density_image_torch
from bevplace.retrieval import BEVIndex

ReferenceProvider = Callable[[int], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class LocalizationResult:
    pose_global: Optional[np.ndarray]
    pose_relative: np.ndarray
    matched_id: int
    topk: Tuple[int, ...]
    distances: Tuple[float, ...]
    inliers_ratio: float
    num_matches: int
    timings: dict
    descriptor_q: Optional[np.ndarray]


class BEVLocalizer:
    def __init__(
        self,
        model: torch.nn.Module,
        index: BEVIndex,
        bev_params: BEVParams,
        device: str | torch.device = "cpu",
        reference_provider: Optional[ReferenceProvider] = None,
    ) -> None:
        self.model = model
        self.index = index
        self.params = bev_params
        self.device = torch.device(device)
        self.reference_provider = reference_provider

    @torch.no_grad()
    def localize(self, point_cloud: np.ndarray | torch.Tensor, k: int = 1) -> LocalizationResult:
        timings: dict = {}
        t0 = time.perf_counter()

        # LiDAR -> BEV
        if isinstance(point_cloud, np.ndarray):
            pts = torch.from_numpy(point_cloud).to(self.device)
        else:
            pts = point_cloud.to(self.device)
        bev_img = bev_density_image_torch(pts, self.params).to(self.device)
        timings["bev_ms"] = (time.perf_counter() - t0) * 1000

        # REIN forward
        t1 = time.perf_counter()
        self.model.eval().to(self.device)
        out1, rem_map_q, desc = self.model(bev_img.unsqueeze(0))
        desc_np = desc.detach().cpu().numpy()
        timings["rein_ms"] = (time.perf_counter() - t1) * 1000

        # Retrieval
        t2 = time.perf_counter()
        hits = self.index.search(desc_np, k=k)
        top_ids = tuple(h.id for h in hits[0])
        top_dist = tuple(h.distance for h in hits[0])
        matched_id = top_ids[0]
        timings["retrieval_ms"] = (time.perf_counter() - t2) * 1000

        # Optional pose estimation if provider is available
        inliers_ratio = 0.0
        num_matches = 0
        T_rel = np.eye(3, dtype=np.float32)
        if self.reference_provider is not None:
            t3 = time.perf_counter()
            bev_m, rem_map_m = self.reference_provider(matched_id)
            # Simple keypoint grid for demo matching
            H, W = bev_img.shape[-2], bev_img.shape[-1]
            step = max(4, int(round(self.params.g / 0.2)))
            rows = np.arange(0, H, step)
            cols = np.arange(0, W, step)
            kps_rc = np.array([(r, c) for r in rows for c in cols], dtype=np.int64)

            desc_q = sample_descriptors(rem_map_q, kps_rc)
            desc_m = sample_descriptors(rem_map_m, kps_rc)
            matches = match_descriptors(desc_q, desc_m, ratio=0.9, mutual=True, max_matches=2000)
            num_matches = len(matches)
            if num_matches >= 2:
                pts_q = np.array([[r, c] for (r, c) in kps_rc[[m[0] for m in matches]]], dtype=np.float32)
                pts_m = np.array([[r, c] for (r, c) in kps_rc[[m[1] for m in matches]]], dtype=np.float32)
                T_rel, report = estimate_rigid_transform_ransac(
                    pts_q, pts_m, pixel_thresh=2.0, max_iters=2000, confidence=0.99
                )
                inliers_ratio = report["inliers"] / max(num_matches, 1)
            timings["pose_ms"] = (time.perf_counter() - t3) * 1000

        return LocalizationResult(
            pose_global=None,
            pose_relative=T_rel.astype(np.float32),
            matched_id=matched_id,
            topk=top_ids,
            distances=top_dist,
            inliers_ratio=float(inliers_ratio),
            num_matches=int(num_matches),
            timings=timings,
            descriptor_q=desc_np,
        )
