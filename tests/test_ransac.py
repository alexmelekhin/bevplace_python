import math

import numpy as np

from bevplace.pose.ransac import estimate_rigid_transform_ransac


def make_transform(theta_rad: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    out = (T @ pts_h.T).T[:, :2]
    return out


def test_ransac_recovers_transform_with_noise():
    rng = np.random.default_rng(0)
    N = 200
    pts_q = rng.uniform(low=-50, high=50, size=(N, 2)).astype(np.float32)
    theta = math.radians(20.0)
    T_gt = make_transform(theta, tx=5.0, ty=-3.0)
    pts_m = apply_transform(T_gt, pts_q)
    pts_m += rng.normal(scale=0.5, size=pts_m.shape).astype(np.float32)

    T_est, report = estimate_rigid_transform_ransac(
        pts_q, pts_m, pixel_thresh=2.0, max_iters=2000, confidence=0.99
    )
    assert report["inliers"] >= int(0.8 * N)
    # Compare rotation angle and translation
    est_theta = math.atan2(T_est[1, 0], T_est[0, 0])
    assert abs((est_theta - theta + math.pi) % (2 * math.pi) - math.pi) < math.radians(2.0)
    assert np.allclose(T_est[:2, 2], T_gt[:2, 2], atol=1.0)


def test_ransac_handles_outliers():
    rng = np.random.default_rng(1)
    N_in = 150
    N_out = 50
    pts_q_in = rng.uniform(low=-30, high=30, size=(N_in, 2)).astype(np.float32)
    theta = math.radians(-15.0)
    T_gt = make_transform(theta, tx=-7.0, ty=4.0)
    pts_m_in = apply_transform(T_gt, pts_q_in)
    pts_m_in += rng.normal(scale=0.3, size=pts_m_in.shape).astype(np.float32)

    # add outliers
    pts_q = np.vstack([pts_q_in, rng.uniform(low=-50, high=50, size=(N_out, 2)).astype(np.float32)])
    pts_m = np.vstack([pts_m_in, rng.uniform(low=-50, high=50, size=(N_out, 2)).astype(np.float32)])

    T_est, report = estimate_rigid_transform_ransac(
        pts_q, pts_m, pixel_thresh=2.0, max_iters=5000, confidence=0.99
    )
    assert report["inliers"] >= int(0.6 * (N_in + N_out))  # tolerate outliers
    est_theta = math.atan2(T_est[1, 0], T_est[0, 0])
    assert abs((est_theta - theta + math.pi) % (2 * math.pi) - math.pi) < math.radians(3.0)
    assert np.allclose(T_est[:2, 2], T_gt[:2, 2], atol=1.0)


def test_ransac_graceful_on_too_few_points():
    pts_q = np.zeros((1, 2), dtype=np.float32)
    pts_m = np.zeros((1, 2), dtype=np.float32)
    T_est, report = estimate_rigid_transform_ransac(
        pts_q, pts_m, pixel_thresh=1.0, max_iters=100, confidence=0.9
    )
    assert np.allclose(T_est, np.eye(3, dtype=np.float32))
    assert report["inliers"] == 0
