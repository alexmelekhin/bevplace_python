from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, Field, validator


class BEVParams(BaseModel):
    """Parameters for BEV density image generation.

    Attributes:
        D: Half-size of the square crop window in meters (range [-D, D]).
        g: Grid resolution in meters per pixel.
        clamp_per_cell: Legacy clamp parameter (kept for backward compatibility).
        voxel_size: Voxel downsampling size in meters. If None, uses ``g``.
        increment_cap: Per-cell increment cap before normalization (default 10 as upstream).
        per_image_max_normalize: Whether to scale intensities so image max equals 255.
    """

    D: float = Field(default=40.0, gt=0.0)
    g: float = Field(default=0.4, gt=0.0)
    clamp_per_cell: int = Field(default=10, ge=1)
    voxel_size: float | None = Field(default=None)
    increment_cap: int = Field(default=10, ge=1)
    per_image_max_normalize: bool = Field(default=True)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Return (H, W) using floor-based grid sizing to match upstream.

        For bounds x,y in [-D, D] and grid size ``g``, the image side length is:
        ``floor(D/g) - floor(-D/g) + 1``.
        """
        import math

        x_min_ind = math.floor(-self.D / self.g)
        x_max_ind = math.floor(self.D / self.g)
        size = (x_max_ind - x_min_ind) + 1
        return int(size), int(size)

    @validator("D", "g")
    def finite_positive(cls, v: float) -> float:  # noqa: D401
        """Ensure values are finite and positive."""
        if not (v > 0.0):  # also excludes NaN/inf
            raise ValueError("must be > 0")
        return float(v)

    @validator("voxel_size")
    def voxel_positive_if_set(cls, v: float | None) -> float | None:  # noqa: D401
        """Ensure voxel size, when provided, is positive."""
        if v is None:
            return None
        if not (v > 0.0):
            raise ValueError("voxel_size must be > 0 if set")
        return float(v)


class Pose2D(BaseModel):
    """2D pose in meters and radians."""

    x_m: float
    y_m: float
    yaw_rad: float


class Transform2D(BaseModel):
    """SE(2) transform as 3x3 row-major list for serialization."""

    mat_3x3: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
