from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, Field, validator


class BEVParams(BaseModel):
    """Parameters for BEV density image generation.

    Attributes:
        D: Half-size of the square crop window in meters (range [-D, D]).
        g: Grid resolution in meters per pixel.
        clamp_per_cell: Maximum count per cell before normalization.
    """

    D: float = Field(default=40.0, gt=0.0)
    g: float = Field(default=0.4, gt=0.0)
    clamp_per_cell: int = Field(default=10, ge=1)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Return (H, W) computed as round(2*D/g)."""
        size = int(round(2 * self.D / self.g))
        return size, size

    @validator("D", "g")
    def finite_positive(cls, v: float) -> float:  # noqa: D401
        """Ensure values are finite and positive."""
        if not (v > 0.0):  # also excludes NaN/inf
            raise ValueError("must be > 0")
        return float(v)


class Pose2D(BaseModel):
    """2D pose in meters and radians."""

    x_m: float
    y_m: float
    yaw_rad: float


class Transform2D(BaseModel):
    """SE(2) transform as 3x3 row-major list for serialization."""

    mat_3x3: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
