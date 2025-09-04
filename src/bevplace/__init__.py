"""BEVPlace++ inference package (work in progress)."""

import pathlib
import re

from .preprocess.bev import bev_density_image_torch  # noqa: F401


def _get_version():
    pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    version_pattern = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']')
    try:
        with pyproject_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version"):
                    match = version_pattern.match(line.strip())
                    if match:
                        return match.group(1)
    except Exception:
        pass
    return "0.0.0"


__version__ = _get_version()
