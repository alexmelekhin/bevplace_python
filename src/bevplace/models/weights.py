from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch

from bevplace.models.rein import REIN


# Official BEVPlace2 checkpoint (best model) URL
# Source: https://github.com/zjuluolun/BEVPlace2 (runs/Aug08_10-17-29/model_best.pth.tar)
DEFAULT_WEIGHTS_URL = (
    "https://github.com/zjuluolun/BEVPlace2/raw/refs/heads/main/runs/Aug08_10-17-29/model_best.pth.tar"
)


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    d = base / "bevplace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_checkpoint(url: str, dest_dir: Optional[Path] = None, timeout: int = 20) -> Path:
    import urllib.request

    dest_dir = dest_dir or _default_cache_dir()
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12] + "_model_best.pth.tar"
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    with urllib.request.urlopen(url, timeout=timeout) as r:  # nosec - user-approved URL
        data = r.read()
    dest.write_bytes(data)
    return dest


def load_state_dict_into_rein(model: REIN, ckpt_path: Path, map_location: Optional[str] = None) -> None:
    ml = map_location or ("cpu" if not torch.cuda.is_available() else None)
    checkpoint = torch.load(str(ckpt_path), map_location=ml)
    state = checkpoint.get("state_dict", checkpoint)

    # Some checkpoints may have 'module.' prefix (DataParallel)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module.") :]] = v
        else:
            new_state[k] = v

    # Load non-strict to allow minor naming differences
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # It's okay to ignore missing/unexpected; caller can decide to warn/log if needed


def ensure_default_weights(model: REIN, quiet: bool = False) -> bool:
    """Try to download and load the official BEVPlace2 checkpoint into model.

    Falls back silently if download/load fails.
    """
    try:
        path = download_checkpoint(DEFAULT_WEIGHTS_URL)
        load_state_dict_into_rein(model, path)
        if not quiet:
            print("Loaded official BEVPlace2 weights", flush=True)
        return True
    except Exception as _:
        if not quiet:
            print("Warning: failed to load official weights; proceeding without pretrained.", flush=True)
        return False
