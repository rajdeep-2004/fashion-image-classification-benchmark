"""Small helpers shared by every phase: seeding, timing, saving figures and JSON."""
import json
import random
import time

import numpy as np
import torch

from src import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python's, NumPy's and PyTorch's random generators so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Timer:
    """Measure wall-clock time of a block.

    with Timer("fit") as t:
        model.fit(X, y)
    print(t.seconds)
    """

    def __init__(self, label: str):
        self.label = label
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.seconds = time.perf_counter() - self._start
        print(f"{self.label}: {self.seconds:.1f}s")
        return False  # never hide an exception raised inside the block


def display_path(path) -> str:
    """Path relative to the repo root for short messages, or the full path if it lives elsewhere (e.g. Google Drive)."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def save_fig(fig, name: str):
    """Save a matplotlib figure as results/figures/{name}.png at 300 dpi and return the path."""
    if not name or "/" in name or name.endswith(".png"):
        raise ValueError(f"Figure name must be a plain name without folder or extension, got {name!r}")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    print(f"saved {display_path(path)}")
    return path


def _to_plain(value):
    """json.dump fallback: turn NumPy numbers/arrays into plain Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot save {type(value).__name__} to JSON")


def save_json(obj: dict, path) -> None:
    """Write a dict to a JSON file (creating the folder), converting NumPy types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_to_plain)
    print(f"saved {display_path(path)}")
