"""Phase 2: the CNN models, their data loaders and a .predict() wrapper for the shared evaluate().

Architectures and settings: docs/decisions.md D-034. CPU, seeded, N_JOBS threads: D-036. Wrapper: D-037.
This module never builds a test loader and never reads X_test (AGENTS.md §4, rules 8-11).
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import load_splits


def _use_cpu_threads() -> None:
    """Same thread count as the Phase 1 sklearn models, so timings are comparable (D-036)."""
    torch.set_num_threads(config.N_JOBS)


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """(N, 28, 28) float32 images -> (N, 1, 28, 28) tensor: a CNN expects a channel dimension (1 = grayscale)."""
    return torch.from_numpy(images).unsqueeze(1)


# --- 1. Data -------------------------------------------------------------------------

def make_loaders(batch_size: int = config.CNN_BATCH_SIZE) -> dict:
    """Train (shuffled, seeded) and val DataLoaders from the fixed Phase 1 split. There is no test loader."""
    splits = load_splits()
    train = TensorDataset(_to_tensor(splits["X_train"]), torch.from_numpy(splits["y_train"]).long())
    val = TensorDataset(_to_tensor(splits["X_val"]), torch.from_numpy(splits["y_val"]).long())
    shuffle_order = torch.Generator().manual_seed(config.SEED)  # same batch order on every run
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, generator=shuffle_order, num_workers=0),
        "val": DataLoader(val, batch_size=config.CNN_PREDICT_BATCH, shuffle=False, num_workers=0),
    }


# --- 2. Models -------------------------------------------------------------------------

def _conv_block(in_channels: int, out_channels: int) -> list:
    """3x3 convolution (padding 1 keeps the size) + ReLU + 2x2 max pooling (halves height and width)."""
    return [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]


class CNN2(nn.Sequential):
    """2 conv blocks: 1x28x28 -> 32x14x14 -> 64x7x7 -> 3,136 -> 128 -> 10. 421,642 parameters."""

    def __init__(self):
        super().__init__(
            *_conv_block(1, 32),
            *_conv_block(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(128, config.NUM_CLASSES),
        )


class CNN3(nn.Sequential):
    """3 conv blocks: ... -> 128x3x3 -> 1,152 -> 128 -> 10. 241,546 parameters (fewer: the dense layer is smaller)."""

    def __init__(self):
        super().__init__(
            *_conv_block(1, 32),
            *_conv_block(32, 64),
            *_conv_block(64, 128),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(128, config.NUM_CLASSES),
        )


def count_params(model: nn.Module) -> int:
    """Number of trainable weights and biases."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- 3. Prediction wrapper for evaluate() ----------------------------------------------------

class CNNPredictor:
    """Gives a trained network the .predict(X) that evaluate() expects, like an sklearn model (D-037)."""

    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 28, 28) float32 images -> (N,) int64 class indices. Dropout off, no gradients, 1,000 images at a time."""
        if X.ndim != 3 or X.shape[1:] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
            raise ValueError(f"Expected images of shape (N, 28, 28), got {X.shape}")
        _use_cpu_threads()
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), config.CNN_PREDICT_BATCH):
                batch = _to_tensor(np.ascontiguousarray(X[start:start + config.CNN_PREDICT_BATCH]))
                preds.append(self.model(batch).argmax(dim=1).numpy())
        return np.concatenate(preds).astype(np.int64)
