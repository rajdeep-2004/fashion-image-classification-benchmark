"""Phase 2: the CNN models, their data loaders and a .predict() wrapper for the shared evaluate().

Architectures and settings: docs/decisions.md D-034. CPU, seeded, N_JOBS threads: D-036. Wrapper: D-037.
This module never builds a test loader and never reads X_test (AGENTS.md §4, rules 8-11).
"""
import copy
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import load_splits
from src.utils import display_path, save_json


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


# --- 4. Training with early stopping (D-034, D-035) ----------------------------------------

def training_settings() -> dict:
    """The fixed training settings, stored in the val and test JSONs."""
    return {"batch_size": config.CNN_BATCH_SIZE, "lr": config.CNN_LR, "max_epochs": config.CNN_MAX_EPOCHS,
            "patience": config.CNN_PATIENCE, "dropout": config.CNN_DROPOUT, "optimizer": "adam"}


def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer=None):
    """One pass over the loader. With an optimizer: train (dropout on). Without: score only (dropout off, no gradients).

    Returns (mean loss, accuracy, predictions as a NumPy array).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, preds = 0.0, 0, []
    with torch.set_grad_enabled(training):
        for X, y in loader:
            scores = model(X)
            loss = loss_fn(scores, y)
            if training:
                optimizer.zero_grad()
                loss.backward()   # backpropagation: gradient of the loss for every weight
                optimizer.step()  # Adam update
            total_loss += loss.item() * len(y)
            pred = scores.argmax(dim=1)
            correct += int((pred == y).sum())
            preds.append(pred)
    n = len(loader.dataset)
    return total_loss / n, correct / n, torch.cat(preds).numpy()


def train_cnn(name: str, model: nn.Module, loaders: dict) -> dict:
    """Train on the 54k train split, stop when val loss hasn't improved for CNN_PATIENCE epochs, restore the best weights.

    Writes the checkpoint, the per-epoch history CSV and the val JSON (data-formats.md §13-15); returns the val summary.
    Call utils.set_seed() before building `model` so its starting weights are the same on every run.
    """
    _use_cpu_threads()
    torch.use_deterministic_algorithms(True)
    loss_fn = nn.CrossEntropyLoss()  # softmax + negative log-likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CNN_LR)
    history, best_loss, best_state, epochs_without_gain = [], float("inf"), None, 0
    run_start = time.perf_counter()
    for epoch in range(1, config.CNN_MAX_EPOCHS + 1):
        start = time.perf_counter()
        train_loss, train_acc, _ = _run_epoch(model, loaders["train"], loss_fn, optimizer)
        val_loss, val_acc, _ = _run_epoch(model, loaders["val"], loss_fn)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                        "epoch_s": round(time.perf_counter() - start, 1)})
        print(f"{name} epoch {epoch:2d}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {history[-1]['epoch_s']}s")
        if val_loss < best_loss:
            best_loss, best_state, epochs_without_gain = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.CNN_PATIENCE:
                print(f"early stop: no lower val loss for {config.CNN_PATIENCE} epochs")
                break
    model.load_state_dict(best_state)
    return _save_training(name, model, history, loaders["val"], time.perf_counter() - run_start)


def _save_training(name: str, model: nn.Module, history: list, val_loader: DataLoader, train_time_s: float) -> dict:
    """Score the restored best weights on val, then write the checkpoint, history CSV and val JSON."""
    experiment = f"{name}_images"
    val_loss, val_acc, y_pred = _run_epoch(model, val_loader, nn.CrossEntropyLoss())
    y_val = val_loader.dataset.tensors[1].numpy()
    best = min(history, key=lambda row: row["val_loss"])  # min() keeps the first of equal values, like the loop

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = config.MODELS_DIR / f"{experiment}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"saved {display_path(checkpoint)}")
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.TRAINING_DIR / f"{experiment}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"saved {display_path(history_path)}")

    summary = {
        "experiment": experiment, "model": name, "features": "images", "split": "val",
        "settings": training_settings(), "best_epoch": best["epoch"], "epochs_run": len(history),
        "best_val_loss": round(val_loss, 4), "best_val_acc": round(val_acc, 4),
        "val_macro_f1": round(float(f1_score(y_val, y_pred, average="macro")), 4),
        "train_acc_at_best": best["train_acc"], "param_count": count_params(model),
        "train_time_s": round(train_time_s, 2), "device": "cpu", "torch": torch.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(summary, config.METRICS_DIR / f"{experiment}_val.json")
    print(f"{experiment}: best epoch {summary['best_epoch']} of {summary['epochs_run']}, "
          f"val acc {summary['best_val_acc']:.4f}, val macro F1 {summary['val_macro_f1']:.4f}")
    return summary
