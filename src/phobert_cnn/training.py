from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import classification_metrics


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: Any = None) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if training else "eval"):
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            if loss is None:
                raise RuntimeError("Model did not return a loss")
            if training:
                loss.backward()
                optimizer.step()
            predictions = output["logits"].argmax(dim=1)
            total_loss += loss.item() * batch["labels"].size(0)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return EpochResult(loss=total_loss / total, accuracy=correct / total)


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="predict"):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            total_loss += float(output["loss"].item()) * batch["labels"].size(0)
            total += batch["labels"].size(0)
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
    return labels, predictions, total_loss / total


def fit(model: torch.nn.Module, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device, epochs: int, learning_rate: float, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_loss = float("inf")
    for epoch in range(epochs):
        train_result = run_epoch(model, train_loader, device, optimizer)
        validation_result = run_epoch(model, validation_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} | train loss={train_result.loss:.4f} acc={train_result.accuracy:.4f} | val loss={validation_result.loss:.4f} acc={validation_result.accuracy:.4f}")
        if validation_result.loss < best_loss:
            best_loss = validation_result.loss
            torch.save(model.state_dict(), output_path / "best_model.pt")


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    labels, predictions, loss = predict(model, loader, device)
    result = classification_metrics(labels, predictions)
    result["loss"] = loss
    return result


def train_and_evaluate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train one experiment and return only test metrics for comparison tables."""
    fit(model, train_loader, validation_loader, device, epochs, learning_rate, output_dir)
    checkpoint = Path(output_dir) / "best_model.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return evaluate(model, test_loader, device)
