from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import clean_dataset, load_source, select_balanced_labels, select_subset, split_dataset, tokenize_splits
from .model import PhoBERTCNN, PhoBERTClassifier, TinyEncoder
from .training import evaluate, fit, train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a PhoBERT-CNN text classifier")
    parser.add_argument("--source", default="ICCIES-2025-DetectAI/vietnamese_news_human_ai", help="HF dataset name or local CSV/JSON path")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--text-column", default="Text")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-label", type=int, default=5000, help="Balanced source subset size per label")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional train cap; 0 uses the complete train split")
    parser.add_argument("--validation-limit", type=int, default=0, help="Optional validation cap; 0 uses the complete validation split")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional test cap; 0 uses the complete test split")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny local model without downloading PhoBERT")
    parser.add_argument("--compare", action="store_true", help="Run PhoBERT and CNN ablations and export comparison metrics")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_smoke(args: argparse.Namespace) -> None:
    from datasets import Dataset
    dataset = Dataset.from_dict({"Text": [f"sample text {index}" for index in range(60)], "Label": [index % 2 for index in range(60)]})
    splits = split_dataset(clean_dataset(dataset), seed=args.seed)
    class SimpleTokenizer:
        def __call__(self, texts, **_: object):
            return {"input_ids": [[(sum(ord(char) for char in text) + offset) % 255 for offset in range(8)] for text in texts], "attention_mask": [[1] * 8 for _ in texts]}
    tokenized = tokenize_splits(splits, SimpleTokenizer(), max_length=8)
    model = PhoBERTCNN(pretrained=False, encoder=TinyEncoder(), kernel_sizes=(2, 3, 4))
    device = torch.device("cpu")
    loaders = [DataLoader(part, batch_size=4) for part in (tokenized.train, tokenized.validation, tokenized.test)]
    fit(model, loaders[0], loaders[1], device, epochs=1, learning_rate=1e-3, output_dir=args.output_dir)
    print(json.dumps(evaluate(model, loaders[2], device), indent=2))


def run_training(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    source = clean_dataset(load_source(args.source, args.text_column, args.label_column), args.text_column, args.label_column)
    source = select_balanced_labels(source, args.samples_per_label, args.label_column, args.seed)
    print(f"Balanced dataset: {len(source)} rows | label 0: {source[args.label_column].count(0)} | label 1: {source[args.label_column].count(1)}")
    splits = split_dataset(source, args.label_column, args.seed)
    splits = type(splits)(
        select_subset(splits.train, args.train_limit, args.seed),
        select_subset(splits.validation, args.validation_limit, args.seed),
        select_subset(splits.test, args.test_limit, args.seed),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_splits(splits, tokenizer, args.max_length, args.text_column, args.label_column)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = [DataLoader(part, batch_size=args.batch_size, shuffle=index == 0) for index, part in enumerate((tokenized.train, tokenized.validation, tokenized.test))]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.compare:
        run_comparison(args, tokenized, loaders, device)
        return
    model = PhoBERTCNN(model_name=args.model_name).to(device)
    metrics = train_and_evaluate(model, loaders[0], loaders[1], loaders[2], device, args.epochs, args.learning_rate, args.output_dir)
    (Path(args.output_dir) / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def run_comparison(args: argparse.Namespace, tokenized: object, loaders: list[DataLoader], device: torch.device) -> None:
    """Run all requested ablations on identical data splits and save a flat table."""
    del tokenized
    experiments = [
        ("PhoBERT", lambda: PhoBERTClassifier(model_name=args.model_name)),
        ("PhoBERT-CNN (2)", lambda: PhoBERTCNN(model_name=args.model_name, kernel_sizes=(2,))),
        ("PhoBERT-CNN (3)", lambda: PhoBERTCNN(model_name=args.model_name, kernel_sizes=(3,))),
    ]
    rows = []
    for name, create_model in experiments:
        print(f"\n===== {name} =====")
        experiment_dir = Path(args.output_dir) / name.lower().replace(" ", "_").replace(",", "")
        model = create_model()
        metrics = train_and_evaluate(model.to(device), loaders[0], loaders[1], loaders[2], device, args.epochs, args.learning_rate, experiment_dir)
        rows.append({
            "model": name,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        })
    comparison_path = Path(args.output_dir) / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "loss", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
        writer.writeheader()
        writer.writerows(rows)
    (Path(args.output_dir) / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    if args.smoke:
        run_smoke(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
