from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitDatasets:
    train: Dataset
    validation: Dataset
    test: Dataset


def load_source(source: str, text_column: str = "Text", label_column: str = "Label") -> Dataset:
    """Load a Hugging Face dataset name or a local CSV/JSON file."""
    path = Path(source)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=suffix == ".jsonl")
        else:
            raise ValueError(f"Unsupported local file type: {suffix}")
        return Dataset.from_pandas(frame[[text_column, label_column]], preserve_index=False)

    loaded = load_dataset(source)
    if isinstance(loaded, DatasetDict):
        if "train" not in loaded:
            raise ValueError("DatasetDict must contain a 'train' split")
        return loaded["train"]
    return loaded


def clean_dataset(dataset: Dataset, text_column: str = "Text", label_column: str = "Label") -> Dataset:
    """Remove null/empty rows and duplicate texts before splitting."""
    frame = dataset.to_pandas()
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=[text_column, label_column]).copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame = frame[frame[text_column] != ""]
    frame = frame.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    frame[label_column] = frame[label_column].astype(int)
    if frame[label_column].nunique() < 2:
        raise ValueError("Classification requires at least two distinct labels")
    return Dataset.from_pandas(frame[[text_column, label_column]], preserve_index=False)


def select_balanced_labels(
    dataset: Dataset,
    samples_per_label: int = 5000,
    label_column: str = "Label",
    seed: int = 42,
) -> Dataset:
    """Select exactly ``samples_per_label`` rows for labels 0 and 1."""
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive")
    labels = [int(label) for label in dataset[label_column]]
    available = {label: labels.count(label) for label in sorted(set(labels))}
    missing = {label: samples_per_label - available.get(label, 0) for label in (0, 1) if available.get(label, 0) < samples_per_label}
    if missing:
        raise ValueError(f"Not enough rows for balanced sampling: {missing}")

    selected_indices: list[int] = []
    for label in (0, 1):
        label_indices = [index for index, value in enumerate(labels) if value == label]
        rng = np.random.default_rng(seed + label)
        selected_indices.extend(rng.choice(label_indices, size=samples_per_label, replace=False).tolist())
    np.random.default_rng(seed).shuffle(selected_indices)
    return dataset.select(selected_indices)


def split_dataset(
    dataset: Dataset,
    label_column: str = "Label",
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitDatasets:
    """Create stratified 70/15/15-style train, validation and test splits."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")
    held_out = validation_size + test_size
    indices = list(range(len(dataset)))
    labels = dataset[label_column]
    train_indices, held_out_indices = train_test_split(
        indices,
        test_size=held_out,
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_size / held_out
    validation_indices, test_indices = train_test_split(
        held_out_indices,
        test_size=relative_test,
        random_state=seed,
        stratify=[labels[index] for index in held_out_indices],
    )
    return SplitDatasets(
        train=dataset.select(train_indices),
        validation=dataset.select(validation_indices),
        test=dataset.select(test_indices),
    )


def select_subset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(size))


def tokenize_splits(
    splits: SplitDatasets,
    tokenizer: Any,
    max_length: int = 256,
    text_column: str = "Text",
    label_column: str = "Label",
) -> SplitDatasets:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length)

    def transform(dataset: Dataset) -> Dataset:
        tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing")
        tokenized = tokenized.rename_column(label_column, "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    return SplitDatasets(transform(splits.train), transform(splits.validation), transform(splits.test))
