from datasets import Dataset
import torch

from phobert_cnn.data import clean_dataset, select_balanced_labels, split_dataset
from phobert_cnn.model import PhoBERTCNN, PhoBERTClassifier, TextCNN, TinyEncoder


def test_clean_dataset_removes_empty_and_duplicate_texts():
    dataset = Dataset.from_dict({"Text": [" A ", "A", "", None, "B"], "Label": [0, 0, 1, 1, 1]})
    cleaned = clean_dataset(dataset)
    assert cleaned["Text"] == ["A", "B"]


def test_split_is_stratified_and_disjoint():
    dataset = Dataset.from_dict({"Text": [f"text {i}" for i in range(40)], "Label": [i % 2 for i in range(40)]})
    splits = split_dataset(dataset)
    assert len(splits.train) + len(splits.validation) + len(splits.test) == 40
    assert set(splits.train["Text"]).isdisjoint(splits.test["Text"])


def test_select_balanced_labels_returns_requested_counts():
    dataset = Dataset.from_dict({"Text": [f"text {i}" for i in range(20)], "Label": [i % 2 for i in range(20)]})
    balanced = select_balanced_labels(dataset, samples_per_label=5)
    assert len(balanced) == 10
    assert balanced["Label"].count(0) == 5
    assert balanced["Label"].count(1) == 5


def test_phobert_cnn_output_shape():
    model = PhoBERTCNN(encoder=TinyEncoder(vocab_size=64, hidden_size=16), pretrained=False)
    output = model(torch.randint(0, 64, (2, 8)), torch.ones(2, 8, dtype=torch.long), torch.tensor([0, 1]))
    assert output["logits"].shape == (2, 2)
    assert output["loss"].ndim == 0


def test_phobert_baseline_output_shape():
    model = PhoBERTClassifier(encoder=TinyEncoder(vocab_size=64, hidden_size=16), pretrained=False)
    output = model(torch.randint(0, 64, (2, 8)), torch.ones(2, 8, dtype=torch.long), torch.tensor([0, 1]))
    assert output["logits"].shape == (2, 2)
    assert output["loss"].ndim == 0


def test_text_cnn_baseline_output_shape():
    model = TextCNN(vocab_size=64, embedding_size=16)
    output = model(torch.randint(0, 64, (2, 8)), torch.ones(2, 8, dtype=torch.long), torch.tensor([0, 1]))
    assert output["logits"].shape == (2, 2)
    assert output["loss"].ndim == 0
