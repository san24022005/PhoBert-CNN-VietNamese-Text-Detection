from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class PhoBERTCNN(nn.Module):
    """PhoBERT contextual features followed by parallel 1D CNN branches."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_classes: int = 2,
        dropout: float = 0.3,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        pretrained: bool = True,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if encoder is None and not pretrained:
            raise ValueError("Pass an encoder when pretrained=False; no local PhoBERT config is bundled")
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(model_name)
        self.encoder = encoder
        hidden_size = int(self.encoder.config.hidden_size)
        self.kernel_sizes = kernel_sizes
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(hidden_size, 128, kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128 * len(kernel_sizes), num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Optional[Tensor]]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = hidden.transpose(1, 2)
        pooled = [torch.relu(conv(features)).amax(dim=2) for conv in self.convolutions]
        logits = self.classifier(self.dropout(torch.cat(pooled, dim=1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class PhoBERTClassifier(nn.Module):
    """PhoBERT-only baseline using masked mean pooling and a linear head."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if encoder is None and not pretrained:
            raise ValueError("Pass an encoder when pretrained=False; no local PhoBERT config is bundled")
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(model_name)
        self.encoder = encoder
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Optional[Tensor]]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class TextCNN(nn.Module):
    """CNN-only baseline over a trainable token embedding."""

    def __init__(
        self,
        vocab_size: int = 64000,
        embedding_size: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_size, 128, kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128 * len(kernel_sizes), num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Optional[Tensor]]:
        del attention_mask
        features = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.relu(conv(features)).amax(dim=2) for conv in self.convolutions]
        logits = self.classifier(self.dropout(torch.cat(pooled, dim=1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class TinyEncoder(nn.Module):
    """Small local encoder used by smoke tests; it avoids downloading PhoBERT."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> object:
        del attention_mask
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()
