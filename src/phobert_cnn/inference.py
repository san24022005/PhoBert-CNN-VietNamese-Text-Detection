from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import PhoBERTCNN, PhoBERTClassifier


MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256
LABELS = {0: "Human", 1: "AI"}
ROOT = Path(__file__).resolve().parents[2]

MODEL_SPECS = (
    ("PhoBERT", ROOT / "pytorch" / "PhoBERT.pt", "phobert", None),
    ("PhoBERT + CNN-2", ROOT / "pytorch" / "PhoBERT___CNN-2.pt", "cnn", (2,)),
    ("PhoBERT + CNN-3", ROOT / "pytorch" / "PhoBERT___CNN-3.pt", "cnn", (3,)),
)


def _read_state_dict(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint}. "
            "Hãy đặt đúng file .pt vào thư mục pytorch/ của dự án."
        )

    try:
        payload: Any = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Không đọc được checkpoint {checkpoint}. "
            "Ứng dụng yêu cầu state_dict hoặc checkpoint có khóa 'model_state_dict'; "
            "không thể nạp an toàn model object đã pickle."
        ) from error

    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping) or not payload or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        raise TypeError(
            f"Checkpoint {checkpoint} không chứa state_dict dạng mapping tên-tensor."
        )
    return payload


def load_models() -> tuple[dict[str, nn.Module], Any, torch.device]:
    """Load each trained state_dict once and return eval-mode models and tokenizer."""
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Thiếu transformers. Cài dependencies bằng: pip install -r requirements.txt"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cpu" else torch.float32
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Không tải được config/tokenizer PhoBERT ({MODEL_NAME}). "
            "Kiểm tra kết nối mạng hoặc cache Hugging Face."
        ) from error

    models: dict[str, nn.Module] = {}
    for name, checkpoint, kind, kernel_sizes in MODEL_SPECS:
        try:
            state_dict = _read_state_dict(checkpoint)
            with torch.device("meta"):
                encoder = AutoModel.from_config(config)
                if kind == "phobert":
                    model: nn.Module = PhoBERTClassifier(
                        model_name=MODEL_NAME,
                        encoder=encoder,
                    )
                else:
                    model = PhoBERTCNN(
                        model_name=MODEL_NAME,
                        kernel_sizes=kernel_sizes,
                        encoder=encoder,
                    )
            model.load_state_dict(state_dict, strict=True, assign=True)
            embeddings = model.encoder.embeddings
            position_ids = torch.arange(embeddings.position_ids.shape[1]).unsqueeze(0)
            embeddings.position_ids = position_ids
            embeddings.token_type_ids = torch.zeros_like(position_ids)
            model.to(device=device, dtype=model_dtype)
            del state_dict
            model.eval()
        except Exception as error:
            raise RuntimeError(
                f"Không thể nạp {name} từ {checkpoint}: {error}. "
                "Kiểm tra checkpoint và kiến trúc model tương ứng trong model.py."
            ) from error
        models[name] = model

    return models, tokenizer, device


def predict_all(
    text: str,
    models: Mapping[str, nn.Module],
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not text.strip():
        raise ValueError("Văn bản rỗng. Hãy nhập nội dung tiếng Việt trước khi phân loại.")

    try:
        full_tokens = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        was_truncated = len(full_tokens) > MAX_LENGTH
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    except Exception as error:
        raise RuntimeError(f"Tokenizer PhoBERT không xử lý được văn bản: {error}") from error

    encoded = {key: value.to(device) for key, value in encoded.items()}
    results: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for name, model in models.items():
            output = model(**encoded)
            probabilities = torch.softmax(output["logits"].float(), dim=-1)[0].cpu().tolist()
            predicted_id = int(torch.tensor(probabilities).argmax().item())
            results[name] = {
                "label": LABELS[predicted_id],
                "confidence": probabilities[predicted_id],
                "probabilities": {
                    LABELS[index]: probability
                    for index, probability in enumerate(probabilities)
                },
            }
    return results, was_truncated