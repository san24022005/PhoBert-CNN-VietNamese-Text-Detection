from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phobert_cnn.inference import MAX_LENGTH, MODEL_SPECS as INFERENCE_MODEL_SPECS  # noqa: E402
from phobert_cnn.inference import load_models, predict_all  # noqa: E402


MODEL_SPECS = {
    model_id: {"name": name, "checkpoint": checkpoint}
    for model_id, (name, checkpoint, _kind, _kernel_sizes) in zip(
        ("phobert", "phobert-cnn-2", "phobert-cnn-3"), INFERENCE_MODEL_SPECS
    )
}


class PredictionService:
    def __init__(self) -> None:
        self._loaded: tuple[dict[str, Any], Any, Any, dict[str, str]] | None = None
        self._load_lock = threading.Lock()

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "name": spec["name"],
                "available": spec["checkpoint"].is_file(),
            }
            for model_id, spec in MODEL_SPECS.items()
        ]

    def _load(self) -> tuple[dict[str, Any], Any, Any, dict[str, str]]:
        if self._loaded is not None:
            return self._loaded

        with self._load_lock:
            if self._loaded is None:
                errors: dict[str, str] = {}
                models, tokenizer, device = load_models(errors=errors)
                self._loaded = models, tokenizer, device, errors
        return self._loaded

    def predict(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Vui lòng nhập văn bản trước khi phân tích.")

        models, tokenizer, device, errors = self._load()
        results, was_truncated = predict_all(
            cleaned_text,
            models,
            tokenizer,
            device,
            errors=errors,
        )
        model_results = []
        for model_id, spec in MODEL_SPECS.items():
            model_name = spec["name"]
            if model_name not in results:
                model_results.append({
                    "model": model_name,
                    "model_id": model_id,
                    "error": errors.get(model_name, "Model chưa có kết quả."),
                })
                continue

            result = results[model_name]
            model_results.append({
                "model": model_name,
                "model_id": model_id,
                "label": "AI-generated" if result["label"] == "AI" else "Human",
                "confidence": result["confidence"],
                "probabilities": {
                    "AI-generated": result["probabilities"]["AI"],
                    "Human": result["probabilities"]["Human"],
                },
            })
        return {
            "results": model_results,
            "was_truncated": was_truncated,
        }