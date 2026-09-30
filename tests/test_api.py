import torch

from backend import app as app_module
from backend.predictor import MAX_LENGTH


def test_predict_api_runs_all_models_with_independent_confidence(monkeypatch):
    captured = {}

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            captured.setdefault("calls", []).append((text, kwargs))
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
            return {"input_ids": list(range(MAX_LENGTH + 1))}

    class FakeModel:
        def __init__(self, logits):
            self.logits = logits

        def __call__(self, **_kwargs):
            return {"logits": torch.tensor([self.logits])}

    models = {
        "PhoBERT": FakeModel([0.0, 2.0]),
        "PhoBERT + CNN-2": FakeModel([0.0, 3.0]),
        "PhoBERT + CNN-3": FakeModel([4.0, 0.0]),
    }
    monkeypatch.setattr(app_module.prediction_service, "_load", lambda: (models, FakeTokenizer(), torch.device("cpu"), {}))
    response = app_module.app.test_client().post(
        "/api/predict",
        json={"text": "  Văn bản thử nghiệm  "},
    )

    assert response.status_code == 200
    payload = response.get_json()
    results = payload["results"]
    assert [result["model_id"] for result in results] == ["phobert", "phobert-cnn-2", "phobert-cnn-3"]
    assert [result["label"] for result in results] == ["AI-generated", "AI-generated", "Human"]
    assert len({result["confidence"] for result in results}) == 3
    assert all(result["confidence"] == result["probabilities"][result["label"]] for result in results)
    assert payload["was_truncated"] is True
    assert captured["calls"][0] == ("Văn bản thử nghiệm", {"add_special_tokens": True, "truncation": False})
    assert captured["calls"][1] == ("Văn bản thử nghiệm", {
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
        "max_length": MAX_LENGTH,
    })


def test_predict_api_keeps_other_results_when_one_model_fails(monkeypatch):
    class FakeTokenizer:
        def __call__(self, _text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
            return {"input_ids": [1, 2]}

    class FakeModel:
        def __init__(self, fails=False):
            self.fails = fails

        def __call__(self, **_kwargs):
            if self.fails:
                raise RuntimeError("inference error")
            return {"logits": torch.tensor([[0.0, 2.0]])}

    models = {
        "PhoBERT": FakeModel(),
        "PhoBERT + CNN-2": FakeModel(fails=True),
        "PhoBERT + CNN-3": FakeModel(),
    }
    monkeypatch.setattr(app_module.prediction_service, "_load", lambda: (models, FakeTokenizer(), torch.device("cpu"), {}))
    response = app_module.app.test_client().post("/api/predict", json={"text": "Văn bản thử nghiệm"})

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert "confidence" in results[0]
    assert "inference error" in results[1]["error"]
    assert "confidence" in results[2]


def test_predict_api_rejects_blank_text():
    response = app_module.app.test_client().post(
        "/api/predict",
        json={"text": "  "},
    )

    assert response.status_code == 400
    assert "Vui lòng nhập văn bản" in response.get_json()["error"]


def test_frontend_assets_are_served_by_api():
    client = app_module.app.test_client()

    page = client.get("/")
    script = client.get("/assets/script.js")
    stylesheet = client.get("/assets/style.css")

    assert page.status_code == 200
    assert b"/assets/script.js" in page.data
    assert script.status_code == 200
    assert stylesheet.status_code == 200