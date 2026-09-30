from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from backend.predictor import MAX_LENGTH, MODEL_SPECS, ROOT, PredictionService


app = Flask(__name__, static_folder=None)
prediction_service = PredictionService()


@app.get("/")
def index():
    return send_from_directory(ROOT / "frontend", "index.html")


@app.get("/assets/<path:filename>")
def frontend_assets(filename: str):
    return send_from_directory(ROOT / "frontend", filename)


@app.get("/api/health")
def health():
    available = sum(model["available"] for model in prediction_service.list_models())
    return jsonify({"status": "ok", "available_models": available, "total_models": len(MODEL_SPECS)})


@app.get("/api/models")
def models():
    return jsonify({"models": prediction_service.list_models(), "max_length": MAX_LENGTH})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body phải là JSON object."}), 400

    text = payload.get("text")
    if not isinstance(text, str):
        return jsonify({"error": "Trường 'text' phải là chuỗi."}), 400

    try:
        return jsonify(prediction_service.predict(text))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Không thể tải model hoặc hoàn tất prediction."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)