from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phobert_cnn.model import PhoBERTCNN, PhoBERTClassifier  # noqa: E402


MODEL_NAME = "vinai/phobert-base"
DEFAULT_RESULTS = pd.DataFrame(
    [
        {"Mô hình": "PhoBERT-CNN (2)", "Accuracy": 0.9913, "Precision": 0.99, "Recall": 0.99, "F1-score": 0.99},
        {"Mô hình": "PhoBERT", "Accuracy": 0.9873, "Precision": 0.99, "Recall": 0.99, "F1-score": 0.99},
        {"Mô hình": "PhoBERT-CNN (3)", "Accuracy": 0.9747, "Precision": 0.98, "Recall": 0.97, "F1-score": 0.97},
    ]
)


st.set_page_config(page_title="PhoBERT-CNN | AI Text Detector", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink: #17212b; --muted: #68737d; --accent: #d95d39; --cream: #f7f3eb; }
    .stApp { background: var(--cream); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,243,235,0.92); }
    .hero { padding: 2.8rem 0 1.4rem; border-bottom: 1px solid #ded8ce; }
    .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { font-family: Georgia, serif; font-size: clamp(2.2rem, 5vw, 4.8rem); line-height: .98; margin: .4rem 0 1rem; max-width: 780px; }
    .hero p { color: var(--muted); font-size: 1.08rem; max-width: 680px; }
    .metric-card { background: #fffdf9; border: 1px solid #ded8ce; border-radius: 8px; padding: 1.1rem 1.25rem; min-height: 112px; }
    .metric-label { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: var(--accent); font-family: Georgia, serif; font-size: 2.1rem; font-weight: 700; margin-top: .35rem; }
    .result { background: #17212b; border-radius: 8px; padding: 1.5rem; color: white; }
    .result-label { color: #f4b09b; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; }
    .result h2 { color: white; font-family: Georgia, serif; margin: .35rem 0; }
    .result p { color: #d9e0e5; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_results() -> pd.DataFrame:
    candidates = [ROOT / "artifacts" / "comparison-10k" / "comparison.json", ROOT / "artifacts" / "comparison" / "comparison.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(rows)
            return frame.rename(
                columns={
                    "model": "Mô hình",
                    "accuracy": "Accuracy",
                    "precision_weighted": "Precision",
                    "recall_weighted": "Recall",
                    "f1_weighted": "F1-score",
                }
            )
        except (OSError, ValueError, TypeError):
            break
    return DEFAULT_RESULTS


@st.cache_resource(show_spinner=False)
def load_inference(model_type: str, checkpoint: str) -> tuple[Any, Any, torch.device]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model_type == "PhoBERT":
        model = PhoBERTClassifier(model_name=MODEL_NAME)
    else:
        kernel_sizes = (2,) if "kernel 2" in model_type else (3,)
        model = PhoBERTCNN(model_name=MODEL_NAME, kernel_sizes=kernel_sizes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, torch.device("cpu")


def predict(text: str, model: Any, tokenizer: Any, device: torch.device) -> tuple[str, list[float]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded)["logits"], dim=1)[0].tolist()
    label = "AI" if probabilities[1] >= probabilities[0] else "Human"
    return label, probabilities


st.markdown(
    '<div class="hero"><div class="eyebrow">AI text detection · Vietnamese NLP</div>'
    '<h1>Văn bản này được viết bởi ai?</h1>'
    '<p>Thử nghiệm mô hình PhoBERT và PhoBERT-CNN trên một đoạn văn tiếng Việt. Kết quả chỉ mang tính hỗ trợ phân tích, không phải bằng chứng tuyệt đối về nguồn gốc văn bản.</p></div>',
    unsafe_allow_html=True,
)

results = load_results()
best_row = results.iloc[results["Accuracy"].astype(float).argmax()]
metric_columns = st.columns(3)
for column, label, value in zip(metric_columns, ["Mô hình tốt nhất", "Accuracy cao nhất", "Số mô hình"], [best_row["Mô hình"], f'{float(best_row["Accuracy"]):.2%}', str(len(results))]):
    column.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Phân tích một văn bản")
    model_type = st.selectbox("Mô hình suy luận", ["PhoBERT", "PhoBERT-CNN (2)", "PhoBERT-CNN (3)"])
    default_checkpoint = ROOT / "artifacts" / "comparison-pilot" / "cnn" / "best_model.pt"
    checkpoint_input = st.text_input("Đường dẫn checkpoint (.pt)", value=str(default_checkpoint) if default_checkpoint.exists() else "")
    text = st.text_area("Văn bản cần kiểm tra", height=220, placeholder="Dán một đoạn văn tiếng Việt vào đây...")
    run = st.button("Phân tích văn bản", type="primary", use_container_width=True)

    if run:
        if not text.strip():
            st.warning("Vui lòng nhập văn bản trước khi phân tích.")
        elif not checkpoint_input or not Path(checkpoint_input).exists():
            st.info("Chưa có checkpoint. Hãy chạy lệnh train/compare trong README rồi nhập đường dẫn file .pt.")
        else:
            try:
                with st.spinner("Đang tải tokenizer và mô hình..."):
                    model, tokenizer, device = load_inference(model_type, checkpoint_input)
                    label, probabilities = predict(text.strip(), model, tokenizer, device)
                st.markdown(f'<div class="result"><div class="result-label">Kết luận</div><h2>{label}</h2><p>Mô hình: {model_type} · {len(text.split())} từ</p></div>', unsafe_allow_html=True)
                st.write("")
                st.progress(float(max(probabilities)), text=f"Độ tin cậy: {max(probabilities):.2%}")
                st.bar_chart(pd.DataFrame({"Xác suất": probabilities}, index=["Human", "AI"]), horizontal=True)
            except Exception as error:
                st.error(f"Không thể tải checkpoint hoặc dự đoán: {error}")

with right:
    st.subheader("So sánh thực nghiệm")
    display = results.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        if column in display:
            display[column] = display[column].astype(float).map(lambda value: f"{value:.2%}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("Các chỉ số được lấy từ comparison.json nếu artifact tồn tại; nếu chưa có, app hiển thị kết quả đã ghi trong báo cáo.")
    st.markdown("**Quy trình**")
    st.markdown("`Input text` → `PhoBERT tokenizer` → `Encoder/CNN` → `Human / AI`")

st.divider()
st.caption("PhoBERT-CNN · Đồ án phân loại văn bản tiếng Việt · Seed 42 · Max length 256")