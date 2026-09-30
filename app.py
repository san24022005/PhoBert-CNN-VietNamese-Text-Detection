from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phobert_cnn.inference import load_models, predict_all


st.set_page_config(
    page_title="Phân loại văn bản tiếng Việt do AI tạo",
    page_icon="📄",
    layout="centered",
)

st.title("PHÂN LOẠI VĂN BẢN TIẾNG VIỆT DO AI TẠO")
st.write("Nhập văn bản để xem dự đoán từ ba mô hình đã huấn luyện.")


@st.cache_resource(show_spinner="Đang tải tokenizer và ba model đã huấn luyện...")
def cached_models():
    return load_models()


try:
    models, tokenizer, device = cached_models()
except Exception as error:
    st.error(f"Lỗi khởi tạo model: {error}")
    st.stop()

st.caption(f"Thiết bị suy luận: {device.type.upper()} · PhoBERT · max length {256} tokens")

with st.form("inference_form"):
    text = st.text_area(
        "Văn bản cần phân loại",
        height=220,
        max_chars=50000,
        placeholder="Nhập văn bản cần phân loại...",
    )
    submitted = st.form_submit_button("Phân loại văn bản", type="primary")

if submitted:
    if not text.strip():
        st.warning("Vui lòng nhập văn bản trước khi phân loại.")
    else:
        try:
            results, was_truncated = predict_all(text.strip(), models, tokenizer, device)
        except Exception as error:
            st.error(f"Không thể phân loại văn bản: {error}")
        else:
            st.subheader("KẾT QUẢ PHÂN LOẠI")
            columns = st.columns(len(results))
            for column, (name, result) in zip(columns, results.items()):
                with column:
                    st.markdown(f"**{name}**")
                    st.metric("Dự đoán", result["label"])
                    st.caption(f"Confidence: {result['confidence']:.2%}")
                    for label in ("Human", "AI"):
                        probability = result["probabilities"][label]
                        st.progress(probability, text=f"{label}: {probability:.2%}")

            if was_truncated:
                st.warning(
                    "Văn bản vượt quá 256 tokens. Các model sử dụng 256 tokens đầu, "
                    "giống cấu hình preprocessing khi huấn luyện."
                )

            majority_label, votes = Counter(
                result["label"] for result in results.values()
            ).most_common(1)[0]
            st.subheader("TỔNG HỢP")
            for name, result in results.items():
                st.write(f"{name} → {result['label']}")
            st.markdown(f"**Kết quả đa số: {majority_label} ({votes}/3 model)**")
            st.caption("Đây là tổng hợp các dự đoán riêng lẻ, không phải một model ensemble.")