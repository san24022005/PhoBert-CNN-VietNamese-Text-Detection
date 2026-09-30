const form = document.querySelector("#predictForm");
const textInput = document.querySelector("#textInput");
const submitButton = document.querySelector("#submitButton");
const submitLabel = document.querySelector("#submitLabel");
const clearButton = document.querySelector("#clearButton");
const errorMessage = document.querySelector("#errorMessage");
const connectionStatus = document.querySelector("#connectionStatus");
const comparisonSection = document.querySelector("#comparisonSection");
const resultCards = [...document.querySelectorAll(".result-card")];
const truncationNote = document.querySelector("#truncationNote");

function updateCounts() {
  const text = textInput.value;
  const trimmed = text.trim();
  document.querySelector("#charCount").textContent = Array.from(text).length;
  document.querySelector("#wordCount").textContent = trimmed ? trimmed.split(/\s+/u).length : 0;
}

function setConnection(online) {
  connectionStatus.classList.toggle("is-online", online);
  connectionStatus.classList.toggle("is-offline", !online);
  connectionStatus.lastChild.textContent = online ? " API sẵn sàng" : " API không kết nối";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.classList.toggle("is-loading", loading);
  submitLabel.textContent = loading ? "Đang chạy 3 model..." : "Phân loại văn bản";
  submitButton.setAttribute("aria-busy", String(loading));
  if (loading) {
    comparisonSection.hidden = false;
    resultCards.forEach((card) => {
      card.classList.remove("has-error");
      card.classList.add("is-loading");
      card.querySelector('[data-field="prediction"]').textContent = "Đang chạy...";
      card.querySelector('[data-field="metrics"]').hidden = true;
      card.querySelector('[data-field="error"]').hidden = true;
      card.querySelector(".card-state").textContent = "Đang suy luận";
    });
  }
}

function formatProbability(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderResults(payload) {
  const resultsByModel = new Map(payload.results.map((result) => [result.model_id, result]));
  comparisonSection.hidden = false;
  truncationNote.hidden = !payload.was_truncated;

  resultCards.forEach((card) => {
    const result = resultsByModel.get(card.dataset.modelId);
    const prediction = card.querySelector('[data-field="prediction"]');
    const metrics = card.querySelector('[data-field="metrics"]');
    const error = card.querySelector('[data-field="error"]');
    card.classList.remove("is-loading", "has-error");

    if (!result || result.error) {
      card.classList.add("has-error");
      card.querySelector(".card-state").textContent = "Không khả dụng";
      prediction.textContent = "Model gặp lỗi";
      metrics.hidden = true;
      error.textContent = result?.error || "Backend không trả kết quả cho model này.";
      error.hidden = false;
      return;
    }

    card.querySelector(".card-state").textContent = "Hoàn tất";
    prediction.textContent = result.label;
    card.querySelector('[data-field="confidence"]').textContent = formatProbability(result.confidence);
    card.querySelector('[data-field="ai-value"]').textContent = formatProbability(result.probabilities["AI-generated"]);
    card.querySelector('[data-field="human-value"]').textContent = formatProbability(result.probabilities.Human);
    card.querySelector('[data-field="ai-bar"]').style.width = formatProbability(result.probabilities["AI-generated"]);
    card.querySelector('[data-field="human-bar"]').style.width = formatProbability(result.probabilities.Human);
    metrics.hidden = false;
    error.hidden = true;
  });
}

async function checkConnection() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Không thể kiểm tra backend.");
    const data = await response.json();
    if (data.available_models === 0) {
      showError("Chưa tìm thấy checkpoint model. Kiểm tra thư mục pytorch/.");
      submitButton.disabled = true;
      setConnection(false);
      return;
    }
    submitButton.disabled = false;
    setConnection(true);
  } catch (error) {
    setConnection(false);
    showError(error.message || "Không thể kết nối backend. Hãy kiểm tra máy chủ Python.");
    submitButton.disabled = true;
  }
}

textInput.addEventListener("input", () => {
  updateCounts();
  errorMessage.hidden = true;
});

clearButton.addEventListener("click", () => {
  textInput.value = "";
  updateCounts();
  errorMessage.hidden = true;
  textInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  if (!textInput.value.trim()) {
    showError("Vui lòng nhập văn bản trước khi phân tích.");
    textInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textInput.value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction không thành công.");
    renderResults(result);
    setConnection(true);
  } catch (error) {
    const message = error.message || "Không thể kết nối backend. Vui lòng thử lại.";
    showError(message);
    resultCards.forEach((card) => {
      if (!card.classList.contains("is-loading")) return;
      card.classList.remove("is-loading");
      card.classList.add("has-error");
      card.querySelector(".card-state").textContent = "Lỗi yêu cầu";
      card.querySelector('[data-field="prediction"]').textContent = "Không có kết quả";
      card.querySelector('[data-field="error"]').textContent = message;
      card.querySelector('[data-field="error"]').hidden = false;
    });
    if (error instanceof TypeError) setConnection(false);
  } finally {
    setLoading(false);
  }
});

updateCounts();
checkConnection();