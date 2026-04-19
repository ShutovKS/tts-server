const presetList = document.getElementById("preset-list");
const textInput = document.getElementById("tts-text");
const charCounter = document.getElementById("char-counter");
const synthesizeButton = document.getElementById("synthesize-button");
const previewButton = document.getElementById("preview-button");
const statusOverlay = document.getElementById("status-overlay");
const audioPanel = document.getElementById("audio-panel");
const resultAudio = document.getElementById("result-audio");
const downloadLink = document.getElementById("download-link");

const searchParams = new URLSearchParams(window.location.search);
const apiBaseUrl = (searchParams.get("apiBaseUrl") || "http://127.0.0.1:8000").replace(/\/$/, "");

let presets = [];
let selectedPresetId = null;
let currentObjectUrl = null;
let runtimeSupportsClone = false;
let previewAudio = new Audio();

function apiUrl(path) { return `${apiBaseUrl}${path}`; }

function setStatus(message, type = "error") {
  if (!message) {
    statusOverlay.hidden = true;
    return;
  }
  statusOverlay.hidden = false;
  statusOverlay.textContent = message;
  statusOverlay.className = `status-overlay ${type}`;
}

function setBusy(isBusy) {
  textInput.disabled = isBusy;
  synthesizeButton.disabled = isBusy;
  previewButton.disabled = isBusy;

  if (isBusy) {
    synthesizeButton.classList.add("is-busy");
  } else {
    synthesizeButton.classList.remove("is-busy");
  }

  presetList.querySelectorAll("button").forEach(btn => btn.disabled = isBusy);
  if (isBusy) setStatus("ГЕНЕРАЦИЯ...", "info");
}

function selectPreset(presetId) {
  selectedPresetId = presetId;
  presetList.querySelectorAll(".preset-card").forEach((button) => {
    const isSelected = button.dataset.presetId === presetId;
    button.classList.toggle("is-selected", isSelected);
    button.setAttribute("aria-checked", String(isSelected));
  });
  
  // Останавливаем превью при смене пресета
  if (!previewAudio.paused) {
    previewAudio.pause();
    previewAudio.currentTime = 0;
  }
}

function renderPresets(items) {
  presetList.innerHTML = "";
  items.forEach((preset, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-card";
    button.dataset.presetId = preset.id;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", "false");
    
    // Вёрстка карточки в стиле референса
    button.innerHTML = `
      <img src="${preset.avatar}" alt="${preset.label}" class="avatar-img" />
      <span class="preset-name">${preset.label}</span>
      <span class="tech-decals">SYS_${preset.language}</span>
    `;
    
    button.addEventListener("click", () => selectPreset(preset.id));
    presetList.appendChild(button);
    if (index === 0) selectPreset(preset.id);
  });
}

async function loadVoicePresets() {
  try {
    const response = await fetch("./voice-presets.json");
    const payload = await response.json();
    presets = payload;
    renderPresets(presets);
  } catch (error) {
    setStatus("ОШИБКА КОНФИГА ПРЕСЕТОВ", "error");
  }
}

async function loadRuntimeStatus() {
  try {
    const response = await fetch(apiUrl("/health/ready"));
    const payload = await response.json();
    runtimeSupportsClone = Boolean(payload?.checks?.models?.backend_capabilities?.supports_clone);
    
    if (!runtimeSupportsClone) {
      setStatus("CLONE-РЕЖИМ НЕДОСТУПЕН", "error");
      setBusy(true);
    }
  } catch (error) {
    setStatus("ОШИБКА ПОДКЛЮЧЕНИЯ К СЕРВЕРУ", "error");
  }
}

async function submitClone() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus("ВВЕДИТЕ ТЕКСТ", "error");
    textInput.focus();
    return;
  }

  const preset = presets.find((item) => item.id === selectedPresetId);
  if (!preset) return;

  // Очистка старого аудио и скрытие панели
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  audioPanel.hidden = true;
  resultAudio.removeAttribute("src");

  setBusy(true);

  try {
    const formData = new FormData();
    formData.set("text", text);
    if (preset.language) formData.set("language", preset.language);
    if (preset.referenceText) formData.set("ref_text", preset.referenceText);
    if (preset.model) formData.set("model", preset.model);
    
    // Получаем файл пресета
    const refAudio = await fetch(preset.referenceAudioPath);
    const audioBlob = await refAudio.blob();
    formData.set("ref_audio", audioBlob, `${preset.id}.wav`);

    const response = await fetch(apiUrl("/api/v1/tts/clone"), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("СБОЙ СИНТЕЗА");

    const resultBlob = await response.blob();
    currentObjectUrl = URL.createObjectURL(resultBlob);
    
    // Показываем панель с аудио
    resultAudio.src = currentObjectUrl;
    downloadLink.href = currentObjectUrl;
    audioPanel.hidden = false;
    
    setStatus(""); // Успех, убираем ошибки
  } catch (error) {
    setStatus(error.message || "СИСТЕМНАЯ ОШИБКА", "error");
  } finally {
    setBusy(false);
  }
}

// Обработчик кнопки
synthesizeButton.addEventListener("click", submitClone);

// Обработчик превью
previewButton.addEventListener("click", () => {
  const preset = presets.find((item) => item.id === selectedPresetId);
  if (!preset) return;

  if (previewAudio.src && !previewAudio.src.endsWith(preset.referenceAudioPath) || !previewAudio.src) {
      previewAudio.src = preset.referenceAudioPath;
  }

  if (previewAudio.paused) {
    previewAudio.play();
  } else {
    previewAudio.pause();
    previewAudio.currentTime = 0;
  }
});

// Обработчик Enter (без Shift) для быстрого синтеза
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault(); // отменяем перенос строки
    submitClone();
  }
});

textInput.addEventListener("input", () => {
  const currentLength = textInput.value.length;
  const maxLength = Number(textInput.getAttribute("maxlength") || 0);

  charCounter.textContent = `[ ${currentLength} / ${maxLength} ]`;
  charCounter.classList.toggle("max-reached", currentLength >= maxLength);
});

// Инициализация
Promise.all([loadRuntimeStatus(), loadVoicePresets()]);
