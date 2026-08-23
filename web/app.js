// Mini-R1 Interactive Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const modelSelect = document.getElementById("model-select");
  const modelDesc = document.getElementById("model-desc");
  const gpuText = document.getElementById("gpu-text");
  const promptInput = document.getElementById("prompt-input");
  const btnGenerate = document.getElementById("btn-generate");
  const btnClear = document.getElementById("btn-clear");
  const chips = document.querySelectorAll(".chip");

  // Params
  const tempSlider = document.getElementById("temperature");
  const topPSlider = document.getElementById("top_p");
  const repSlider = document.getElementById("rep-penalty");
  const maxTokensSlider = document.getElementById("max-tokens");

  const valTemp = document.getElementById("val-temperature");
  const valTopP = document.getElementById("val-top_p");
  const valRep = document.getElementById("val-rep-penalty");
  const valMaxTokens = document.getElementById("val-max-tokens");

  // Output Elements
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const tabFormatted = document.getElementById("tab-formatted");
  const tabRaw = document.getElementById("tab-raw");
  const tabBtns = document.querySelectorAll(".tab-btn");

  const reasoningBlock = document.getElementById("reasoning-block");
  const outputThink = document.getElementById("output-think");
  const outputAnswer = document.getElementById("output-answer");
  const outputRaw = document.getElementById("output-raw");

  const telemetryBar = document.getElementById("telemetry-bar");
  const telTokens = document.getElementById("tel-tokens");
  const telSpeed = document.getElementById("tel-speed");
  const telLatency = document.getElementById("tel-latency");

  // Model Descriptions Map
  let availableModels = {};

  // 1. Fetch Models and Server Status
  async function initServerStatus() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();

      if (data.gpu_name) {
        gpuText.textContent = `${data.device.toUpperCase()}: ${data.gpu_name}`;
      } else {
        gpuText.textContent = `${data.device.toUpperCase()} Ready`;
      }

      if (data.models) {
        data.models.forEach((m) => {
          availableModels[m.id] = m;
        });
        updateModelDescription();
      }
    } catch (err) {
      console.warn("Could not connect to API:", err);
      gpuText.textContent = "Offline / Error";
    }
  }

  function updateModelDescription() {
    const selected = modelSelect.value;
    if (availableModels[selected]) {
      modelDesc.textContent = availableModels[selected].description;
    }
  }

  modelSelect.addEventListener("change", updateModelDescription);

  // 2. Slider Event Listeners
  tempSlider.addEventListener("input", (e) => {
    valTemp.textContent = parseFloat(e.target.value).toFixed(2);
  });

  topPSlider.addEventListener("input", (e) => {
    valTopP.textContent = parseFloat(e.target.value).toFixed(2);
  });

  repSlider.addEventListener("input", (e) => {
    valRep.textContent = parseFloat(e.target.value).toFixed(2);
  });

  maxTokensSlider.addEventListener("input", (e) => {
    valMaxTokens.textContent = e.target.value;
  });

  // 3. Preset Chip Buttons
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      promptInput.value = chip.getAttribute("data-prompt");
      promptInput.focus();
    });
  });

  // Clear Button
  btnClear.addEventListener("click", () => {
    promptInput.value = "";
    promptInput.focus();
  });

  // 4. Tab Switching
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      if (target === "formatted") {
        tabFormatted.style.display = "flex";
        tabRaw.style.display = "none";
      } else {
        tabFormatted.style.display = "none";
        tabRaw.style.display = "block";
      }
    });
  });

  // 5. Run Generation
  async function runGeneration() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      promptInput.focus();
      return;
    }

    // UI Loading State
    btnGenerate.disabled = true;
    btnGenerate.innerHTML = `<span class="btn-icon">⏳</span> Running...`;
    emptyState.style.display = "none";
    tabFormatted.style.display = "none";
    tabRaw.style.display = "none";
    loadingState.style.display = "block";
    telemetryBar.style.display = "none";

    const payload = {
      model: modelSelect.value,
      prompt: prompt,
      temperature: parseFloat(tempSlider.value),
      top_p: parseFloat(topPSlider.value),
      repetition_penalty: parseFloat(repSlider.value),
      max_tokens: parseInt(maxTokensSlider.value, 10),
    };

    try {
      const startTime = performance.now();
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      const clientLatency = Math.round(performance.now() - startTime);

      if (!data.success) {
        throw new Error(data.error || "Generation error");
      }

      const result = data.result;

      // Populate Outputs
      outputRaw.textContent = result.raw_completion;

      if (result.extracted_think) {
        reasoningBlock.style.display = "block";
        outputThink.textContent = result.extracted_think;
      } else {
        reasoningBlock.style.display = "none";
      }

      if (result.extracted_answer) {
        outputAnswer.textContent = result.extracted_answer;
      } else if (result.fallback_answer) {
        outputAnswer.textContent = `${result.fallback_answer} (Extracted)`;
      } else {
        outputAnswer.textContent = "No answer tag found";
      }

      // Update Telemetry
      telTokens.textContent = result.tokens_generated;
      telSpeed.textContent = result.tok_per_sec;
      telLatency.textContent = clientLatency;
      telemetryBar.style.display = "flex";

      // Show formatted view by default
      loadingState.style.display = "none";
      const activeTab = document.querySelector(".tab-btn.active").getAttribute("data-tab");
      if (activeTab === "formatted") {
        tabFormatted.style.display = "flex";
      } else {
        tabRaw.style.display = "block";
      }
    } catch (err) {
      alert("Inference Failed: " + err.message);
      loadingState.style.display = "none";
      emptyState.style.display = "block";
    } finally {
      btnGenerate.disabled = false;
      btnGenerate.innerHTML = `<span class="btn-icon">⚡</span> Run Model`;
    }
  }

  btnGenerate.addEventListener("click", runGeneration);

  // Keyboard shortcut: Ctrl + Enter / Cmd + Enter
  promptInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runGeneration();
    }
  });

  // Init
  initServerStatus();
});
