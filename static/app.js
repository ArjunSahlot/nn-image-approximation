/**
 * NN Image Playground — Frontend Logic
 * Network editor, training controls, live visualization, gradient editor
 */

// ─── State ──────────────────────────────────────────────────────────────────

const APP = {
    layers: [],
    ws: null,
    training: false,
    hasModel: false,
    hasImage: false,
    imgWidth: 0,
    imgHeight: 0,
    colorStops: [
        { value: 0, color: '#000000' },
        { value: 1, color: '#ffffff' },
    ],
    lossHistory: [],
};

// Color gradient presets
const PRESETS = {
    grayscale: [
        { value: 0, color: '#000000' },
        { value: 1, color: '#ffffff' },
    ],
    heatmap: [
        { value: 0, color: '#000033' },
        { value: 0.2, color: '#0000ff' },
        { value: 0.4, color: '#00ffff' },
        { value: 0.6, color: '#ffff00' },
        { value: 0.8, color: '#ff6600' },
        { value: 1, color: '#ff0000' },
    ],
    cool: [
        { value: 0, color: '#0d0221' },
        { value: 0.25, color: '#0a4c8a' },
        { value: 0.5, color: '#18a1cd' },
        { value: 0.75, color: '#77e6d0' },
        { value: 1, color: '#e0f7fa' },
    ],
    viridis: [
        { value: 0, color: '#440154' },
        { value: 0.25, color: '#3b528b' },
        { value: 0.5, color: '#21918c' },
        { value: 0.75, color: '#5ec962' },
        { value: 1, color: '#fde725' },
    ],
    plasma: [
        { value: 0, color: '#0d0887' },
        { value: 0.25, color: '#7e03a8' },
        { value: 0.5, color: '#cc4778' },
        { value: 0.75, color: '#f89540' },
        { value: 1, color: '#f0f921' },
    ],
    inferno: [
        { value: 0, color: '#000004' },
        { value: 0.25, color: '#420a68' },
        { value: 0.5, color: '#932667' },
        { value: 0.75, color: '#dd513a' },
        { value: 1, color: '#fcffa4' },
    ],
};

// Layer type configs: what params each type has
const LAYER_PARAMS = {
    linear: [{ key: 'out_features', label: 'Neurons', default: 64, type: 'number' }],
    conv1d: [
        { key: 'out_channels', label: 'Channels', default: 32, type: 'number' },
        { key: 'kernel_size', label: 'Kernel', default: 3, type: 'number' },
    ],
    relu: [],
    leakyrelu: [{ key: 'negative_slope', label: 'Slope', default: 0.01, type: 'number', step: 0.01 }],
    sigmoid: [],
    tanh: [],
    silu: [],
    gelu: [],
    batchnorm: [],
    layernorm: [],
    dropout: [{ key: 'p', label: 'Rate', default: 0.1, type: 'number', step: 0.05 }],
};

// ─── DOM References ─────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    layerList: $('#layer-list'),
    layerSelect: $('#layer-type-select'),
    btnAddLayer: $('#btn-add-layer'),
    btnBuild: $('#btn-build'),
    btnStart: $('#btn-start'),
    btnStop: $('#btn-stop'),
    btnResume: $('#btn-resume'),
    btnApplyColor: $('#btn-apply-color'),
    btnSave: $('#btn-save'),
    btnAddStop: $('#btn-add-stop'),
    dropZone: $('#drop-zone'),
    fileInput: $('#file-input'),
    inputLR: $('#input-lr'),
    inputEpochs: $('#input-epochs'),
    inputBatch: $('#input-batch'),
    inputViz: $('#input-viz'),
    statEpoch: $('#stat-epoch'),
    statLoss: $('#stat-loss'),
    trainStats: $('#train-stats'),
    modelInfo: $('#model-info'),
    paramCount: $('#param-count'),
    canvasOriginal: $('#canvas-original'),
    canvasOutput: $('#canvas-output'),
    canvasColorized: $('#canvas-colorized'),
    gradientPreview: $('#gradient-preview'),
    stopList: $('#stop-list'),
    statusBadge: $('#status-badge'),
    deviceBadge: $('#device-badge'),
};

// ─── Init ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initWebSocket();
    renderStops();
    drawGradientPreview();
    fetchStatus();

    // Add a default simple network
    addLayer('linear', { out_features: 128 });
    addLayer('relu');
    addLayer('linear', { out_features: 256 });
    addLayer('relu');
    addLayer('linear', { out_features: 256 });
    addLayer('relu');
    addLayer('linear', { out_features: 128 });
    addLayer('relu');
    addLayer('linear', { out_features: 1 });
    addLayer('sigmoid');
});

function initEventListeners() {
    DOM.btnAddLayer.addEventListener('click', () => {
        addLayer(DOM.layerSelect.value);
    });

    DOM.btnBuild.addEventListener('click', buildNetwork);
    DOM.btnStart.addEventListener('click', startTraining);
    DOM.btnStop.addEventListener('click', stopTraining);
    DOM.btnResume.addEventListener('click', resumeTraining);
    DOM.btnApplyColor.addEventListener('click', applyColorMap);
    DOM.btnSave.addEventListener('click', saveImage);
    DOM.btnAddStop.addEventListener('click', () => addColorStop(0.5, '#888888'));

    // File upload
    DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());
    DOM.fileInput.addEventListener('change', handleFileSelect);
    DOM.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); DOM.dropZone.classList.add('drag-over'); });
    DOM.dropZone.addEventListener('dragleave', () => DOM.dropZone.classList.remove('drag-over'));
    DOM.dropZone.addEventListener('drop', handleFileDrop);

    // Presets
    $$('.preset-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const preset = PRESETS[btn.dataset.preset];
            if (preset) {
                APP.colorStops = JSON.parse(JSON.stringify(preset));
                renderStops();
                drawGradientPreview();
            }
        });
    });
}

// ─── WebSocket ──────────────────────────────────────────────────────────────

function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    APP.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    APP.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'train_update') {
            DOM.statEpoch.textContent = data.epoch;
            DOM.statLoss.textContent = data.loss.toExponential(4);
            APP.lossHistory.push(data.loss);
            drawLossChart();

            if (data.image) {
                drawBase64OnCanvas(DOM.canvasOutput, data.image, 'L');
            }
        }

        if (data.type === 'train_done') {
            APP.training = false;
            updateTrainingUI();
            DOM.statEpoch.textContent = data.epoch;
            DOM.statLoss.textContent = data.loss.toExponential(4);
            if (data.image) {
                drawBase64OnCanvas(DOM.canvasOutput, data.image, 'L');
            }
            toast('Training complete!', 'success');
        }
    };

    APP.ws.onclose = () => {
        // Reconnect after a short delay
        setTimeout(initWebSocket, 2000);
    };

    APP.ws.onerror = () => {
        // Will trigger onclose
    };
}

// ─── Layer Management ───────────────────────────────────────────────────────

let layerIdCounter = 0;

function addLayer(type, params = null) {
    const id = ++layerIdCounter;
    const layerParams = params || {};

    // Set defaults for any params not provided
    const paramDefs = LAYER_PARAMS[type] || [];
    paramDefs.forEach((pd) => {
        if (!(pd.key in layerParams)) {
            layerParams[pd.key] = pd.default;
        }
    });

    APP.layers.push({ id, type, params: layerParams });
    renderLayers();
}

function removeLayer(id) {
    APP.layers = APP.layers.filter((l) => l.id !== id);
    renderLayers();
}

function moveLayer(id, dir) {
    const idx = APP.layers.findIndex((l) => l.id === id);
    const newIdx = idx + dir;
    if (newIdx < 0 || newIdx >= APP.layers.length) return;
    [APP.layers[idx], APP.layers[newIdx]] = [APP.layers[newIdx], APP.layers[idx]];
    renderLayers();
}

function renderLayers() {
    DOM.layerList.innerHTML = '';
    APP.layers.forEach((layer, idx) => {
        const card = document.createElement('div');
        card.className = 'layer-card';

        const paramDefs = LAYER_PARAMS[layer.type] || [];
        let paramsHTML = '';
        paramDefs.forEach((pd) => {
            const step = pd.step || 1;
            paramsHTML += `
                <div class="param-label">
                    ${pd.label}
                    <input type="number" value="${layer.params[pd.key]}" step="${step}"
                        data-layer-id="${layer.id}" data-param-key="${pd.key}"
                        onchange="updateLayerParam(${layer.id}, '${pd.key}', this.value)">
                </div>
            `;
        });

        card.innerHTML = `
            <span class="layer-index">${idx + 1}</span>
            <span class="layer-type">${layer.type}</span>
            <div class="layer-params">${paramsHTML}</div>
            <div class="layer-actions">
                <button class="move-btn" title="Move up" onclick="moveLayer(${layer.id}, -1)">
                    <span class="material-symbols-outlined">keyboard_arrow_up</span>
                </button>
                <button class="move-btn" title="Move down" onclick="moveLayer(${layer.id}, 1)">
                    <span class="material-symbols-outlined">keyboard_arrow_down</span>
                </button>
                <button title="Remove" onclick="removeLayer(${layer.id})">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
        `;

        DOM.layerList.appendChild(card);
    });
}

// Make functions global for inline handlers
window.removeLayer = removeLayer;
window.moveLayer = moveLayer;
window.updateLayerParam = (id, key, value) => {
    const layer = APP.layers.find((l) => l.id === id);
    if (layer) {
        layer.params[key] = parseFloat(value);
    }
};

// ─── Network Build ──────────────────────────────────────────────────────────

async function buildNetwork() {
    const layers = APP.layers.map((l) => ({
        type: l.type,
        params: l.params,
    }));

    try {
        const res = await fetch('/network', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ layers }),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            APP.hasModel = true;
            DOM.modelInfo.classList.remove('hidden');
            DOM.paramCount.textContent = data.total_params.toLocaleString();
            toast(`Model built: ${data.trainable_params.toLocaleString()} params`, 'success');
        } else {
            toast(`Error: ${data.message}`, 'error');
        }
    } catch (e) {
        toast(`Build failed: ${e.message}`, 'error');
    }
}

// ─── Image Upload ───────────────────────────────────────────────────────────

function handleFileSelect(e) {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
}

function handleFileDrop(e) {
    e.preventDefault();
    DOM.dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'ok') {
            APP.hasImage = true;
            APP.imgWidth = data.width;
            APP.imgHeight = data.height;
            DOM.dropZone.classList.add('has-file');
            DOM.dropZone.innerHTML = `
                <span class="material-symbols-outlined">check_circle</span>
                <span>${data.width}×${data.height} (${data.pixels.toLocaleString()} px)</span>
            `;

            // Draw original on canvas
            drawBase64OnCanvas(DOM.canvasOriginal, data.preview, 'L');

            // Clear output canvas
            const ctx = DOM.canvasOutput.getContext('2d');
            ctx.clearRect(0, 0, DOM.canvasOutput.width, DOM.canvasOutput.height);

            toast('Image uploaded!', 'success');
        }
    } catch (e) {
        toast(`Upload failed: ${e.message}`, 'error');
    }
}

// ─── Training Controls ─────────────────────────────────────────────────────

async function startTraining() {
    const config = getTrainConfig();
    APP.lossHistory = [];

    try {
        const res = await fetch('/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            APP.training = true;
            updateTrainingUI();
            DOM.trainStats.classList.remove('hidden');
            toast('Training started', 'info');
        } else {
            toast(data.message, 'error');
        }
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
    }
}

async function stopTraining() {
    try {
        const res = await fetch('/train/stop', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            toast('Stopping...', 'info');
        } else {
            toast(data.message, 'error');
        }
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
    }
}

async function resumeTraining() {
    const config = getTrainConfig();

    try {
        const res = await fetch('/train/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            APP.training = true;
            updateTrainingUI();
            toast('Resumed training', 'info');
        } else {
            toast(data.message, 'error');
        }
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
    }
}

function getTrainConfig() {
    return {
        lr: parseFloat(DOM.inputLR.value),
        epochs: parseInt(DOM.inputEpochs.value),
        batch_size: parseInt(DOM.inputBatch.value),
        viz_interval: parseInt(DOM.inputViz.value),
    };
}

function updateTrainingUI() {
    DOM.btnStart.disabled = APP.training;
    DOM.btnStop.disabled = !APP.training;
    DOM.btnResume.disabled = APP.training;

    DOM.statusBadge.textContent = APP.training ? 'Training' : 'Idle';
    DOM.statusBadge.className = APP.training ? 'badge badge-training' : 'badge badge-idle';
}

// ─── Canvas Drawing ─────────────────────────────────────────────────────────

function drawBase64OnCanvas(canvas, b64, mode) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${b64}`;
}

// ─── Loss Chart ─────────────────────────────────────────────────────────────

function drawLossChart() {
    // Simple inline sparkline in the stats area
    // We'll just update text for now, could add canvas chart later
}

// ─── Color Gradient Editor ──────────────────────────────────────────────────

function renderStops() {
    DOM.stopList.innerHTML = '';
    APP.colorStops
        .sort((a, b) => a.value - b.value)
        .forEach((stop, idx) => {
            const row = document.createElement('div');
            row.className = 'stop-row';
            row.innerHTML = `
                <input type="color" value="${stop.color}"
                    onchange="updateStop(${idx}, 'color', this.value)">
                <input type="number" value="${stop.value}" min="0" max="1" step="0.01"
                    onchange="updateStop(${idx}, 'value', parseFloat(this.value))">
                <button onclick="removeStop(${idx})" title="Remove stop">
                    <span class="material-symbols-outlined">close</span>
                </button>
            `;
            DOM.stopList.appendChild(row);
        });
}

function addColorStop(value, color) {
    APP.colorStops.push({ value, color });
    APP.colorStops.sort((a, b) => a.value - b.value);
    renderStops();
    drawGradientPreview();
}

window.updateStop = (idx, key, value) => {
    APP.colorStops[idx][key] = value;
    APP.colorStops.sort((a, b) => a.value - b.value);
    drawGradientPreview();
};

window.removeStop = (idx) => {
    if (APP.colorStops.length <= 2) {
        toast('Need at least 2 color stops', 'error');
        return;
    }
    APP.colorStops.splice(idx, 1);
    renderStops();
    drawGradientPreview();
};

function drawGradientPreview() {
    const canvas = DOM.gradientPreview;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    const sorted = [...APP.colorStops].sort((a, b) => a.value - b.value);
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    sorted.forEach((s) => grad.addColorStop(s.value, s.color));

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
}

async function applyColorMap() {
    try {
        const res = await fetch('/colorize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stops: APP.colorStops }),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            drawBase64OnCanvas(DOM.canvasColorized, data.image, 'RGB');
            toast('Color map applied!', 'success');
        } else {
            toast(data.message, 'error');
        }
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
    }
}

async function saveImage() {
    const stops = JSON.stringify(APP.colorStops);
    const url = `/export?colorized=true&stops=${encodeURIComponent(stops)}`;

    try {
        const res = await fetch(url);
        if (!res.ok) {
            toast('No output to save', 'error');
            return;
        }
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'nn_output.png';
        a.click();
        URL.revokeObjectURL(a.href);
        toast('Image saved!', 'success');
    } catch (e) {
        toast(`Save failed: ${e.message}`, 'error');
    }
}

// ─── Status ─────────────────────────────────────────────────────────────────

async function fetchStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        DOM.deviceBadge.textContent = data.device.toUpperCase();
    } catch (e) {
        DOM.deviceBadge.textContent = 'Offline';
    }
}

// ─── Toast Notifications ────────────────────────────────────────────────────

function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}
