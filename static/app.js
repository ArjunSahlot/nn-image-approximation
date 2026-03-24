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
    vizHistory: [],
    timelineIndex: 0,
    isPlayingTimeline: false,
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
    timelineContainer: $('#timeline-container'),
    timelineSlider: $('#timeline-slider'),
    timelineChart: $('#timeline-chart'),
    timelinePlayBtn: $('#timeline-play-btn'),
    timelineInfo: $('#timeline-info'),
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

    // Timeline controls
    DOM.timelineSlider.addEventListener('input', (e) => {
        APP.timelineIndex = parseInt(e.target.value);
        updateTimelineDisplay();
    });

    DOM.timelinePlayBtn.addEventListener('click', toggleTimelinePlayback);
    DOM.timelineChart.addEventListener('click', handleTimelineChartClick);
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

            if (data.image) {
                // Store visualization history
                APP.vizHistory.push({
                    epoch: data.epoch,
                    loss: data.loss,
                    image: data.image,
                });

                // Draw current output
                drawBase64OnCanvas(DOM.canvasOutput, data.image, 'L');

                // Update timeline
                updateTimelineSlider();
                drawTimelineChart();
            }

            drawLossChart();
        }

        if (data.type === 'train_done') {
            APP.training = false;
            updateTrainingUI();
            DOM.statEpoch.textContent = data.epoch;
            DOM.statLoss.textContent = data.loss.toExponential(4);

            if (data.image) {
                // Store final visualization
                APP.vizHistory.push({
                    epoch: data.epoch,
                    loss: data.loss,
                    image: data.image,
                });

                drawBase64OnCanvas(DOM.canvasOutput, data.image, 'L');
                updateTimelineSlider();
                drawTimelineChart();
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
        card.draggable = true;
        card.dataset.layerId = layer.id;

        const paramDefs = LAYER_PARAMS[layer.type] || [];
        let paramsHTML = '';
        paramDefs.forEach((pd) => {
            const step = pd.step || 1;
            paramsHTML += `
                <div class="param-label">
                    ${pd.label}
                    <input type="number" value="${layer.params[pd.key]}" step="${step}"
                        data-layer-id="${layer.id}" data-param-key="${pd.key}" class="param-input">
                </div>
            `;
        });

        card.innerHTML = `
            <div class="layer-drag-handle">
                <span class="material-symbols-outlined">drag_handle</span>
            </div>
            <span class="layer-index">${idx + 1}</span>
            <span class="layer-type">${layer.type}</span>
            <div class="layer-params">${paramsHTML}</div>
            <div class="layer-actions">
                <button class="move-btn up-btn" data-layer-id="${layer.id}" title="Move up">
                    <span class="material-symbols-outlined">keyboard_arrow_up</span>
                </button>
                <button class="move-btn down-btn" data-layer-id="${layer.id}" title="Move down">
                    <span class="material-symbols-outlined">keyboard_arrow_down</span>
                </button>
                <button class="delete-btn" data-layer-id="${layer.id}" title="Remove">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
        `;

        // Attach event listeners
        const upBtn = card.querySelector('.up-btn');
        const downBtn = card.querySelector('.down-btn');
        const deleteBtn = card.querySelector('.delete-btn');
        const paramInputs = card.querySelectorAll('.param-input');

        upBtn.addEventListener('click', () => moveLayer(layer.id, -1));
        downBtn.addEventListener('click', () => moveLayer(layer.id, 1));
        deleteBtn.addEventListener('click', () => removeLayer(layer.id));

        paramInputs.forEach((input) => {
            input.addEventListener('change', (e) => {
                const layerId = parseInt(e.target.dataset.layerId);
                const paramKey = e.target.dataset.paramKey;
                const value = parseFloat(e.target.value);
                updateLayerParam(layerId, paramKey, value);
            });
        });

        // Drag events
        card.addEventListener('dragstart', handleLayerDragStart);
        card.addEventListener('dragover', handleLayerDragOver);
        card.addEventListener('drop', handleLayerDrop);
        card.addEventListener('dragend', handleLayerDragEnd);
        card.addEventListener('dragleave', handleLayerDragLeave);

        DOM.layerList.appendChild(card);
    });
}

let draggedElement = null;

function handleLayerDragStart(e) {
    draggedElement = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
}

function handleLayerDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
    e.dataTransfer.dropEffect = 'move';

    if (this !== draggedElement && this.classList.contains('layer-card')) {
        this.classList.add('drag-over');
    }
    return false;
}

function handleLayerDrop(e) {
    if (e.stopPropagation) {
        e.stopPropagation();
    }

    if (this !== draggedElement && draggedElement) {
        // Swap the layers
        const draggedId = parseInt(draggedElement.dataset.layerId);
        const targetId = parseInt(this.dataset.layerId);

        const draggedIdx = APP.layers.findIndex((l) => l.id === draggedId);
        const targetIdx = APP.layers.findIndex((l) => l.id === targetId);

        if (draggedIdx !== -1 && targetIdx !== -1) {
            [APP.layers[draggedIdx], APP.layers[targetIdx]] = [APP.layers[targetIdx], APP.layers[draggedIdx]];
            renderLayers();
        }
    }

    return false;
}

function handleLayerDragEnd(e) {
    this.classList.remove('dragging');
    $$('.layer-card').forEach((card) => {
        card.classList.remove('drag-over');
    });
}

function handleLayerDragLeave(e) {
    if (e.target === this) {
        this.classList.remove('drag-over');
    }
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
    APP.vizHistory = [];
    APP.timelineIndex = 0;
    APP.isPlayingTimeline = false;
    DOM.timelineContainer.classList.add('hidden');
    DOM.timelinePlayBtn.classList.remove('playing');
    DOM.timelinePlayBtn.innerHTML = '<span class="material-symbols-outlined">play_arrow</span>';

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

// ─── Timeline Management ────────────────────────────────────────────────────

function updateTimelineSlider() {
    const maxFrames = Math.max(0, APP.vizHistory.length - 1);
    DOM.timelineSlider.max = maxFrames;
    DOM.timelineSlider.value = maxFrames;
    APP.timelineIndex = maxFrames;

    if (APP.vizHistory.length > 0) {
        DOM.timelineContainer.classList.remove('hidden');
    }
}

function drawTimelineChart() {
    const canvas = DOM.timelineChart;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.fillStyle = 'rgba(30, 41, 59, 0.6)';
    ctx.fillRect(0, 0, w, h);

    if (APP.lossHistory.length === 0) return;

    // Draw loss line chart
    const losses = APP.lossHistory;
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();

    losses.forEach((loss, idx) => {
        const x = (idx / (losses.length - 1 || 1)) * w;
        const y = h - ((loss - minLoss) / range) * (h - 4) - 2;

        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw playhead
    if (APP.vizHistory.length > 0) {
        const playheadX = (APP.timelineIndex / Math.max(1, APP.vizHistory.length - 1)) * w;
        ctx.strokeStyle = 'rgba(99, 102, 241, 1)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, h);
        ctx.stroke();
    }
}

function updateTimelineDisplay() {
    if (APP.vizHistory.length === 0) return;

    const frame = APP.vizHistory[APP.timelineIndex];
    if (!frame) return;

    // Update the canvas output to show the selected frame
    drawBase64OnCanvas(DOM.canvasOutput, frame.image, 'L');

    // Update info text
    DOM.timelineInfo.textContent = `Epoch ${frame.epoch} / ${APP.vizHistory[APP.vizHistory.length - 1]?.epoch || 0}`;

    // Redraw timeline chart with playhead
    drawTimelineChart();
}

function handleTimelineChartClick(e) {
    const canvas = DOM.timelineChart;
    const rect = canvas.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const x = e.clientX - canvasRect.left;
    const percent = x / canvasRect.width;
    const index = Math.round(percent * (APP.vizHistory.length - 1));
    const clampedIndex = Math.max(0, Math.min(index, APP.vizHistory.length - 1));

    APP.timelineIndex = clampedIndex;
    DOM.timelineSlider.value = clampedIndex;
    updateTimelineDisplay();
}

function toggleTimelinePlayback() {
    APP.isPlayingTimeline = !APP.isPlayingTimeline;

    if (APP.isPlayingTimeline) {
        DOM.timelinePlayBtn.classList.add('playing');
        DOM.timelinePlayBtn.innerHTML = '<span class="material-symbols-outlined">pause</span>';
        playTimelineSequence();
    } else {
        DOM.timelinePlayBtn.classList.remove('playing');
        DOM.timelinePlayBtn.innerHTML = '<span class="material-symbols-outlined">play_arrow</span>';
    }
}

async function playTimelineSequence() {
    while (APP.isPlayingTimeline && APP.timelineIndex < APP.vizHistory.length - 1) {
        APP.timelineIndex++;
        DOM.timelineSlider.value = APP.timelineIndex;
        updateTimelineDisplay();
        await new Promise((resolve) => setTimeout(resolve, 200)); // 200ms between frames
    }

    if (APP.isPlayingTimeline) {
        // Loop back to start
        APP.timelineIndex = 0;
        DOM.timelineSlider.value = 0;
        updateTimelineDisplay();
        await new Promise((resolve) => setTimeout(resolve, 500));
        playTimelineSequence();
    }
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
