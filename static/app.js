/**
 * NN Image Playground - Frontend Logic
 * Drag-and-drop network editor, timeline scrubbing, training controls, and color map tools.
 */

const APP = {
    layers: [],
    ws: null,
    training: false,
    hasModel: false,
    hasImage: false,
    imgWidth: 0,
    imgHeight: 0,
    colorStops: [],
    stopIdCounter: 0,
    selectedStopId: null,
    draggingStopId: null,
    suppressNextGradientClick: false,
    lossHistory: [],
    layerIdCounter: 0,
    dragLayerId: null,
    timelineFrames: [],
    timelineIndex: -1,
    followLatestFrame: true,
    theme: 'light',
};

const THEME_STORAGE_KEY = 'nn-playground-theme';

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

const LAYER_PARAMS = {
    linear: [{ key: 'out_features', label: 'Neurons', default: 64, type: 'number', step: 1 }],
    conv1d: [
        { key: 'out_channels', label: 'Channels', default: 32, type: 'number', step: 1 },
        { key: 'kernel_size', label: 'Kernel', default: 3, type: 'number', step: 1 },
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
    btnSaveFrame: $('#btn-save-frame'),
    btnJumpLatest: $('#btn-jump-latest'),
    btnAddStop: $('#btn-add-stop'),
    btnThemeToggle: $('#btn-theme-toggle'),
    themeToggleLabel: $('#theme-toggle-label'),
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
    gradientMarkers: $('#gradient-markers'),
    stopList: $('#stop-list'),
    statusBadge: $('#status-badge'),
    deviceBadge: $('#device-badge'),
    timelinePanel: $('#timeline-panel'),
    timelineScrubber: $('#timeline-scrubber'),
    timelineCaption: $('#timeline-caption'),
};

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    resetToDefaultStops();
    initEventListeners();
    initWebSocket();
    renderStops();
    drawGradientPreview();
    fetchStatus();

    // Default starter network.
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
    DOM.btnAddLayer.addEventListener('click', () => addLayer(DOM.layerSelect.value));
    DOM.btnBuild.addEventListener('click', buildNetwork);
    DOM.btnStart.addEventListener('click', startTraining);
    DOM.btnStop.addEventListener('click', stopTraining);
    DOM.btnResume.addEventListener('click', resumeTraining);
    DOM.btnApplyColor.addEventListener('click', applyColorMap);
    DOM.btnSave.addEventListener('click', saveImage);
    DOM.btnSaveFrame.addEventListener('click', saveSelectedFrame);
    DOM.btnJumpLatest.addEventListener('click', () => {
        if (!APP.timelineFrames.length) return;
        APP.followLatestFrame = true;
        setTimelineIndex(APP.timelineFrames.length - 1, true);
    });

    DOM.btnAddStop.addEventListener('click', () => addColorStop(0.5, '#888888'));
    DOM.btnThemeToggle.addEventListener('click', toggleTheme);

    DOM.fileInput.addEventListener('change', handleFileSelect);
    DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());
    DOM.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.dropZone.classList.add('drag-over');
    });
    DOM.dropZone.addEventListener('dragleave', () => DOM.dropZone.classList.remove('drag-over'));
    DOM.dropZone.addEventListener('drop', handleFileDrop);

    DOM.layerList.addEventListener('click', handleLayerActionClick);
    DOM.layerList.addEventListener('input', handleLayerParamInput);
    DOM.layerList.addEventListener('dragstart', handleLayerDragStart);
    DOM.layerList.addEventListener('dragover', handleLayerDragOver);
    DOM.layerList.addEventListener('drop', handleLayerDrop);
    DOM.layerList.addEventListener('dragend', handleLayerDragEnd);

    DOM.stopList.addEventListener('input', handleStopInput);
    DOM.stopList.addEventListener('click', handleStopActionClick);
    DOM.gradientPreview.addEventListener('click', handleGradientPreviewClick);
    DOM.gradientMarkers.addEventListener('click', handleGradientMarkerClick);
    DOM.gradientMarkers.addEventListener('pointerdown', handleGradientMarkerPointerDown);

    DOM.timelineScrubber.addEventListener('input', () => {
        APP.followLatestFrame = false;
        setTimelineIndex(Number.parseInt(DOM.timelineScrubber.value, 10), true);
    });

    $$('.preset-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const preset = PRESETS[btn.dataset.preset];
            if (!preset) return;
            setColorStopsFromPreset(preset);
            renderStops();
            drawGradientPreview();
        });
    });

    $$('.train-preset-btn').forEach((btn) => {
        btn.addEventListener('click', () => applyTrainPreset(btn.dataset.trainPreset));
    });
}

function createColorStop(value, color) {
    APP.stopIdCounter += 1;
    return {
        id: APP.stopIdCounter,
        value: Math.max(0, Math.min(1, value)),
        color,
    };
}

function resetToDefaultStops() {
    setColorStopsFromPreset(PRESETS.grayscale);
}

function setColorStopsFromPreset(presetStops) {
    APP.colorStops = presetStops.map((stop) => createColorStop(stop.value, stop.color));
    APP.selectedStopId = APP.colorStops.length ? APP.colorStops[0].id : null;
}

function getSerializableStops() {
    return [...APP.colorStops]
        .sort((a, b) => a.value - b.value)
        .map((stop) => ({ value: stop.value, color: stop.color }));
}

function initializeTheme() {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    applyTheme(stored === 'dark' || stored === 'light' ? stored : preferred);
}

function applyTheme(theme) {
    APP.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);

    const isDark = theme === 'dark';
    DOM.themeToggleLabel.textContent = isDark ? 'Light' : 'Dark';
    const icon = DOM.btnThemeToggle.querySelector('.material-symbols-outlined');
    if (icon) {
        icon.textContent = isDark ? 'light_mode' : 'dark_mode';
    }
}

function toggleTheme() {
    applyTheme(APP.theme === 'dark' ? 'light' : 'dark');
}

function applyTrainPreset(preset) {
    const presets = {
        fast: { lr: 0.003, epochs: 250, batchSize: 4096, vizInterval: 5 },
        balanced: { lr: 0.001, epochs: 600, batchSize: 4096, vizInterval: 10 },
        quality: { lr: 0.0006, epochs: 1200, batchSize: 2048, vizInterval: 20 },
    };

    const cfg = presets[preset];
    if (!cfg) return;

    DOM.inputLR.value = String(cfg.lr);
    DOM.inputEpochs.value = String(cfg.epochs);
    DOM.inputBatch.value = String(cfg.batchSize);
    DOM.inputViz.value = String(cfg.vizInterval);

    toast(`Applied ${preset} settings.`, 'info');
}

function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    APP.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    APP.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'train_update') {
            DOM.statEpoch.textContent = data.epoch;
            DOM.statLoss.textContent = Number(data.loss).toExponential(4);
            APP.lossHistory.push(data.loss);
            upsertTimelineFrame(data.epoch, data.loss, data.image);
            if (APP.followLatestFrame) {
                setTimelineIndex(APP.timelineFrames.length - 1, true);
            }
        }

        if (data.type === 'train_done') {
            APP.training = false;
            updateTrainingUI();
            DOM.statEpoch.textContent = data.epoch;
            DOM.statLoss.textContent = Number(data.loss).toExponential(4);
            if (data.image) {
                upsertTimelineFrame(data.epoch, data.loss, data.image);
                setTimelineIndex(APP.timelineFrames.length - 1, true);
            }
            APP.followLatestFrame = true;
            toast('Training complete!', 'success');
        }
    };

    APP.ws.onclose = () => {
        setTimeout(initWebSocket, 2000);
    };

    APP.ws.onerror = () => {
        // Errors are handled by onclose reconnect flow.
    };
}

function createLayer(type, params = {}) {
    const id = ++APP.layerIdCounter;
    const resolvedParams = { ...params };
    const defs = LAYER_PARAMS[type] || [];
    defs.forEach((def) => {
        if (!(def.key in resolvedParams)) {
            resolvedParams[def.key] = def.default;
        }
    });
    return { id, type, params: resolvedParams };
}

function addLayer(type, params = {}) {
    APP.layers.push(createLayer(type, params));
    renderLayers();
}

function duplicateLayer(id) {
    const original = APP.layers.find((layer) => layer.id === id);
    if (!original) return;
    const idx = APP.layers.findIndex((layer) => layer.id === id);
    const clone = createLayer(original.type, { ...original.params });
    APP.layers.splice(idx + 1, 0, clone);
    renderLayers();
}

function removeLayer(id) {
    APP.layers = APP.layers.filter((layer) => layer.id !== id);
    renderLayers();
}

function moveLayer(id, direction) {
    const idx = APP.layers.findIndex((layer) => layer.id === id);
    const target = idx + direction;
    if (idx < 0 || target < 0 || target >= APP.layers.length) return;
    [APP.layers[idx], APP.layers[target]] = [APP.layers[target], APP.layers[idx]];
    renderLayers();
}

function renderLayers() {
    DOM.layerList.innerHTML = '';

    if (!APP.layers.length) {
        DOM.layerList.innerHTML = '<div class="layer-empty">No layers yet. Add one from the selector above.</div>';
        return;
    }

    APP.layers.forEach((layer, idx) => {
        const card = document.createElement('div');
        card.className = 'layer-card';
        card.dataset.layerId = String(layer.id);
        card.draggable = true;

        const paramDefs = LAYER_PARAMS[layer.type] || [];
        const paramsHTML = paramDefs.map((def) => {
            const value = layer.params[def.key];
            const step = def.step || 1;
            return `
                <label class="param-field">
                    <span>${def.label}</span>
                    <input
                        type="number"
                        value="${value}"
                        step="${step}"
                        data-layer-id="${layer.id}"
                        data-param-key="${def.key}"
                    >
                </label>
            `;
        }).join('');

        card.innerHTML = `
            <div class="layer-grip" title="Drag to reorder">
                <span class="material-symbols-outlined">drag_indicator</span>
            </div>
            <div class="layer-main">
                <div class="layer-topline">
                    <span class="layer-index">#${idx + 1}</span>
                    <span class="layer-type">${layer.type}</span>
                </div>
                <div class="layer-params">${paramsHTML || '<span class="param-none">No params</span>'}</div>
            </div>
            <div class="layer-actions">
                <button data-action="up" data-layer-id="${layer.id}" title="Move up"><span class="material-symbols-outlined">keyboard_arrow_up</span></button>
                <button data-action="down" data-layer-id="${layer.id}" title="Move down"><span class="material-symbols-outlined">keyboard_arrow_down</span></button>
                <button data-action="duplicate" data-layer-id="${layer.id}" title="Duplicate layer"><span class="material-symbols-outlined">content_copy</span></button>
                <button data-action="remove" data-layer-id="${layer.id}" title="Remove layer"><span class="material-symbols-outlined">close</span></button>
            </div>
        `;

        DOM.layerList.appendChild(card);
    });
}

function handleLayerActionClick(event) {
    const button = event.target.closest('button[data-action]');
    if (!button) return;

    const id = Number.parseInt(button.dataset.layerId, 10);
    const action = button.dataset.action;

    if (action === 'up') moveLayer(id, -1);
    if (action === 'down') moveLayer(id, 1);
    if (action === 'duplicate') duplicateLayer(id);
    if (action === 'remove') removeLayer(id);
}

function handleLayerParamInput(event) {
    const input = event.target.closest('input[data-layer-id][data-param-key]');
    if (!input) return;

    const layerId = Number.parseInt(input.dataset.layerId, 10);
    const key = input.dataset.paramKey;
    const value = Number.parseFloat(input.value);
    if (Number.isNaN(value)) return;

    const layer = APP.layers.find((item) => item.id === layerId);
    if (!layer) return;
    layer.params[key] = value;
}

function handleLayerDragStart(event) {
    const card = event.target.closest('.layer-card');
    if (!card) return;

    const grip = event.target.closest('.layer-grip');
    if (!grip) {
        event.preventDefault();
        return;
    }

    APP.dragLayerId = Number.parseInt(card.dataset.layerId, 10);
    card.classList.add('dragging');
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', card.dataset.layerId);
}

function handleLayerDragOver(event) {
    event.preventDefault();
    const dragging = DOM.layerList.querySelector('.layer-card.dragging');
    if (!dragging) return;

    const afterElement = getDragAfterElement(DOM.layerList, event.clientY);
    if (!afterElement) {
        DOM.layerList.appendChild(dragging);
    } else {
        DOM.layerList.insertBefore(dragging, afterElement);
    }
}

function handleLayerDrop(event) {
    event.preventDefault();
    syncLayerOrderFromDom();
}

function handleLayerDragEnd(event) {
    const card = event.target.closest('.layer-card');
    if (card) {
        card.classList.remove('dragging');
    }
    APP.dragLayerId = null;
    syncLayerOrderFromDom();
    renderLayers();
}

function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.layer-card:not(.dragging)')];

    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;

        if (offset < 0 && offset > closest.offset) {
            return { offset, element: child };
        }

        return closest;
    }, { offset: Number.NEGATIVE_INFINITY, element: null }).element;
}

function syncLayerOrderFromDom() {
    const ids = [...DOM.layerList.querySelectorAll('.layer-card')].map((card) => Number.parseInt(card.dataset.layerId, 10));
    if (!ids.length) return;

    const byId = new Map(APP.layers.map((layer) => [layer.id, layer]));
    APP.layers = ids.map((id) => byId.get(id)).filter(Boolean);
}

async function buildNetwork() {
    const layers = APP.layers.map((layer) => ({
        type: layer.type,
        params: layer.params,
    }));

    if (!layers.length) {
        toast('Add at least one layer before compiling.', 'error');
        return;
    }

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
            DOM.paramCount.textContent = Number(data.total_params).toLocaleString();
            toast(`Model compiled: ${Number(data.trainable_params).toLocaleString()} params`, 'success');
        } else {
            toast(`Error: ${data.message}`, 'error');
        }
    } catch (error) {
        toast(`Build failed: ${error.message}`, 'error');
    }
}

function handleFileSelect(event) {
    if (event.target.files.length > 0) {
        uploadFile(event.target.files[0]);
    }
}

function handleFileDrop(event) {
    event.preventDefault();
    DOM.dropZone.classList.remove('drag-over');
    if (event.dataTransfer.files.length > 0) {
        uploadFile(event.dataTransfer.files[0]);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status !== 'ok') {
            toast(data.message || 'Upload failed.', 'error');
            return;
        }

        APP.hasImage = true;
        APP.imgWidth = data.width;
        APP.imgHeight = data.height;

        resetTimeline();

        DOM.dropZone.classList.add('has-file');
        DOM.dropZone.innerHTML = `
            <span class="material-symbols-outlined">check_circle</span>
            <span>${data.width}x${data.height} (${Number(data.pixels).toLocaleString()} px)</span>
        `;

        drawBase64OnCanvas(DOM.canvasOriginal, data.preview);
        clearCanvas(DOM.canvasOutput);
        clearCanvas(DOM.canvasColorized);

        toast('Image uploaded.', 'success');
    } catch (error) {
        toast(`Upload failed: ${error.message}`, 'error');
    }
}

function getTrainConfig() {
    return {
        lr: Number.parseFloat(DOM.inputLR.value),
        epochs: Number.parseInt(DOM.inputEpochs.value, 10),
        batch_size: Number.parseInt(DOM.inputBatch.value, 10),
        viz_interval: Number.parseInt(DOM.inputViz.value, 10),
    };
}

async function startTraining() {
    const config = getTrainConfig();
    APP.lossHistory = [];
    resetTimeline();

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
            toast('Training started.', 'info');
        } else {
            toast(data.message, 'error');
        }
    } catch (error) {
        toast(`Failed: ${error.message}`, 'error');
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
    } catch (error) {
        toast(`Failed: ${error.message}`, 'error');
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
            toast('Resumed training.', 'info');
        } else {
            toast(data.message, 'error');
        }
    } catch (error) {
        toast(`Failed: ${error.message}`, 'error');
    }
}

function updateTrainingUI() {
    DOM.btnStart.disabled = APP.training;
    DOM.btnStop.disabled = !APP.training;
    DOM.btnResume.disabled = APP.training;

    DOM.statusBadge.textContent = APP.training ? 'Training' : 'Idle';
    DOM.statusBadge.className = APP.training ? 'badge badge-training' : 'badge badge-idle';
}

function upsertTimelineFrame(epoch, loss, imageB64) {
    const existingIndex = APP.timelineFrames.findIndex((frame) => frame.epoch === epoch);
    const next = { epoch, loss, image: imageB64 };

    if (existingIndex >= 0) {
        APP.timelineFrames[existingIndex] = next;
    } else {
        APP.timelineFrames.push(next);
        APP.timelineFrames.sort((a, b) => a.epoch - b.epoch);
    }

    renderTimelineControls();
}

function resetTimeline() {
    APP.timelineFrames = [];
    APP.timelineIndex = -1;
    APP.followLatestFrame = true;
    renderTimelineControls();
}

function setTimelineIndex(index, redraw = false) {
    if (!APP.timelineFrames.length) {
        APP.timelineIndex = -1;
        renderTimelineControls();
        return;
    }

    const clamped = Math.max(0, Math.min(index, APP.timelineFrames.length - 1));
    APP.timelineIndex = clamped;
    if (redraw) {
        const frame = APP.timelineFrames[clamped];
        drawBase64OnCanvas(DOM.canvasOutput, frame.image);
    }
    renderTimelineControls();
}

function getSelectedFrame() {
    if (APP.timelineIndex < 0 || APP.timelineIndex >= APP.timelineFrames.length) {
        return null;
    }
    return APP.timelineFrames[APP.timelineIndex];
}

function renderTimelineControls() {
    const hasFrames = APP.timelineFrames.length > 0;

    DOM.timelinePanel.classList.toggle('hidden', !hasFrames);
    DOM.btnJumpLatest.disabled = !hasFrames;
    DOM.btnSaveFrame.disabled = !hasFrames;
    DOM.timelineScrubber.disabled = !hasFrames;

    if (!hasFrames) {
        DOM.timelineCaption.textContent = 'No frames yet';
        DOM.timelineScrubber.min = '0';
        DOM.timelineScrubber.max = '0';
        DOM.timelineScrubber.value = '0';
        return;
    }

    DOM.timelineScrubber.min = '0';
    DOM.timelineScrubber.max = String(APP.timelineFrames.length - 1);

    if (APP.timelineIndex < 0 || APP.timelineIndex >= APP.timelineFrames.length) {
        APP.timelineIndex = APP.timelineFrames.length - 1;
    }

    DOM.timelineScrubber.value = String(APP.timelineIndex);
    const frame = APP.timelineFrames[APP.timelineIndex];
    DOM.timelineCaption.textContent = `Epoch ${frame.epoch} - Loss ${Number(frame.loss).toExponential(4)} (${APP.timelineIndex + 1}/${APP.timelineFrames.length})`;
}

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawBase64OnCanvas(canvas, b64) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${b64}`;
}

function renderStops() {
    DOM.stopList.innerHTML = '';
    APP.colorStops
        .sort((a, b) => a.value - b.value)
        .forEach((stop) => {
            const row = document.createElement('div');
            row.className = stop.id === APP.selectedStopId ? 'stop-row selected' : 'stop-row';
            row.dataset.stopId = String(stop.id);
            row.innerHTML = `
                <input type="color" value="${stop.color}" data-stop-key="color" data-stop-id="${stop.id}">
                <input type="number" value="${stop.value}" min="0" max="1" step="0.01" data-stop-key="value" data-stop-id="${stop.id}">
                <button data-stop-action="remove" data-stop-id="${stop.id}" title="Remove stop">
                    <span class="material-symbols-outlined">close</span>
                </button>
            `;
            DOM.stopList.appendChild(row);
        });
}

function handleStopInput(event) {
    const input = event.target.closest('[data-stop-key][data-stop-id]');
    if (!input) return;

    const stopId = Number.parseInt(input.dataset.stopId, 10);
    const key = input.dataset.stopKey;
    const stop = APP.colorStops.find((item) => item.id === stopId);
    if (!stop) return;

    APP.selectedStopId = stopId;

    if (key === 'color') {
        stop.color = input.value;
    } else {
        const value = Number.parseFloat(input.value);
        if (Number.isNaN(value)) return;
        stop.value = Math.max(0, Math.min(1, value));
    }

    APP.colorStops.sort((a, b) => a.value - b.value);
    renderStops();
    drawGradientPreview();
}

function handleStopActionClick(event) {
    const button = event.target.closest('[data-stop-action="remove"][data-stop-id]');

    if (!button) {
        const row = event.target.closest('.stop-row[data-stop-id]');
        if (!row) return;
        APP.selectedStopId = Number.parseInt(row.dataset.stopId, 10);
        renderStops();
        drawGradientPreview();
        return;
    }

    if (APP.colorStops.length <= 2) {
        toast('Need at least 2 color stops.', 'error');
        return;
    }

    const stopId = Number.parseInt(button.dataset.stopId, 10);
    APP.colorStops = APP.colorStops.filter((stop) => stop.id !== stopId);
    if (!APP.colorStops.some((stop) => stop.id === APP.selectedStopId)) {
        APP.selectedStopId = APP.colorStops.length ? APP.colorStops[0].id : null;
    }
    renderStops();
    drawGradientPreview();
}

function addColorStop(value, color, selectNew = true) {
    const stop = createColorStop(value, color);
    APP.colorStops.push(stop);
    if (selectNew) {
        APP.selectedStopId = stop.id;
    }
    APP.colorStops.sort((a, b) => a.value - b.value);
    renderStops();
    drawGradientPreview();
}

function handleGradientPreviewClick(event) {
    if (APP.suppressNextGradientClick) {
        APP.suppressNextGradientClick = false;
        return;
    }

    const rect = DOM.gradientPreview.getBoundingClientRect();
    if (!rect.width) return;

    const scaleX = DOM.gradientPreview.width / rect.width;
    const xPx = (event.clientX - rect.left) * scaleX;
    const clampedXPx = Math.max(0, Math.min(DOM.gradientPreview.width - 1, xPx));
    const value = clampedXPx / Math.max(DOM.gradientPreview.width - 1, 1);

    const ctx = DOM.gradientPreview.getContext('2d');
    const sample = ctx.getImageData(Math.round(clampedXPx), Math.floor(DOM.gradientPreview.height / 2), 1, 1).data;
    const color = rgbToHex(sample[0], sample[1], sample[2]);

    addColorStop(value, color, true);
}

function handleGradientMarkerClick(event) {
    const marker = event.target.closest('.gradient-marker[data-stop-id]');
    if (!marker) return;
    APP.selectedStopId = Number.parseInt(marker.dataset.stopId, 10);
    renderStops();
    drawGradientPreview();
}

function handleGradientMarkerPointerDown(event) {
    const marker = event.target.closest('.gradient-marker[data-stop-id]');
    if (!marker) return;

    event.preventDefault();
    APP.draggingStopId = Number.parseInt(marker.dataset.stopId, 10);
    APP.selectedStopId = APP.draggingStopId;
    APP.suppressNextGradientClick = true;

    const updateFromPointer = (clientX) => {
        const stop = APP.colorStops.find((item) => item.id === APP.draggingStopId);
        if (!stop) return;

        const rect = DOM.gradientPreview.getBoundingClientRect();
        if (!rect.width) return;

        const ratio = (clientX - rect.left) / rect.width;
        stop.value = Math.max(0, Math.min(1, ratio));
        APP.colorStops.sort((a, b) => a.value - b.value);
        renderStops();
        drawGradientPreview();
    };

    const onMove = (moveEvent) => {
        if (APP.draggingStopId === null) return;
        updateFromPointer(moveEvent.clientX);
    };

    const onUp = () => {
        APP.draggingStopId = null;
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);

        // Let current click sequence finish before re-enabling click-to-add.
        window.setTimeout(() => {
            APP.suppressNextGradientClick = false;
        }, 0);
    };

    updateFromPointer(event.clientX);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
}

function renderGradientMarkers() {
    DOM.gradientMarkers.innerHTML = '';
    const sorted = [...APP.colorStops].sort((a, b) => a.value - b.value);

    sorted.forEach((stop) => {
        const marker = document.createElement('button');
        marker.className = stop.id === APP.selectedStopId ? 'gradient-marker selected' : 'gradient-marker';
        marker.dataset.stopId = String(stop.id);
        marker.title = `Stop ${stop.value.toFixed(2)}`;
        marker.style.left = `${stop.value * 100}%`;
        marker.style.background = stop.color;
        DOM.gradientMarkers.appendChild(marker);
    });
}

function rgbToHex(r, g, b) {
    return `#${[r, g, b].map((part) => part.toString(16).padStart(2, '0')).join('')}`;
}

function drawGradientPreview() {
    const canvas = DOM.gradientPreview;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    const sorted = [...APP.colorStops].sort((a, b) => a.value - b.value);
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    sorted.forEach((stop) => grad.addColorStop(stop.value, stop.color));

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    renderGradientMarkers();
}

async function applyColorMap() {
    const frame = getSelectedFrame();

    try {
        const res = await fetch('/colorize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stops: getSerializableStops(),
                epoch: frame ? frame.epoch : null,
            }),
        });
        const data = await res.json();

        if (data.status === 'ok') {
            drawBase64OnCanvas(DOM.canvasColorized, data.image);
            toast('Color map applied.', 'success');
        } else {
            toast(data.message, 'error');
        }
    } catch (error) {
        toast(`Failed: ${error.message}`, 'error');
    }
}

async function saveSelectedFrame() {
    await saveImage(true);
}

async function saveImage(forceSelected = false) {
    const frame = getSelectedFrame();
    const selectedEpoch = forceSelected ? (frame ? frame.epoch : null) : (frame ? frame.epoch : null);

    const params = new URLSearchParams();
    params.set('colorized', 'true');
    params.set('stops', JSON.stringify(getSerializableStops()));
    if (selectedEpoch !== null) {
        params.set('epoch', String(selectedEpoch));
    }

    try {
        const res = await fetch(`/export?${params.toString()}`);
        if (!res.ok) {
            toast('No output to save.', 'error');
            return;
        }
        const blob = await res.blob();
        const epochTag = selectedEpoch !== null ? `_epoch_${selectedEpoch}` : '';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `nn_output${epochTag}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
        toast('Image saved.', 'success');
    } catch (error) {
        toast(`Save failed: ${error.message}`, 'error');
    }
}

async function fetchStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        DOM.deviceBadge.textContent = String(data.device || 'Unknown').toUpperCase();
    } catch (_error) {
        DOM.deviceBadge.textContent = 'OFFLINE';
    }
}

function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}
