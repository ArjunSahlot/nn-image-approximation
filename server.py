"""
Neural Network Image Approximation Playground - Backend Server
FastAPI + PyTorch server with WebSocket-based live training visualization.
"""

import asyncio
import base64
import io
import json
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="NN Image Playground")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global State ────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """Detect the best available device, with actual CUDA compatibility testing."""
    import os
    force = os.environ.get("DEVICE", "").lower()
    if force in ("cpu", "cuda"):
        return torch.device(force)

    if torch.cuda.is_available():
        try:
            # Actually test that CUDA kernels work on this GPU
            t = torch.randn(4, device="cuda")
            _ = t + t
            del t
            torch.cuda.empty_cache()
            print("CUDA test passed — using GPU")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA available but not compatible: {e}")
            print("Falling back to CPU")
    return torch.device("cpu")


device = detect_device()
print(f"Using device: {device}")

# Thread-safe message queue for broadcasting to WebSocket clients
msg_queue: queue.Queue = queue.Queue()


class AppState:
    def __init__(self):
        self.image_gray: Optional[np.ndarray] = None
        self.img_h: int = 0
        self.img_w: int = 0
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.coords: Optional[torch.Tensor] = None
        self.pixels: Optional[torch.Tensor] = None
        self.layer_config: list = []
        self.training: bool = False
        self.stop_flag = threading.Event()
        self.current_epoch: int = 0
        self.total_epochs_done: int = 0
        self.last_loss: float = 0.0
        self.last_output: Optional[np.ndarray] = None
        self.frame_history: list[dict] = []
        self.ws_clients: list[WebSocket] = []
        self.train_thread: Optional[threading.Thread] = None
        self.lr: float = 0.001
        self.batch_size: int = 4096
        self.viz_interval: int = 10


state = AppState()

# ─── Pydantic Models ─────────────────────────────────────────────────────────


class LayerDef(BaseModel):
    type: str
    params: dict = {}


class NetworkConfig(BaseModel):
    layers: list[LayerDef]


class TrainConfig(BaseModel):
    epochs: int = 500
    lr: float = 0.001
    batch_size: int = 4096
    viz_interval: int = 10


class ColorStop(BaseModel):
    value: float
    color: str


class ColorizeConfig(BaseModel):
    stops: list[ColorStop]
    epoch: Optional[int] = None


# ─── Network Builder ─────────────────────────────────────────────────────────


class Reshape(nn.Module):
    """Helper module to reshape tensors in nn.Sequential."""
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


def build_network(layer_defs: list[LayerDef]) -> nn.Module:
    """Build a sequential network from layer definitions.
    Input is always 2 (x,y coords), output should be 1 (intensity).
    """
    layers = []
    in_features = 2
    mode = "linear"

    for ldef in layer_defs:
        t = ldef.type.lower()
        p = ldef.params

        if t == "linear":
            out_f = int(p.get("out_features", 64))
            if mode == "conv":
                layers.append(nn.Flatten())
            layers.append(nn.Linear(in_features, out_f))
            in_features = out_f
            mode = "linear"

        elif t == "conv1d":
            out_ch = int(p.get("out_channels", 32))
            ks = int(p.get("kernel_size", 3))
            pad = ks // 2
            if mode == "linear":
                layers.append(Reshape(-1, 1, in_features))
                layers.append(nn.Conv1d(1, out_ch, ks, padding=pad))
                in_features = out_ch * in_features
            else:
                in_ch = int(p.get("_in_channels", 1))
                layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=pad))
            mode = "conv"

        elif t == "relu":
            layers.append(nn.ReLU())
        elif t == "leakyrelu":
            layers.append(nn.LeakyReLU(float(p.get("negative_slope", 0.01))))
        elif t == "sigmoid":
            layers.append(nn.Sigmoid())
        elif t == "tanh":
            layers.append(nn.Tanh())
        elif t == "silu" or t == "swish":
            layers.append(nn.SiLU())
        elif t == "gelu":
            layers.append(nn.GELU())
        elif t == "batchnorm":
            if mode == "conv":
                ch = int(p.get("num_features", in_features))
                layers.append(nn.BatchNorm1d(ch))
            else:
                layers.append(nn.BatchNorm1d(in_features))
        elif t == "layernorm":
            layers.append(nn.LayerNorm(in_features))
        elif t == "dropout":
            layers.append(nn.Dropout(float(p.get("p", 0.1))))

    return nn.Sequential(*layers)


# ─── Training ────────────────────────────────────────────────────────────────

def generate_output_image(model: nn.Module, coords: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Generate the full output image from the model."""
    model.eval()
    with torch.no_grad():
        chunk = 8192
        outputs = []
        for i in range(0, coords.shape[0], chunk):
            out = model(coords[i:i+chunk])
            outputs.append(out)
        output = torch.cat(outputs, dim=0)
        img = output.cpu().numpy().reshape(h, w)
        img = np.clip(img, 0, 1)
    model.train()
    return img


def numpy_to_b64png(arr: np.ndarray) -> str:
    """Convert a float32 [0,1] HxW array to base64 PNG."""
    img_pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def enqueue_message(msg: str):
    """Thread-safe: put message on queue for WebSocket broadcaster."""
    msg_queue.put(msg)


def record_timeline_frame(epoch: int, loss: float, img: np.ndarray):
    """Store timeline snapshots so the UI can scrub and export earlier frames."""
    state.last_output = img

    replaced = False
    for i, frame in enumerate(state.frame_history):
        if frame["epoch"] == epoch:
            state.frame_history[i] = {
                "epoch": epoch,
                "loss": float(loss),
                "image": img,
            }
            replaced = True
            break

    if not replaced:
        state.frame_history.append({
            "epoch": epoch,
            "loss": float(loss),
            "image": img,
        })

    max_frames = 2500
    if len(state.frame_history) > max_frames:
        state.frame_history = state.frame_history[-max_frames:]


def get_output_for_epoch(epoch: Optional[int] = None) -> Optional[np.ndarray]:
    """Return output for a given epoch or the latest image when no epoch is provided."""
    if epoch is None:
        return state.last_output

    for frame in state.frame_history:
        if frame["epoch"] == epoch:
            return frame["image"]
    return None


def training_loop(epochs: int, lr: float, batch_size: int, viz_interval: int):
    """Main training loop, runs in a separate thread."""
    state.training = True
    state.stop_flag.clear()

    model = state.model.to(device)
    coords = state.coords.to(device)
    pixels = state.pixels.to(device)

    state.optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = coords.shape[0]
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        if state.stop_flag.is_set():
            break

        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_coords = coords[idx]
            batch_pixels = pixels[idx]

            pred = model(batch_coords)
            loss = loss_fn(pred, batch_pixels)

            state.optimizer.zero_grad()
            loss.backward()
            state.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        state.current_epoch = state.total_epochs_done + epoch + 1
        state.last_loss = avg_loss

        # Send viz update at interval
        if (epoch + 1) % viz_interval == 0 or epoch == epochs - 1:
            img = generate_output_image(model, coords, state.img_h, state.img_w)
            record_timeline_frame(state.current_epoch, avg_loss, img)
            img_b64 = numpy_to_b64png(img)

            enqueue_message(json.dumps({
                "type": "train_update",
                "epoch": state.current_epoch,
                "loss": round(avg_loss, 8),
                "image": img_b64,
            }))

    state.total_epochs_done = state.current_epoch
    state.training = False

    # Final update
    if state.last_output is not None:
        img_b64 = numpy_to_b64png(state.last_output)
    else:
        img_b64 = ""

    enqueue_message(json.dumps({
        "type": "train_done",
        "epoch": state.current_epoch,
        "loss": round(state.last_loss, 8),
        "image": img_b64,
    }))


# ─── WebSocket Broadcaster Task ─────────────────────────────────────────────

async def ws_broadcaster():
    """Background async task that pulls from the queue and sends to all WS clients."""
    while True:
        # Poll the queue every 50ms
        try:
            msg = msg_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        # Send to all connected clients
        disconnected = []
        for ws in list(state.ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in state.ws_clients:
                state.ws_clients.remove(ws)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(ws_broadcaster())


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.post("/upload")
async def upload_image(file: UploadFile):
    """Upload image, convert to grayscale, prepare training data."""
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        return {"status": "error", "message": f"Could not open image: {e}"}

    # Resize if too large
    max_side = 256
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    gray = np.array(img, dtype=np.float32) / 255.0
    state.image_gray = gray
    state.img_h, state.img_w = gray.shape

    ys = np.linspace(-1, 1, state.img_h, dtype=np.float32)
    xs = np.linspace(-1, 1, state.img_w, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    pixels = gray.ravel().reshape(-1, 1)

    state.coords = torch.from_numpy(coords)
    state.pixels = torch.from_numpy(pixels)
    state.total_epochs_done = 0
    state.current_epoch = 0
    state.last_output = None
    state.frame_history = []

    preview_b64 = numpy_to_b64png(gray)
    return {
        "status": "ok",
        "width": state.img_w,
        "height": state.img_h,
        "pixels": state.img_w * state.img_h,
        "preview": preview_b64,
    }


@app.post("/network")
async def build_model(config: NetworkConfig):
    """Build the neural network from layer config."""
    try:
        state.layer_config = config.layers
        model = build_network(config.layers)
        state.model = model.to(device)
        state.total_epochs_done = 0
        state.current_epoch = 0
        state.last_output = None
        state.frame_history = []
        state.optimizer = None

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "status": "ok",
            "total_params": total_params,
            "trainable_params": trainable,
            "architecture": str(model),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train/start")
async def train_start(config: TrainConfig):
    """Start training in a background thread."""
    if state.model is None:
        return {"status": "error", "message": "No model built. Build a network first."}
    if state.coords is None:
        return {"status": "error", "message": "No image uploaded. Upload an image first."}
    if state.training:
        return {"status": "error", "message": "Training is already running."}

    # Rebuild model from scratch for fresh start
    if state.layer_config:
        model = build_network(state.layer_config)
        state.model = model.to(device)
    state.total_epochs_done = 0
    state.current_epoch = 0
    state.last_output = None
    state.frame_history = []

    # Drain any old messages
    while not msg_queue.empty():
        try:
            msg_queue.get_nowait()
        except queue.Empty:
            break

    t = threading.Thread(
        target=training_loop,
        args=(config.epochs, config.lr, config.batch_size, config.viz_interval),
        daemon=True,
    )
    state.train_thread = t
    t.start()
    return {"status": "ok", "message": "Training started."}


@app.post("/train/resume")
async def train_resume(config: TrainConfig):
    """Resume training from current weights."""
    if state.model is None:
        return {"status": "error", "message": "No model built."}
    if state.coords is None:
        return {"status": "error", "message": "No image uploaded."}
    if state.training:
        return {"status": "error", "message": "Training is already running."}

    t = threading.Thread(
        target=training_loop,
        args=(config.epochs, config.lr, config.batch_size, config.viz_interval),
        daemon=True,
    )
    state.train_thread = t
    t.start()
    return {"status": "ok", "message": f"Resumed training for {config.epochs} more epochs."}


@app.post("/train/stop")
async def train_stop():
    """Stop training after current epoch."""
    if not state.training:
        return {"status": "error", "message": "Not currently training."}
    state.stop_flag.set()
    return {"status": "ok", "message": "Stop signal sent."}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for live training updates."""
    await ws.accept()
    state.ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


@app.post("/colorize")
async def colorize(config: ColorizeConfig):
    """Apply color gradient to the current output image."""
    img = get_output_for_epoch(config.epoch)
    if img is None:
        return {"status": "error", "message": "No output image available. Train first."}

    stops = sorted(config.stops, key=lambda s: s.value)
    if len(stops) < 2:
        return {"status": "error", "message": "Need at least 2 color stops."}

    stop_values = [s.value for s in stops]
    stop_colors = []
    for s in stops:
        hx = s.color.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        stop_colors.append([r, g, b])

    h, w = img.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        channel_values = [c[i] for c in stop_colors]
        colored[:, :, i] = np.interp(
            img.ravel(), stop_values, channel_values
        ).reshape(h, w).astype(np.uint8)

    pil_img = Image.fromarray(colored, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"status": "ok", "image": b64}


@app.get("/export")
async def export_image(colorized: bool = False, stops: str = "", epoch: Optional[int] = None):
    """Export the current output as a downloadable PNG."""
    selected = get_output_for_epoch(epoch)
    if selected is None:
        return Response(content="No output image", status_code=400)

    if colorized and stops:
        stop_list = json.loads(stops)
        stop_list.sort(key=lambda s: s["value"])
        stop_values = [s["value"] for s in stop_list]
        stop_colors = []
        for s in stop_list:
            hx = s["color"].lstrip("#")
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            stop_colors.append([r, g, b])

        img = selected
        h, w = img.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            channel_values = [c[i] for c in stop_colors]
            colored[:, :, i] = np.interp(
                img.ravel(), stop_values, channel_values
            ).reshape(h, w).astype(np.uint8)

        pil_img = Image.fromarray(colored, mode="RGB")
    else:
        pil_img = Image.fromarray(
            (selected * 255).astype(np.uint8), mode="L"
        )

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=nn_output.png"},
    )


@app.get("/status")
async def get_status():
    """Get current training status."""
    return {
        "training": state.training,
        "epoch": state.current_epoch,
        "total_epochs_done": state.total_epochs_done,
        "loss": state.last_loss,
        "has_model": state.model is not None,
        "has_image": state.image_gray is not None,
        "timeline_frames": len(state.frame_history),
        "device": str(device),
        "image_size": (
            f"{state.img_w}x{state.img_h}"
            if state.image_gray is not None
            else None
        ),
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
