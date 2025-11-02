import os
import math
import heapq
import time
import base64
from io import BytesIO
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, render_template, request, jsonify, abort, redirect, url_for
import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


# -----------------------------
# Timing helpers
# -----------------------------
def ms(seconds: float) -> float:
    return seconds * 1000.0

@contextmanager
def measure(timings: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


# -----------------------------
# Core Ops (fast impls; keep names)
# -----------------------------
def fft_denoise(image: np.ndarray, keep_fraction: float = 0.1) -> np.ndarray:
    kf = float(np.clip(keep_fraction, 0.01, 0.99))
    sigma = 0.6 + (1.0 - kf) * 3.4
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

def apply_clahe_rgb_lab(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def LoG(x: float, y: float, sigma: float) -> float:
    r2 = x**2 + y**2
    return -1.0 / (math.pi * sigma**4) * (1.0 - r2 / (2.0 * sigma**2)) * math.exp(-r2 / (2.0 * sigma**2))

def LoG_Kernel_Generator(size: int, sigma: float) -> np.ndarray:
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            kernel[dy + k, dx + k] = LoG(dx, dy, sigma)
    return kernel

def local_variance_cv(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    image = image.astype(np.float32)
    mean = cv2.blur(image, (ksize, ksize))
    mean_sq = cv2.blur(image**2, (ksize, ksize))
    variance = mean_sq - mean**2
    pad = ksize // 2
    variance[:pad, :] = variance[-pad:, :] = variance[:, :pad] = variance[:, -pad:] = 0
    return variance

def robust_laplacian_edge_detector(image_gray: np.ndarray, log_image: np.ndarray, threshold_value: float) -> np.ndarray:
    variance_image = local_variance_cv(image_gray, 3)
    zc_vert = (log_image[2:, 1:-1] * log_image[:-2, 1:-1]) < 0
    zc_horz = (log_image[1:-1, 2:] * log_image[1:-1, :-2]) < 0
    zc = np.logical_or(zc_vert, zc_horz)
    var_gate = variance_image[1:-1, 1:-1] > threshold_value
    edges = np.zeros_like(image_gray, dtype=np.uint8)
    edges[1:-1, 1:-1] = (zc & var_gate).astype(np.uint8) * 255
    return edges


# -----------------------------
# A* Pathfinding (unchanged)
# -----------------------------
def get_neighbors(pos, rows, cols):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar_pathfinding(cost_map: np.ndarray, start, goal):
    rows, cols = cost_map.shape
    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in get_neighbors(current, rows, cols):
            tentative_g = g_score[current] + float(cost_map[neighbor])
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def slope_to_clock_if(m_img: float) -> str:
    m_math = -m_img
    vx, vy = (1.0, m_math) if m_math >= 0 else (-1.0, -m_math)
    angle = np.degrees(np.arctan2(vy, vx))
    angle = float(np.clip(angle, 0, 180))
    if angle < 7.5:   return "3"
    if angle < 22.5:  return "2:30"
    if angle < 37.5:  return "2"
    if angle < 52.5:  return "1:30"
    if angle < 67.5:  return "1"
    if angle < 82.5:  return "12:30"
    if angle < 97.5:  return "12"
    if angle < 112.5: return "11:30"
    if angle < 127.5: return "11"
    if angle < 142.5: return "10:30"
    if angle < 157.5: return "10"
    if angle < 172.5: return "9:30"
    return "9"


# -----------------------------
# MiDaS (TFLite) — load once
# -----------------------------
class MidasTFLite:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.inp = self.input_details[0]
        self.out = self.output_details[0]
        _, self.in_h, self.in_w, self.in_c = self.inp["shape"]
        self.in_dtype = self.inp["dtype"]

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        if self.in_dtype == np.uint8:
            return resized[np.newaxis, ...].astype(np.uint8)
        return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]

    def postprocess(self, out_arr: np.ndarray, out_size_hw) -> np.ndarray:
        x = out_arr
        if x.ndim == 4:
            if x.shape[-1] == 1:
                x = x[0, :, :, 0]
            elif x.shape[1] == 1:
                x = x[0, 0, :, :]
            else:
                x = x[0]
        elif x.ndim == 3:
            x = x[0]
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-9:
            x = np.zeros_like(x, dtype=np.uint8)
        else:
            x = (255.0 * (x - mn) / (mx - mn)).astype(np.uint8)
        H, W = out_size_hw
        return cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        inp = self.preprocess(rgb)
        self.interpreter.set_tensor(self.inp["index"], inp)
        self.interpreter.invoke()
        out_arr = self.interpreter.get_tensor(self.out["index"])
        return self.postprocess(out_arr, (rgb.shape[0], rgb.shape[1]))


# -----------------------------
# Helpers for web display
# -----------------------------
def to_data_url(img: np.ndarray, is_rgb: bool = True, max_w: int = 640) -> str:
    """
    Encode an image (RGB or grayscale) as base64 PNG data URL.
    Downscale to max_w for lighter pages.
    """
    if img.ndim == 3 and is_rgb:
        h, w = img.shape[:2]
        if w > max_w:
            new_h = int(h * max_w / w)
            img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
    else:
        h, w = img.shape[:2]
        if w > max_w:
            new_h = int(h * max_w / w)
            img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def draw_arrow_overlay(img_rgb: np.ndarray, m_img: float) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    c_img = (h - 1) - m_img * (w // 2)
    arrow_len = int(max(10, h / 3))
    y_end = max(h - 1 - arrow_len, 0)
    if abs(m_img) < 1e-6:
        x_end = w // 2
    else:
        x_end = int((y_end - c_img) / m_img)
    x_end = int(np.clip(x_end, 0, w - 1))
    out = img_rgb.copy()
    cv2.arrowedLine(out, (w // 2, h - 1), (x_end, y_end), (255, 255, 0), 3, tipLength=0.2)
    return out


def make_patch_visual(patch01: np.ndarray, path_cells, cell_px: int = 28) -> np.ndarray:
    """
    Visualize 15x15 patch heatmap (0..1) and overlay A* path in red.
    """
    R, C = patch01.shape
    vis = (patch01 * 255.0).astype(np.uint8)
    vis = cv2.resize(vis, (C * cell_px, R * cell_px), interpolation=cv2.INTER_NEAREST)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)

    def cell_center(rc):
        r, c = rc
        return (int(c * cell_px + cell_px / 2), int(r * cell_px + cell_px / 2))

    for i in range(1, len(path_cells)):
        p1 = cell_center(path_cells[i - 1])
        p2 = cell_center(path_cells[i])
        cv2.line(vis_rgb, p1, p2, (255, 0, 0), 2)
    return vis_rgb


# -----------------------------
# Pipeline (returns images + timings)
# -----------------------------
def process_image_array(img_rgb: np.ndarray, depth_net: MidasTFLite, clahe_clip: float, clahe_tiles: tuple[int, int]):
    timings = {}
    images = {}

    with measure(timings, "prep_gray"):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    images["Original"] = img_rgb
    images["Grayscale"] = img_gray

    with measure(timings, "clahe"):
        img_eq = apply_clahe_rgb_lab(img_rgb, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    images["CLAHE (LAB L)"] = img_eq

    with measure(timings, "fft_denoise_1"):
        s1_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY))
    images["FFT Denoise #1"] = s1_gray

    with measure(timings, "log_kernel"):
        log_kernel = LoG_Kernel_Generator(9, 1.0)

    with measure(timings, "log_convolution"):
        log_conv = cv2.filter2D(s1_gray.astype(np.float32), -1, log_kernel)
    images["LoG Convolution"] = cv2.convertScaleAbs(log_conv)

    with measure(timings, "edge_detection"):
        edges = robust_laplacian_edge_detector(img_gray, log_conv, 150.0)
    images["Edges (zero-cross)"] = edges

    with measure(timings, "fft_denoise_2"):
        s2_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY))
        s2_rgb = cv2.cvtColor(s2_gray, cv2.COLOR_GRAY2RGB)
    images["FFT Denoise #2"] = s2_gray

    with measure(timings, "depth_inference_tflite"):
        depth = depth_net(s2_rgb)
    images["Depth Map"] = depth

    with measure(timings, "combine_depth_edge"):
        combined = np.where(edges > 0, 255, depth).astype(np.uint8)
    images["Depth + Edge"] = combined

    with measure(timings, "patchify"):
        R, C = 15, 15
        h, w = combined.shape
        patch = np.zeros((R, C), dtype=np.float32)
        ph, pw = h // R, w // C
        for r in range(R):
            for c in range(C):
                y1, y2 = r * ph, min((r + 1) * ph, h)
                x1, x2 = c * pw, min((c + 1) * pw, w)
                patch[r, c] = np.mean(combined[y1:y2, x1:x2])
        patch01 = patch / 255.0  # 0..1

    with measure(timings, "astar_pathfinding"):
        start = (R - 1, C // 2)
        goal = np.unravel_index(np.argmin(patch01[:5, :]), patch01[:5, :].shape)
        path = astar_pathfinding(patch01, start, goal) or [start, goal]

    with measure(timings, "slope_direction"):
        x_vals = np.array([c - start[1] for r, c in path])
        y_vals = np.array([r - start[0] for r, c in path])
        m = np.sum(x_vals * y_vals) / (np.sum(x_vals**2) + 1e-6)
        m_img = (h / R) / (w / C) * m
        direction = slope_to_clock_if(m_img)

    # visuals for path + arrow
    images["15x15 Patch + Path"] = make_patch_visual(patch01, path, cell_px=28)
    images["Arrow Overlay"] = draw_arrow_overlay(img_rgb, m_img)

    total = sum(timings.values())
    return {
        "direction": direction,
        "slope_m_img": float(m_img),
        "timings_s": timings,
        "total_s": total,
        "images": images,  # dict of {title: np.ndarray}
    }


def print_pipeline_report(image_label: str, result: dict):
    print(f"\n🖼️ Image: {image_label}")
    for k, v in result["timings_s"].items():
        print(f"  {k:<22} {ms(v):8.2f} ms")
    print(f"  {'-'*35}")
    print(f"  TOTAL{'':17} {ms(result['total_s']):8.2f} ms | Direction: {result['direction']} | Slope: {result['slope_m_img']:.4f}")


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
MODEL_PATH = os.environ.get("MIDAS_TFLITE", "models/Midas-V2.tflite")
print("Loading model once...", end=" ", flush=True)
_t0 = time.perf_counter()
depth_net = MidasTFLite(MODEL_PATH)  # load once at startup
print(f"done ({ms(time.perf_counter()-_t0):.1f} ms).")


@app.get("/")
def index():
    return render_template("index.html", defaults={"clahe_clip": 2.0, "tile_w": 8, "tile_h": 8})


def _read_image_from_request(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.post("/process")
def process_form():
    if "image" not in request.files:
        abort(400, "No file part named 'image'")
    f = request.files["image"]
    if f.filename == "":
        abort(400, "No file selected")

    try:
        clahe_clip = float(request.form.get("clahe_clip", 2.0))
        tile_w = int(request.form.get("tile_w", 8))
        tile_h = int(request.form.get("tile_h", 8))
    except Exception:
        abort(400, "Invalid CLAHE parameters")

    img_rgb = _read_image_from_request(f)
    if img_rgb is None:
        abort(400, "Could not decode image")

    result = process_image_array(img_rgb, depth_net, clahe_clip, (tile_w, tile_h))

    # PRINT to terminal
    print_pipeline_report(image_label=f.filename or "upload", result=result)

    # Convert all intermediates to data URLs
    image_cards = []
    for title, arr in result["images"].items():
        is_rgb = (arr.ndim == 3)
        data_url = to_data_url(arr, is_rgb=is_rgb, max_w=700)
        image_cards.append({"title": title, "data_url": data_url})

    # Show a result page with the intermediate images + timings
    timings_rows = [(k, f"{ms(v):.2f}") for k, v in result["timings_s"].items()]
    return render_template(
        "result.html",
        direction=result["direction"],
        slope=f"{result['slope_m_img']:.4f}",
        total_ms=f"{ms(result['total_s']):.2f}",
        timings=timings_rows,
        params={"clahe_clip": clahe_clip, "tile_w": tile_w, "tile_h": tile_h},
        images=image_cards,
    )


# Optional JSON API (prints to terminal too)
@app.post("/api/process")
def process_api():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400
    f = request.files["image"]

    try:
        clahe_clip = float(request.form.get("clahe_clip", 2.0))
        tile_w = int(request.form.get("tile_w", 8))
        tile_h = int(request.form.get("tile_h", 8))
    except Exception:
        return jsonify({"error": "Invalid CLAHE parameters"}), 400

    img_rgb = _read_image_from_request(f)
    if img_rgb is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = process_image_array(img_rgb, depth_net, clahe_clip, (tile_w, tile_h))
    print_pipeline_report(image_label=f.filename or "upload(api)", result=result)

    # also return base64 intermediates if you want (comment out if not needed)
    images64 = {}
    for title, arr in result["images"].items():
        is_rgb = (arr.ndim == 3)
        images64[title] = to_data_url(arr, is_rgb=is_rgb, max_w=512)

    return jsonify(
        {
            "direction": result["direction"],
            "slope_m_img": result["slope_m_img"],
            "total_ms": ms(result["total_s"]),
            "steps_ms": {k: ms(v) for k, v in result["timings_s"].items()},
            "images": images64,
        }
    )


if __name__ == "__main__":
    # Run local dev server: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
