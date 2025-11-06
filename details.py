# ------------------------------------
# Imports
# ------------------------------------
import os
import math
import heapq
import time
import base64
from io import BytesIO
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, render_template, request, jsonify, abort
import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


# ------------------------------------
# Config
# ------------------------------------
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_TILE_W = 8
DEFAULT_TILE_H = 8
MODEL_PATH = os.environ.get("MIDAS_TFLITE", "models/Midas-V2.tflite")


# ------------------------------------
# Timing helpers
# ------------------------------------
def ms(seconds):
    return seconds * 1000.0

@contextmanager
def measure(timings, key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


# ------------------------------------
# Core Image Ops
# ------------------------------------
def apply_clahe_rgb_lab(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ------------------------------------
# A* Pathfinding
# ------------------------------------
def get_neighbors(pos, rows, cols):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar_pathfinding(cost_map, start, goal):
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


def slope_to_clock_if(m_img):
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


# ------------------------------------
# MiDaS (TFLite)
# ------------------------------------
class MidasTFLite:
    def __init__(self, model_path):
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

    def preprocess(self, rgb):
        resized = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        if self.in_dtype == np.uint8:
            return resized[np.newaxis, ...].astype(np.uint8)
        return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]

    def postprocess(self, out_arr, out_size_hw):
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

    def __call__(self, rgb):
        inp = self.preprocess(rgb)
        self.interpreter.set_tensor(self.inp["index"], inp)
        self.interpreter.invoke()
        out_arr = self.interpreter.get_tensor(self.out["index"])
        return self.postprocess(out_arr, (rgb.shape[0], rgb.shape[1]))


# ------------------------------------
# Utility helpers
# ------------------------------------
def to_data_url(img, is_rgb=True, max_w=640):
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


def draw_arrow_overlay(img_rgb, m_img):
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


def make_patch_visual(patch01, path_cells, cell_px=28):
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


# ---- NEW HELPERS: plain patch image + draw path on the original image ----
def make_patch_image_only(patch01, cell_px=28):
    """Returns a plain 15x15 patch visualization (no path overlay)."""
    R, C = patch01.shape
    vis = (patch01 * 255.0).astype(np.uint8)
    vis = cv2.resize(vis, (C * cell_px, R * cell_px), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)


def draw_path_on_original(img_rgb, path_cells, grid_rows, grid_cols, img_h, img_w):
    """
    Draws the path (as points + connecting lines) on the original image.
    path_cells: list of (r, c) in patch space
    """
    ph, pw = img_h // grid_rows, img_w // grid_cols

    def cell_center_px(rc):
        r, c = rc
        y = int(r * ph + ph / 2)
        x = int(c * pw + pw / 2)
        return (x, y)

    out = img_rgb.copy()
    if len(path_cells) > 0:
        # draw points
        for rc in path_cells:
            x, y = cell_center_px(rc)
            cv2.circle(out, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        # draw lines
        for i in range(1, len(path_cells)):
            x1, y1 = cell_center_px(path_cells[i - 1])
            x2, y2 = cell_center_px(path_cells[i])
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# ------------------------------------
# Pipeline
# ------------------------------------
def process_image_array(img_rgb, depth_net, clahe_clip, clahe_tiles):
    timings = {}
    images = {}

    images["Original"] = img_rgb

    # 1) CLAHE (on RGB via LAB-L channel)
    with measure(timings, "clahe"):
        img_eq = apply_clahe_rgb_lab(img_rgb, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)
    images["CLAHE (LAB L)"] = img_eq

    # 2) Gray (convert after CLAHE)
    with measure(timings, "to_gray"):
        img_gray = cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY)
    images["Grayscale"] = img_gray

    # 3) Gaussian blur (AFTER gray)  <-- per request
    with measure(timings, "gaussian_blur_gray"):
        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=1.0)
    images["Gaussian Blur (Gray)"] = img_gray_blur

    # 4) Edges from the blurred gray
    with measure(timings, "canny_edges"):
        canny_edges = cv2.Canny(img_gray_blur, 100, 200, L2gradient=True)
    images["Canny Edges"] = canny_edges

    # 5) Depth on color (keep 3-channel input for MiDaS)
    with measure(timings, "depth_inference_tflite"):
        depth = depth_net(img_eq)  # color (CLAHE) image for MiDaS
    images["Depth Map"] = depth

    # 6) Combine edges and depth
    with measure(timings, "combine_depth_edge"):
        combined = np.where(canny_edges > 0, 255, depth).astype(np.uint8)
    images["Depth + Edge"] = combined

    # 7) Patchify
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
        patch01 = patch / 255.0

    # 8) A* path
    with measure(timings, "astar_pathfinding"):
        start = (R - 1, C // 2)
        goal = np.unravel_index(np.argmin(patch01[:5, :]), patch01[:5, :].shape)
        path = astar_pathfinding(patch01, start, goal) or [start, goal]

    # 9) Direction estimate
    with measure(timings, "slope_direction"):
         # Only use path points in the lower half of the patch (rows 7–15)
        lower_path = [(r, c) for (r, c) in path if r >= 7]
        # Fallback: if no points in that range, use all points
        if len(lower_path) == 0:
            lower_path = path

        x_vals = np.array([c - start[1] for r, c in lower_path])
        y_vals = np.array([r - start[0] for r, c in lower_path])

        m = np.sum(x_vals * y_vals) / (np.sum(x_vals**2) + 1e-6)
        m_img = (h / R) / (w / C) * m
        direction = slope_to_clock_if(m_img)

    # 10) Visuals: separate plain patch + path on original + optional debug
    images["15x15 Patch"] = make_patch_image_only(patch01, cell_px=28)
    images["15x15 Patch + Path"] = make_patch_visual(patch01, path, cell_px=28)

    with measure(timings, "draw_path_on_original"):
        images["Path Points Overlay"] = draw_path_on_original(img_rgb, path, R, C, h, w)

    images["Arrow Overlay"] = draw_arrow_overlay(img_rgb, m_img)

    total = sum(timings.values())
    overlay = images["Arrow Overlay"].copy()
    cv2.putText(
        overlay,
        f"{total:.3f}s",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    images["Arrow Overlay"] = overlay

    return {
        "direction": direction,
        "slope_m_img": float(m_img),
        "timings_s": timings,
        "total_s": total,
        "images": images,
    }


def print_pipeline_report(image_label, result):
    print(f"\n Image: {image_label}")
    for k, v in result["timings_s"].items():
        print(f"  {k:<22} {v:8.3f} s")
    print(f"  {'-'*35}")
    print(
        f"  TOTAL{'':17} {result['total_s']:8.3f} s | "
        f"Direction: {result['direction']} | Slope: {result['slope_m_img']:.4f}"
    )


# ------------------------------------
# Flask App
# ------------------------------------
app = Flask(__name__)

print("Loading model once...", end=" ", flush=True)
_t0 = time.perf_counter()
depth_net = MidasTFLite(MODEL_PATH)
print(f"done ({ms(time.perf_counter()-_t0):.1f} ms).")


@app.get("/")
def index():
    return render_template(
        "index.html",
        defaults={"clahe_clip": DEFAULT_CLAHE_CLIP, "tile_w": DEFAULT_TILE_W, "tile_h": DEFAULT_TILE_H},
    )


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
        clahe_clip = float(request.form.get("clahe_clip", DEFAULT_CLAHE_CLIP))
        tile_w = int(request.form.get("tile_w", DEFAULT_TILE_W))
        tile_h = int(request.form.get("tile_h", DEFAULT_TILE_H))
    except Exception:
        abort(400, "Invalid CLAHE parameters")

    img_rgb = _read_image_from_request(f)
    if img_rgb is None:
        abort(400, "Could not decode image")

    result = process_image_array(img_rgb, depth_net, clahe_clip, (tile_w, tile_h))
    print_pipeline_report(image_label=f.filename or "upload", result=result)

    image_cards = []
    for title, arr in result["images"].items():
        is_rgb = (arr.ndim == 3)
        data_url = to_data_url(arr, is_rgb=is_rgb, max_w=700)
        image_cards.append({"title": title, "data_url": data_url})

    timings_rows = [(k, f"{v:.3f}") for k, v in result["timings_s"].items()]
    return render_template(
        "result.html",
        direction=result["direction"],
        slope=f"{result['slope_m_img']:.4f}",
        total_ms=f"{result['total_s']:.3f}",
        timings=timings_rows,
        params={"clahe_clip": clahe_clip, "tile_w": tile_w, "tile_h": tile_h},
        images=image_cards,
    )


@app.post("/api/process")
def process_api():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400
    f = request.files["image"]

    try:
        clahe_clip = float(request.form.get("clahe_clip", DEFAULT_CLAHE_CLIP))
        tile_w = int(request.form.get("tile_w", DEFAULT_TILE_W))
        tile_h = int(request.form.get("tile_h", DEFAULT_TILE_H))
    except Exception:
        return jsonify({"error": "Invalid CLAHE parameters"}), 400

    img_rgb = _read_image_from_request(f)
    if img_rgb is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = process_image_array(img_rgb, depth_net, clahe_clip, (tile_w, tile_h))
    print_pipeline_report(image_label=f.filename or "upload(api)", result=result)

    return jsonify({"direction": str(result["direction"])})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
