import argparse
import math
import heapq
import time
from contextlib import contextmanager
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
# Core Ops
# -----------------------------
def fft_denoise(image: np.ndarray, keep_fraction: float = 0.1) -> np.ndarray:
    """
    FAST replacement for the original FFT-based low-pass:
    Use OpenCV Gaussian blur (vectorized, C++).
    We map smaller keep_fraction -> stronger blur (larger sigma).
    """
    # clamp and map keep_fraction in [0.01, 0.99] to sigma in ~[0.6, 4.0]
    kf = float(np.clip(keep_fraction, 0.01, 0.99))
    sigma = 0.6 + (1.0 - kf) * 3.4  # keep_fraction=0.1 -> ~3.66 (strong smooth)
    # (0,0) lets OpenCV compute kernel size from sigma; preserves uint8 dtype
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
    """
    FAST vectorized zero-crossing + variance gating (no Python loops).
    Keeps your API and behavior, but runs orders of magnitude faster.
    """
    # local variance (already fast via OpenCV box filters)
    variance_image = local_variance_cv(image_gray, 3)

    # compute zero-crossings along vertical and horizontal directions
    # shapes: center window is [1:-1,1:-1]
    zc_vert = (log_image[2:, 1:-1] * log_image[:-2, 1:-1]) < 0
    zc_horz = (log_image[1:-1, 2:] * log_image[1:-1, :-2]) < 0
    zc = np.logical_or(zc_vert, zc_horz)

    # variance gate at center pixels
    var_gate = variance_image[1:-1, 1:-1] > threshold_value

    # compose edges
    edges = np.zeros_like(image_gray, dtype=np.uint8)
    edges[1:-1, 1:-1] = (zc & var_gate).astype(np.uint8) * 255
    return edges


# -----------------------------
# A* Pathfinding
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
# MiDaS (TFLite)
# -----------------------------
class MidasTFLite:
    def __init__(self, model_path: str):
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
        x = out_arr[0]
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-9:
            x[:] = 0
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
# One-image pipeline
# -----------------------------
def process_image(image_path: Path, depth_net: MidasTFLite, clahe_clip: float, clahe_tiles: tuple[int, int]):
    timings = {}
    with measure(timings, "read_image"):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"Failed to read {image_path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    with measure(timings, "clahe"):
        img_eq = apply_clahe_rgb_lab(img_rgb, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)

    with measure(timings, "fft_denoise_1"):
        s1_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY))

    with measure(timings, "log_kernel"):
        log_kernel = LoG_Kernel_Generator(9, 1.0)

    with measure(timings, "log_convolution"):
        log_conv = cv2.filter2D(s1_gray.astype(np.float32), -1, log_kernel)

    with measure(timings, "edge_detection"):
        edges = robust_laplacian_edge_detector(img_gray, log_conv, 150.0)

    with measure(timings, "fft_denoise_2"):
        s2_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY))
        s2_rgb = cv2.cvtColor(s2_gray, cv2.COLOR_GRAY2RGB)

    with measure(timings, "depth_inference_tflite"):
        depth = depth_net(s2_rgb)

    with measure(timings, "combine_depth_edge"):
        combined = np.where(edges > 0, 255, depth).astype(np.uint8)

    with measure(timings, "patchify"):
        R, C = 15, 15
        h, w = combined.shape
        patch = np.zeros((R, C))
        ph, pw = h // R, w // C
        for r in range(R):
            for c in range(C):
                y1, y2 = r * ph, min((r + 1) * ph, h)
                x1, x2 = c * pw, min((c + 1) * pw, w)
                patch[r, c] = np.mean(combined[y1:y2, x1:x2])
        patch /= 255.0

    with measure(timings, "astar_pathfinding"):
        start = (R - 1, C // 2)
        goal = np.unravel_index(np.argmin(patch[:5, :]), patch[:5, :].shape)
        path = astar_pathfinding(patch, start, goal)
        if not path:
            path = [start, goal]

    with measure(timings, "slope_direction"):
        x_vals = np.array([c - start[1] for r, c in path])
        y_vals = np.array([r - start[0] for r, c in path])
        m = np.sum(x_vals * y_vals) / (np.sum(x_vals**2) + 1e-6)
        m_img = (h / R) / (w / C) * m
        direction = slope_to_clock_if(m_img)

    total = sum(timings.values())
    print(f"\n🖼️ Image: {image_path.name}")
    for k, v in timings.items():
        print(f"  {k:<22} {ms(v):8.2f} ms")
    print(f"  {'-'*35}")
    print(f"  TOTAL{'':17} {ms(total):8.2f} ms | Direction: {direction}")
    return total


# -----------------------------
# Batch driver
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Process multiple images, print timings, and reuse TFLite model once.")
    ap.add_argument("--images", type=str, default="assets/*.jpg")
    ap.add_argument("--model", type=str, default="models/Midas-V2.tflite")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tiles", type=int, nargs=2, default=(8, 8))
    args = ap.parse_args()

    files = sorted(glob(args.images))
    if not files:
        print("No images found.")
        return

    print(f"Loading model once... ", end="", flush=True)
    t0 = time.perf_counter()
    depth_net = MidasTFLite(args.model)
    print(f"done ({ms(time.perf_counter() - t0):.1f} ms).")

    totals = []
    for img_path in files:
        try:
            total = process_image(Path(img_path), depth_net, args.clahe_clip, tuple(args.clahe_tiles))
            totals.append(total)
        except Exception as e:
            print(f"⚠️ Skipped {img_path}: {e}")

    if totals:
        avg = ms(np.mean(totals))
        print(f"\n✅ Processed {len(totals)} image(s). Avg total per image: {avg:.2f} ms.")
    else:
        print("No images processed.")


if __name__ == "__main__":
    main()
