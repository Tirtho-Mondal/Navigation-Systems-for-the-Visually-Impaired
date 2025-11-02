import argparse
import csv
import json
import math
import heapq
import time
from contextlib import contextmanager
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# TFLite interpreter (tflite-runtime preferred; falls back to TF)
# -----------------------------
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
    image_float = image.astype(np.float32) / 255.0
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    keep_r = max(1, int(rows * keep_fraction / 2))
    keep_c = max(1, int(cols * keep_fraction / 2))
    mask[crow - keep_r : crow + keep_r, ccol - keep_c : ccol + keep_c] = 1.0
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_denoised = np.abs(img_back)
    return np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)


def apply_clahe_rgb_lab(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    # CLAHE on L channel in LAB (keeps colors natural)
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
    M, N = image_gray.shape
    edges = np.zeros_like(image_gray, dtype=np.float32)
    variance_image = local_variance_cv(image_gray, 3)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (log_image[i + 1, j] * log_image[i - 1, j] < 0) or (log_image[i, j + 1] * log_image[i, j - 1] < 0):
                edges[i, j] = 255.0 if variance_image[i, j] > threshold_value else 0.0
    return edges.astype(np.uint8)


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
# MiDaS (TFLite) loader + inference
# -----------------------------
class MidasTFLite:
    """Load once and reuse for all images."""
    def __init__(self, model_path: str):
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"TFLite model not found at: {model_p}")
        self.interpreter = Interpreter(model_path=str(model_p))
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
# One-image pipeline (reuses depth_net)
# -----------------------------
def process_image(
    image_path: Path,
    depth_net: MidasTFLite,
    out_dir: Path,
    plot: bool,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
):
    timings = {}
    stem = image_path.stem

    with measure(timings, "read_image"):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    with measure(timings, "clahe"):
        img_eq = apply_clahe_rgb_lab(img_rgb, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)

    with measure(timings, "fft_denoise_1"):
        smooth1_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
        smooth1_rgb = cv2.cvtColor(smooth1_gray, cv2.COLOR_GRAY2RGB)

    with measure(timings, "log_kernel"):
        sigma = 1.0
        kernel_size = int(max(3, (9 * sigma))) | 1
        log_kernel = LoG_Kernel_Generator(kernel_size, sigma)

    with measure(timings, "log_convolution"):
        log_conv = cv2.filter2D(smooth1_gray.astype(np.float32), ddepth=-1, kernel=log_kernel)

    with measure(timings, "edge_detection"):
        edges = robust_laplacian_edge_detector(img_gray, log_conv, threshold_value=150.0)

    with measure(timings, "fft_denoise_2"):
        smooth2_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
        smooth2_rgb = cv2.cvtColor(smooth2_gray, cv2.COLOR_GRAY2RGB)

    with measure(timings, "depth_inference_tflite"):
        depth = depth_net(smooth2_rgb)

    with measure(timings, "combine_depth_edge"):
        combined = np.where(edges > 0, 255, depth).astype(np.uint8)

    with measure(timings, "patchify"):
        PATCH_ROWS, PATCH_COLS = 15, 15
        h, w = combined.shape
        patch_h = max(1, h // PATCH_ROWS)
        patch_w = max(1, w // PATCH_COLS)
        combo_patch = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
        for r in range(PATCH_ROWS):
            for c in range(PATCH_COLS):
                y1, y2 = r * patch_h, min((r + 1) * patch_h, h)
                x1, x2 = c * patch_w, min((c + 1) * patch_w, w)
                patch = combined[y1:y2, x1:x2]
                combo_patch[r, c] = float(np.mean(patch)) if patch.size else 0.0
        combo_patch_f = combo_patch / 255.0

    with measure(timings, "astar_pathfinding"):
        start = (PATCH_ROWS - 1, PATCH_COLS // 2)
        goal_region = combo_patch_f[:5, :]
        goal = np.unravel_index(np.argmin(goal_region), goal_region.shape)
        path = astar_pathfinding(combo_patch_f, start, goal)
        if not path:
            path = [start, goal]

    with measure(timings, "slope_direction"):
        x_vals = np.array([c - start[1] for r, c in path], dtype=np.float32)
        y_vals = np.array([r - start[0] for r, c in path], dtype=np.float32)
        denom = float(np.sum(x_vals**2) + 1e-6)
        m = float(np.sum(x_vals * y_vals) / denom)
        m_img = (h / PATCH_ROWS) / (w / PATCH_COLS) * m
        c_img = (h - 1) - m_img * (w // 2)
        direction = slope_to_clock_if(m_img)

    with measure(timings, "draw_arrow"):
        img_arrow = img_rgb.copy()
        arrow_len = int(max(10, h / 3))
        y_end = max(h - 1 - arrow_len, 0)
        if abs(m_img) < 1e-6:
            x_end = w // 2
        else:
            x_end = int((y_end - c_img) / m_img)
        x_end = int(np.clip(x_end, 0, w - 1))
        cv2.arrowedLine(img_arrow, (w // 2, h - 1), (x_end, y_end), (255, 255, 0), 3, tipLength=0.2)

    out_dir.mkdir(parents=True, exist_ok=True)

    with measure(timings, "save_arrow_png"):
        (out_dir / f"{stem}_arrow.png").write_bytes(
            cv2.imencode(".png", cv2.cvtColor(img_arrow, cv2.COLOR_RGB2BGR))[1]
        )

    if plot:
        with measure(timings, "plot_and_save_grid"):
            images = [img_rgb, img_eq, smooth1_rgb, edges, depth, combined, combo_patch, img_arrow]
            titles = [
                "Original",
                "CLAHE (LAB L)",
                "FFT Denoise #1",
                "Edges (LoG)",
                "Depth Map",
                "Depth + Edge",
                "15x15 Patch + Path",
                f"Clock: {direction} (m={m_img:.3f})",
            ]
            fig, axes = plt.subplots(1, 8, figsize=(42, 5))
            for col_idx, (im, title) in enumerate(zip(images, titles)):
                ax = axes[col_idx]
                if isinstance(im, np.ndarray) and im.ndim == 2:
                    ax.imshow(im, cmap="gray")
                else:
                    ax.imshow(im)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(
                "Pipeline: CLAHE → FFT Denoise → Edge → FFT Denoise → Depth (TFLite) → Combine → Patch → A* → Clock",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            fig_path = out_dir / f"{stem}_pipeline.png"
            plt.savefig(fig_path.as_posix(), dpi=150)
            plt.close(fig)

    # return details needed by caller
    total = sum(timings.values())
    return {
        "image": image_path.name,
        "direction": direction,
        "slope_m_img": float(m_img),
        "timings_s": timings,
        "total_s": total,
    }


# -----------------------------
# Batch driver
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch process images with single-time TFLite model load (CLAHE + timings).")
    ap.add_argument("--images", type=str, default="assets/*.jpg", help="Glob for input images, e.g. 'assets/*.jpg'")
    ap.add_argument("--model", type=str, default="models/Midas-V2.tflite", help="Path to TFLite model")
    ap.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs & timings")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting to speed up")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clipLimit (default: 2.0)")
    ap.add_argument("--clahe-tiles", type=int, nargs=2, metavar=("TILE_W", "TILE_H"), default=(8, 8),
                    help="CLAHE tileGridSize as two ints (default: 8 8)")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    files = sorted(glob(args.images))
    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        raise FileNotFoundError(f"No images matched: {args.images}")

    # Load model ONCE
    global_timings = {}
    with measure(global_timings, "load_model"):
        depth_net = MidasTFLite(args.model)

    # Process each image reusing the same interpreter
    results = []
    for p in files:
        try:
            res = process_image(
                image_path=p,
                depth_net=depth_net,
                out_dir=out_dir,
                plot=(not args.no_plot),
                clahe_clip=args.clahe_clip,
                clahe_tiles=tuple(args.clahe_tiles),
            )
            results.append(res)
            # save per-image timings quickly too
            stem = p.stem
            # CSV
            csv_path = out_dir / f"{stem}_timings.csv"
            total = res["total_s"]
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "time_ms", "percent"])
                for k, v in res["timings_s"].items():
                    pct = (v / total * 100.0) if total > 0 else 0.0
                    w.writerow([k, f"{ms(v):.3f}", f"{pct:.3f}"])
                w.writerow(["TOTAL", f"{ms(total):.3f}", "100.000"])
            # JSON
            json_path = out_dir / f"{stem}_timings.json"
            with json_path.open("w") as f:
                json.dump(
                    {
                        "image": res["image"],
                        "direction": res["direction"],
                        "slope_m_img": res["slope_m_img"],
                        "steps_ms": {k: ms(v) for k, v in res["timings_s"].items()},
                        "total_ms": ms(total),
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    # ---- Batch-level reports ----
    # Big CSV with every image+step
    batch_csv = out_dir / "batch_timings.csv"
    with batch_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "step", "time_ms", "percent_of_image"])
        for res in results:
            total = res["total_s"]
            for k, v in res["timings_s"].items():
                pct = (v / total * 100.0) if total > 0 else 0.0
                w.writerow([res["image"], k, f"{ms(v):.3f}", f"{pct:.3f}"])
            w.writerow([res["image"], "TOTAL", f"{ms(total):.3f}", "100.000"])
        # add a global row for model load (applies to whole batch)
        w.writerow(["(global)", "load_model_once", f"{ms(global_timings['load_model']):.3f}", "n/a"])

    # Compact summary
    batch_summary = out_dir / "batch_summary.csv"
    with batch_summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "direction", "slope_m_img", "total_ms"])
        for res in results:
            w.writerow([res["image"], res["direction"], f"{res['slope_m_img']:.4f}", f"{ms(res['total_s']):.3f}"])

    # JSON overview
    overview_json = out_dir / "batch_overview.json"
    with overview_json.open("w") as f:
        json.dump(
            {
                "model_load_ms": ms(global_timings["load_model"]),
                "images": [
                    {
                        "image": r["image"],
                        "direction": r["direction"],
                        "slope_m_img": r["slope_m_img"],
                        "total_ms": ms(r["total_s"]),
                        "steps_ms": {k: ms(v) for k, v in r["timings_s"].items()},
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"\nLoaded model once in {ms(global_timings['load_model']):.2f} ms.")
    print(f"Processed {len(results)} image(s).")
    print(f"Wrote:\n - {batch_csv}\n - {batch_summary}\n - {overview_json}\nOutputs and per-image files are in: {out_dir}")


if __name__ == "__main__":
    main()
