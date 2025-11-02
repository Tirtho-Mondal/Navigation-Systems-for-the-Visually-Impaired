import os
import math
import heapq
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


# -----------------------------
# Utils & Core Ops
# -----------------------------
def fft_denoise(image: np.ndarray, keep_fraction: float = 0.1) -> np.ndarray:
    """Simple low-pass denoise via FFT (expects single-channel uint8)."""
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


def apply_hist_eq_rgb_v_channel(image_rgb: np.ndarray) -> np.ndarray:
    """
    Histogram equalization on V channel in HSV space to preserve color balance.
    image_rgb expected in RGB order.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Build LUT via CDF
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf / cdf[-1]).flatten()
    lut = np.uint8(255 * cdf_normalized)

    v_eq = cv2.LUT(v, lut)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)


# -----------------------------
# LoG & Edge helper
# -----------------------------
def LoG(x: float, y: float, sigma: float) -> float:
    r2 = x**2 + y**2
    return -1.0 / (math.pi * sigma**4) * (1.0 - r2 / (2.0 * sigma**2)) * math.exp(-r2 / (2.0 * sigma**2))


def LoG_Kernel_Generator(size: int, sigma: float) -> np.ndarray:
    """Generates a standard (not rotated) LoG kernel centered at (k, k)."""
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
    # explicit zero borders (avoid weak edges on borders)
    variance[:pad, :] = 0
    variance[-pad:, :] = 0
    variance[:, :pad] = 0
    variance[:, -pad:] = 0
    return variance


def robust_laplacian_edge_detector(image_gray: np.ndarray, log_image: np.ndarray, threshold_value: float) -> np.ndarray:
    """
    Zero-crossing edge detector with local variance gating.
    image_gray, log_image: float32/uint8, same HxW. Returns uint8 {0,255}.
    """
    M, N = image_gray.shape
    edges = np.zeros_like(image_gray, dtype=np.float32)
    variance_image = local_variance_cv(image_gray, 3)

    # NOTE: loop is fine for clarity; image sizes are typical photos.
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # zero-crossing check (horizontal or vertical neighbors)
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
    # convert slope to a "clock" direction label
    m_math = -m_img
    vx, vy = (1.0, m_math) if m_math >= 0 else (-1.0, -m_math)
    angle = np.degrees(np.arctan2(vy, vx))
    angle = float(np.clip(angle, 0, 180))

    if angle < 7.5:
        return "3"
    elif angle < 22.5:
        return "2:30"
    elif angle < 37.5:
        return "2"
    elif angle < 52.5:
        return "1:30"
    elif angle < 67.5:
        return "1"
    elif angle < 82.5:
        return "12:30"
    elif angle < 97.5:
        return "12"
    elif angle < 112.5:
        return "11:30"
    elif angle < 127.5:
        return "11"
    elif angle < 142.5:
        return "10:30"
    elif angle < 157.5:
        return "10"
    elif angle < 172.5:
        return "9:30"
    else:
        return "9"


# -----------------------------
# MiDaS Depth (Torch Hub)
# -----------------------------
def load_midas(device: torch.device):
    # MiDaS small is fast and good enough
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval().to(device)

    midas_trans = torch.hub.load("intel-isl/MiDaS", "transforms")
    if hasattr(midas_trans, "small_transform"):
        transform = midas_trans.small_transform
    elif hasattr(midas_trans, "default_transform"):
        transform = midas_trans.default_transform
    elif hasattr(midas_trans, "dpt_transform"):
        transform = midas_trans.dpt_transform
    else:
        raise RuntimeError("No compatible MiDaS transform found.")
    return midas, transform


# -----------------------------
# Main
# -----------------------------
def main():
    # Paths
    ROOT = Path(__file__).resolve().parent
    ASSETS = ROOT / "assets"
    OUTDIR = ROOT / "outputs"
    OUTDIR.mkdir(exist_ok=True)

    # Collect the images named as 1.jpg..13.jpg (or whatever exists among them)
    indices = list(range(1, 14))
    img_paths = [ASSETS / f"{i}.jpg" for i in indices if (ASSETS / f"{i}.jpg").exists()]
    if not img_paths:
        raise FileNotFoundError(f"No images found at {ASSETS}. Expected files like 1.jpg, 2.jpg, ...")

    # Torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas, transform = load_midas(device)

    # Prepare figure depending on how many images were found
    n_rows = len(img_paths)
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(42, 4.5 * n_rows))
    # Ensure axes is 2D for unified indexing
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    titles = [
        "Original",
        "Histogram Eq",
        "FFT Denoise #1",
        "Edges (LoG)",
        "Depth Map",
        "Depth + Edge",
        "15x15 Patch + Path",
        "Arrow (Clock dir)"
    ]

    for row_idx, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"{img_path} not found or unreadable. Skipping...")
            continue

        # BGR -> RGB for consistency
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1) Histogram Equalization (V channel)
        img_eq = apply_hist_eq_rgb_v_channel(img_rgb)

        # 2) FFT Denoise #1 on gray from equalized image
        smooth1_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
        smooth1_rgb = cv2.cvtColor(smooth1_gray, cv2.COLOR_GRAY2RGB)

        # 3) Edge Detection (LoG)
        sigma = 1.0
        kernel_size = int(max(3, (9 * sigma))) | 1  # ensure odd, >=3
        log_kernel = LoG_Kernel_Generator(kernel_size, sigma)
        log_conv = cv2.filter2D(smooth1_gray.astype(np.float32), ddepth=-1, kernel=log_kernel)
        edges = robust_laplacian_edge_detector(img_gray, log_conv, threshold_value=150.0)

        # 4) FFT Denoise #2 (repeat denoise on equalized gray)
        smooth2_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
        smooth2_rgb = cv2.cvtColor(smooth2_gray, cv2.COLOR_GRAY2RGB)  # keep input RGB-like for transform

        # 5) Depth Map (MiDaS)
        with torch.no_grad():
            input_batch = transform(smooth2_rgb).to(device)
            pred = midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).squeeze(0)
        depth = cv2.normalize(pred.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 6) Combine Depth + Edge (edges brighter)
        edges_binary = (edges > 0).astype(np.uint8)
        combined = np.where(edges_binary == 1, 255, depth).astype(np.uint8)

        # 7) Patchify (15x15) & A*
        PATCH_ROWS, PATCH_COLS = 15, 15
        h, w = combined.shape
        patch_h = h // PATCH_ROWS
        patch_w = w // PATCH_COLS
        # guard for tiny images
        patch_h = max(1, patch_h)
        patch_w = max(1, patch_w)

        combo_patch = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
        for r in range(PATCH_ROWS):
            for c in range(PATCH_COLS):
                y1, y2 = r * patch_h, min((r + 1) * patch_h, h)
                x1, x2 = c * patch_w, min((c + 1) * patch_w, w)
                patch = combined[y1:y2, x1:x2]
                combo_patch[r, c] = float(np.mean(patch)) if patch.size else 0.0

        combo_patch_f = combo_patch / 255.0

        start = (PATCH_ROWS - 1, PATCH_COLS // 2)
        goal_region = combo_patch_f[:5, :]
        goal = np.unravel_index(np.argmin(goal_region), goal_region.shape)  # closest (smallest cost) in top band

        path = astar_pathfinding(combo_patch_f, start, goal)
        if not path:
            print(f"No path found for {img_path.name}")
            # still continue to plot everything else
            path = [start, goal]

        # 8) Slope & direction
        x_vals = np.array([c - start[1] for r, c in path], dtype=np.float32)
        y_vals = np.array([r - start[0] for r, c in path], dtype=np.float32)
        denom = float(np.sum(x_vals**2) + 1e-6)
        m = float(np.sum(x_vals * y_vals) / denom)

        m_img = (h / PATCH_ROWS) / (w / PATCH_COLS) * m  # scale to image aspect
        c_img = (h - 1) - m_img * (w // 2)
        direction = slope_to_clock_if(m_img)

        # 9) Draw Arrow (with safety for near-horizontal slope)
        img_arrow = img_rgb.copy()
        arrow_len = int(max(10, h / 3))
        y_end = max(h - 1 - arrow_len, 0)

        if abs(m_img) < 1e-6:
            x_end = w // 2
        else:
            x_end = int((y_end - c_img) / m_img)

        x_end = int(np.clip(x_end, 0, w - 1))
        cv2.arrowedLine(img_arrow, (w // 2, h - 1), (x_end, y_end), (255, 255, 0), 3, tipLength=0.2)

        # 10) Plot row
        images = [
            img_rgb,
            img_eq,
            smooth1_rgb,
            edges,
            depth,
            combined,
            combo_patch,
            img_arrow,
        ]

        for col_idx, (im, title) in enumerate(zip(images, titles)):
            ax = axes[row_idx, col_idx]
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")

        # add clock direction into the last subplot title
        axes[row_idx, -1].set_title(f"Clock: {direction} (m={m_img:.3f})")

    fig.suptitle(
        "Full Pipeline: Histogram → FFT Denoise → Edge → FFT Denoise → Depth → Combine → Patch → A* → Clock",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = (OUTDIR / "pipeline.png").as_posix()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
