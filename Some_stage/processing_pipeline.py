import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import tensorflow as tf
import os


# ===========================================
# FFT Denoise
# ===========================================
def fft_denoise(image, keep_fraction=0.1):
    image_float = image.astype(np.float32) / 255.0
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    keep_r = int(rows * keep_fraction / 2)
    keep_c = int(cols * keep_fraction / 2)
    mask[crow - keep_r:crow + keep_r, ccol - keep_c:ccol + keep_c] = 1.0
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_denoised = np.abs(img_back)
    img_denoised = np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)
    return img_denoised


# ===========================================
# Histogram Equalization (RGB → HSV)
# ===========================================
def apply_hist_eq_rgb(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    lut = np.uint8(255 * cdf_normalized)
    v_eq = cv2.LUT(v, lut)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)


# ===========================================
# Laplacian of Gaussian (LoG)
# ===========================================
def LoG(x, y, sigma):
    r2 = x**2 + y**2
    return -1 / (math.pi * sigma**4) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))

def LoG_Kernel_Generator(size, sigma):
    kernel = np.zeros((size, size))
    k = size // 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            kernel[k - j, k + i] = LoG(i, j, sigma)
    return kernel


# ===========================================
# Variance + Edge Detection
# ===========================================
def local_variance_cv(image, ksize=3):
    image = image.astype(np.float32)
    mean = cv2.blur(image, (ksize, ksize))
    mean_sq = cv2.blur(image**2, (ksize, ksize))
    variance = mean_sq - mean**2
    pad = ksize // 2
    variance[:pad, :] = variance[-pad:, :] = variance[:, :pad] = variance[:, -pad:] = 0
    return variance

def robust_laplacian_edge_detector(image, log_image, threshold_value):
    M, N = image.shape
    edges = np.zeros_like(image, dtype=np.float32)
    variance_image = local_variance_cv(image, 3)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (log_image[i+1, j] * log_image[i-1, j] < 0) or (log_image[i, j+1] * log_image[i, j-1] < 0):
                edges[i, j] = 255 if variance_image[i, j] > threshold_value else 0
    return edges


# ===========================================
# Canny + Non-Max Suppression
# ===========================================
def non_maximum_suppression(magnitude, angle):
    M, N = magnitude.shape
    nms = np.zeros((M, N), dtype=np.float32)
    angle = angle % 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = r = 255
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (22.5 <= a < 67.5):
                q, r = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif (67.5 <= a < 112.5):
                q, r = magnitude[i - 1, j], magnitude[i + 1, j]
            elif (112.5 <= a < 157.5):
                q, r = magnitude[i + 1, j + 1], magnitude[i - 1, j - 1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms[i, j] = magnitude[i, j]
    return nms


def custom_canny(image, low_thresh=50, high_thresh=150, sigma=1.0):
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    Gx = cv2.filter2D(blurred, -1, Kx)
    Gy = cv2.filter2D(blurred, -1, Ky)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.rad2deg(np.arctan2(Gy, Gx))
    nms = non_maximum_suppression(magnitude, angle)
    strong, weak = 255, 75
    result = np.zeros_like(nms, dtype=np.uint8)
    result[nms >= high_thresh] = strong
    result[(nms >= low_thresh) & (nms < high_thresh)] = weak
    for i in range(1, nms.shape[0]-1):
        for j in range(1, nms.shape[1]-1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result


# ===========================================
# Pathfinding + Angle Conversion
# ===========================================
def get_neighbors(pos, rows, cols):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),( -1,1),(1,-1),(1,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors

def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar_pathfinding(cost_map, start, goal):
    rows, cols = cost_map.shape
    open_set = [(0, start)]
    came_from, g_score, f_score = {}, {start: 0.0}, {start: heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for neighbor in get_neighbors(current, rows, cols):
            tentative_g = g_score[current] + cost_map[neighbor]
            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def slope_to_clock_if(m_img):
    m_math = -m_img
    vx, vy = (1.0, m_math) if m_math >= 0 else (-1.0, -m_math)
    angle = np.degrees(np.arctan2(vy, vx))
    angle = np.clip(angle, 0, 180)
    if angle < 7.5: return "3"
    elif angle < 22.5: return "2:30"
    elif angle < 37.5: return "2"
    elif angle < 52.5: return "1:30"
    elif angle < 67.5: return "1"
    elif angle < 82.5: return "12:30"
    elif angle < 97.5: return "12"
    elif angle < 112.5: return "11:30"
    elif angle < 127.5: return "11"
    elif angle < 142.5: return "10:30"
    elif angle < 157.5: return "10"
    elif angle < 172.5: return "9:30"
    else: return "9"


# ===========================================
# Full Sequential Pipeline (with Visualization)
# ===========================================
def run_full_pipeline(image_path, show_steps=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found!")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    img_eq = apply_hist_eq_rgb(img_rgb)
    smooth_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
    smooth_rgb = cv2.cvtColor(smooth_gray, cv2.COLOR_GRAY2RGB)

    sigma = 1
    kernel_size = int(9 * sigma) | 1
    log_kernel = LoG_Kernel_Generator(kernel_size, sigma)
    log_conv = cv2.filter2D(smooth_gray.astype(np.float32), -1, log_kernel)
    edges_log = robust_laplacian_edge_detector(img_gray, log_conv, 150)
    edges_canny = custom_canny(edges_log, 50, 150)
    edges = cv2.bitwise_or(edges_canny, edges_log.astype(np.uint8))

    MODEL_PATH = "Midas-V2.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"][1:3]
    img_resized = cv2.resize(smooth_rgb, (input_shape[1], input_shape[0]))
    img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img_input)
    interpreter.invoke()
    depth_map = interpreter.get_tensor(output_details[0]["index"])[0]
    depth = cv2.resize(depth_map, (img.shape[1], img.shape[0]))
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined = np.where(edges > 0, 255, depth).astype(np.uint8)

    PATCH_ROWS, PATCH_COLS = 15, 15
    height, width = combined.shape
    patch_h, patch_w = height // PATCH_ROWS, width // PATCH_COLS
    combo_patch = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
    for r in range(PATCH_ROWS):
        for c in range(PATCH_COLS):
            patch = combined[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w]
            combo_patch[r, c] = np.mean(patch)
    combo_patch_f = combo_patch / 255.0

    start = (PATCH_ROWS - 1, PATCH_COLS // 2)
    goal_region = combo_patch[:5, :]
    goal = np.unravel_index(np.argmin(goal_region), goal_region.shape)
    path = astar_pathfinding(combo_patch_f, start, goal)

    x_vals = np.array([c - start[1] for r, c in path])
    y_vals = np.array([r - start[0] for r, c in path])
    m = np.sum(x_vals * y_vals) / (np.sum(x_vals**2) + 1e-6)
    h, w = img_rgb.shape[:2]
    m_img = (h / PATCH_ROWS) / (w / PATCH_COLS) * m
    direction = slope_to_clock_if(m_img)

    if show_steps:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        images = [
            (img_rgb, "Original RGB"),
            (img_eq, "Histogram Equalized"),
            (smooth_gray, "FFT Denoised"),
            (edges_log, "LoG Edges"),
            (edges_canny, "Canny Edges"),
            (edges, "Combined Edges"),
            (depth, "Depth Map"),
            (combined, "Depth + Edge Combined"),
            (combo_patch, "15×15 Patch Grid (Cost Map)")
        ]
        for ax, (im, title) in zip(axes.ravel(), images):
            ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
            ax.set_title(title, fontsize=13)
            ax.axis('off')
        plt.suptitle(f"Pipeline Visualization — Direction: {direction}, Angle: {m_img:.3f}",
                     fontsize=18, fontweight="bold")
        plt.show()

    return direction, m_img


# ===========================================
# Run Example
# ===========================================
if __name__ == "__main__":
    img_path = "example.jpg"  # replace with your image
    direction, angle = run_full_pipeline(img_path)
    print(f"🧭 Direction: {direction} | 📐 Angle: {angle:.3f}")
