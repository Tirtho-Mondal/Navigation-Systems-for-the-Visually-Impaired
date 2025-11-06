import cv2
import numpy as np
import math
import heapq
import os
import time
import tensorflow as tf

# =========================
#  FFT DENOISING FUNCTION
# =========================
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
    return np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)

# =========================
#  HISTOGRAM EQUALIZATION
# =========================
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

# =========================
#  FAST EDGE DETECTION
# =========================
def local_variance_cv(image, ksize=3):
    image = image.astype(np.float32)
    mean = cv2.blur(image, (ksize, ksize))
    mean_sq = cv2.blur(image**2, (ksize, ksize))
    variance = mean_sq - mean**2
    pad = ksize // 2
    variance[:pad, :] = variance[-pad:, :] = variance[:, :pad] = variance[:, -pad:] = 0
    return variance

def robust_laplacian_edge_detector_fast(image, log_image, threshold_value):
    variance_image = local_variance_cv(image, 3)
    zero_cross = (
        (np.roll(log_image, 1, axis=0) * np.roll(log_image, -1, axis=0) < 0) |
        (np.roll(log_image, 1, axis=1) * np.roll(log_image, -1, axis=1) < 0)
    )
    return np.where((zero_cross) & (variance_image > threshold_value), 255, 0).astype(np.uint8)

def custom_canny(image, low_thresh=50, high_thresh=150, sigma=1.0):
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    Gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255
    strong, weak = 255, 75
    result = np.zeros_like(magnitude, dtype=np.uint8)
    result[magnitude >= high_thresh] = strong
    result[(magnitude >= low_thresh) & (magnitude < high_thresh)] = weak
    return result

# =========================
#  A* PATHFINDING
# =========================
def get_neighbors(pos, rows, cols):
    r, c = pos
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(1,1)]:
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

# =========================
#  CLOCK DIRECTION
# =========================
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

# =========================
#  LOAD & RUN MIDAS (TFLITE)
# =========================
def load_tflite_midas(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def run_tflite_depth(interpreter, input_details, output_details, image_rgb):
    img_input = cv2.resize(image_rgb, (256, 256)) / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    depth = interpreter.get_tensor(output_details[0]['index'])[0]
    depth_resized = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]))
    return cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
#  MAIN PIPELINE (BATCH)
# =========================
ASSET_DIR = "assets"
MODEL_PATH = "models/MiDaS-V2.tflite"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading MiDaS-V2.tflite model...")
model_start = time.time()
interpreter, input_details, output_details = load_tflite_midas(MODEL_PATH)
model_load_time = time.time() - model_start
print(f"✅ Model loaded in {model_load_time:.3f}s\n")

summary = []
overall_start = time.time()

for img_idx in range(1, 15):  # 1.jpg – 14.jpg
    img_path = os.path.join(ASSET_DIR, f"{img_idx}.jpg")
    if not os.path.exists(img_path):
        print(f"⚠️ {img_path} not found — skipping.")
        continue

    print(f"\n🖼️ Processing {img_path} ...")
    timing = {}

    total_start = time.time()

    # Image load
    t = time.time()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    timing["Image Load"] = time.time() - t

    # Histogram Equalization
    t = time.time()
    img_eq = apply_hist_eq_rgb(img_rgb)
    timing["Histogram Equalization"] = time.time() - t

    # FFT Denoise
    t = time.time()
    smooth1_gray = fft_denoise(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), keep_fraction=0.1)
    timing["FFT Denoise"] = time.time() - t

    # Edge Detection
    t = time.time()
    log_conv = cv2.Laplacian(cv2.GaussianBlur(smooth1_gray, (9, 9), 1.0), cv2.CV_32F)
    edges_log = robust_laplacian_edge_detector_fast(img_gray, log_conv, 150)
    edges_canny = custom_canny(edges_log, 50, 150)
    edges = cv2.bitwise_or(edges_canny, edges_log)
    timing["Edge Detection"] = time.time() - t

    # Depth Estimation
    t = time.time()
    depth = run_tflite_depth(interpreter, input_details, output_details,
                             cv2.cvtColor(smooth1_gray, cv2.COLOR_GRAY2RGB))
    timing["Depth Estimation"] = time.time() - t

    # Combine
    t = time.time()
    combined = np.where(edges > 0, 255.0, depth).astype(np.float32)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    timing["Combine Depth + Edges"] = time.time() - t

    # Pathfinding + Direction
    t = time.time()
    PATCH_ROWS, PATCH_COLS = 15, 15
    h, w = combined.shape
    patch_h, patch_w = h // PATCH_ROWS, w // PATCH_COLS
    combo_patch = np.zeros((PATCH_ROWS, PATCH_COLS), dtype=np.float32)
    for r in range(PATCH_ROWS):
        for c in range(PATCH_COLS):
            combo_patch[r, c] = np.mean(combined[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w])
    combo_patch_f = combo_patch / 255.0
    start = (PATCH_ROWS - 1, PATCH_COLS // 2)
    goal_region = combo_patch[:5, :]
    goal = np.unravel_index(np.argmin(goal_region), goal_region.shape)
    path = astar_pathfinding(combo_patch_f, start, goal)
    if path:
        x_vals = np.array([c - start[1] for r, c in path])
        y_vals = np.array([r - start[0] for r, c in path])
        m = np.sum(x_vals * y_vals) / (np.sum(x_vals**2) + 1e-6)
        m_img = (h / PATCH_ROWS) / (w / PATCH_COLS) * m
        c_img = (h - 1) - m_img * (w // 2)
        direction = slope_to_clock_if(m_img)
    else:
        direction = "No path found"
        m_img, c_img = 0, 0
    timing["Pathfinding + Direction"] = time.time() - t

    # Visualization (save)
    t = time.time()
    img_arrow = img_rgb.copy()
    if path:
        arrow_len = int(h / 3)
        y_end = max(h - 1 - arrow_len, 0)
        if abs(m_img) < 1e-6 or np.isinf(m_img):
            x_end = w // 2
        else:
            x_end = int((y_end - c_img) / m_img)
            x_end = np.clip(x_end, 0, w - 1)
        cv2.arrowedLine(img_arrow, (w // 2, h - 1), (x_end, y_end),
                        (255, 255, 0), 3, tipLength=0.2)
    out_path = os.path.join("results", f"{img_idx:02d}_result.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(img_arrow, cv2.COLOR_RGB2BGR))
    timing["Visualization"] = time.time() - t

    total_time = time.time() - total_start

    # ------------------------------
    # Print detailed per-step table
    # ------------------------------
    print(f"\nStep                                | Time (sec)")
    print("-" * 50)
    for k, v in timing.items():
        print(f"{k:35s} | {v:10.3f}")
    print("-" * 50)
    print(f"{'Total (without visualization)':35s} | {total_time - timing['Visualization']:10.3f}")
    print(f"{'Total (with visualization)':35s} | {total_time:10.3f}")
    print("=" * 50)
    print(f"✅ Direction: {direction}\n")

    summary.append({
        "Image": f"{img_idx}.jpg",
        "Direction": direction,
        "Total": total_time
    })

overall_time = time.time() - overall_start
print("\n================ SUMMARY (All Images) ================")
print(f"{'Image':10s} | {'Direction':12s} | {'Total Time (s)':>15s}")
print("-" * 45)
for s in summary:
    print(f"{s['Image']:10s} | {s['Direction']:12s} | {s['Total']:15.3f}")
print("-" * 45)
print(f"Batch total time: {overall_time:.3f}s (avg {overall_time/len(summary):.3f}s/image)")
print("=======================================================")
