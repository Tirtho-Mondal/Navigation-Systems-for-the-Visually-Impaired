from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, math, heapq, os, time, tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "models/MiDaS-V2.tflite"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
#                OPTIMIZED FUNCTIONS
# ==========================================================

def fft_denoise_fast(image, keep_fraction=0.1, scale=0.4):
    """Faster FFT denoise by aggressive downscaling before FFT."""
    small = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image_float = small.astype(np.float32) / 255.0
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)

    rows, cols = small.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    keep_r, keep_c = int(rows * keep_fraction / 2), int(cols * keep_fraction / 2)
    mask[crow - keep_r:crow + keep_r, ccol - keep_c:ccol + keep_c] = 1.0

    f_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    img_denoised = np.clip(img_back * 255.0, 0, 255).astype(np.uint8)

    return cv2.resize(img_denoised, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def apply_hist_eq_rgb(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2RGB)


def robust_edge_detection(image_gray):
    """Optimized edge detection using LoG + Sobel + bitwise fusion."""
    blur = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    log_edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=3)
    sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=3)
    sobel_combined = cv2.bitwise_or(sobelx, sobely)
    return cv2.bitwise_or(log_edges, sobel_combined)


def slope_to_clock_if(m_img):
    m_math = -m_img
    vx, vy = (1.0, m_math) if m_math >= 0 else (-1.0, -m_math)
    angle = np.degrees(np.arctan2(vy, vx))
    angle = np.clip(angle, 0, 180)
    labels = [("3",7.5),("2:30",22.5),("2",37.5),("1:30",52.5),("1",67.5),
              ("12:30",82.5),("12",97.5),("11:30",112.5),("11",127.5),
              ("10:30",142.5),("10",157.5),("9:30",172.5)]
    for name, th in labels:
        if angle < th:
            return name
    return "9"


def heuristic(a, b): 
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_neighbors(pos, rows, cols):
    r, c = pos
    nbs = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(1,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            nbs.append((nr, nc))
    return nbs


def astar_pathfinding(cost_map, start, goal):
    rows, cols = cost_map.shape
    open_set = [(0, start)]
    came_from, g_score = {}, {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for n in get_neighbors(current, rows, cols):
            tentative = g_score[current] + cost_map[n]
            if tentative < g_score.get(n, np.inf):
                came_from[n] = current
                g_score[n] = tentative
                f_score[n] = tentative + heuristic(n, goal)
                heapq.heappush(open_set, (f_score[n], n))
    return []


def load_tflite_midas(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def run_tflite_depth(interpreter, input_details, output_details, image_rgb):
    """Pre-resized depth estimation (optimized for TFLite)."""
    img_input = cv2.resize(image_rgb, (256, 256)) / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    depth = interpreter.get_tensor(output_details[0]['index'])[0]
    depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]))
    return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ==========================================================
#                   MODEL INIT
# ==========================================================
print("Loading MiDaS-V2.tflite ...")
model_start = time.perf_counter()
interpreter, input_details, output_details = load_tflite_midas(MODEL_PATH)
print(f"✅ Model ready in {time.perf_counter()-model_start:.3f}s")

# ==========================================================
#                   FLASK ROUTES
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_path = os.path.join(RESULT_DIR, file.filename)
    file.save(img_path)

    print(f"\n🖼️ Processing {file.filename} ...")
    t_all = time.perf_counter()
    timing = {}

    # 1. Load
    t = time.perf_counter()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    timing["Image Load"] = time.perf_counter() - t

    # 2. Histogram Equalization
    t = time.perf_counter()
    img_eq = apply_hist_eq_rgb(img_rgb)
    timing["Histogram Equalization"] = time.perf_counter() - t

    # 3. FFT Denoise (Fast)
    t = time.perf_counter()
    smooth_gray = fft_denoise_fast(cv2.cvtColor(img_eq, cv2.COLOR_RGB2GRAY), 0.1, 0.4)
    timing["FFT Denoise (fast)"] = time.perf_counter() - t

    # 4. Edge Detection
    t = time.perf_counter()
    edges = robust_edge_detection(smooth_gray)
    timing["Edge Detection"] = time.perf_counter() - t

    # 5. Depth Estimation
    t = time.perf_counter()
    depth = run_tflite_depth(interpreter, input_details, output_details,
                             cv2.cvtColor(smooth_gray, cv2.COLOR_GRAY2RGB))
    timing["Depth Estimation"] = time.perf_counter() - t

    # 6. Combine + Pathfinding
    t = time.perf_counter()
    combined = np.where(edges > 0, 255.0, depth).astype(np.float32)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
        direction = slope_to_clock_if(m_img)
    else:
        direction = "No path found"
    timing["Pathfinding + Direction"] = time.perf_counter() - t

    total_time = time.perf_counter() - t_all

    # Terminal summary
    print("\nStep                                | Time (sec)")
    print("-" * 50)
    for k, v in timing.items():
        print(f"{k:35s} | {v:10.3f}")
    print("-" * 50)
    print(f"{'Total Time':35s} | {total_time:10.3f}")
    print("=" * 50)
    print(f"✅ Direction Detected: {direction}")
    print("=" * 50 + "\n")

    return jsonify({
        "filename": file.filename,
        "direction": direction,
        "timing": timing,
        "total_time_sec": round(total_time, 3)
    })


if __name__ == '__main__':
    app.run(debug=True)
