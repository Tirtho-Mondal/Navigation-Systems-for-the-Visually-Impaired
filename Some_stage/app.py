from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, math, heapq, os, time, tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "models/MiDaS-V2.tflite"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------- Utility Functions ----------
def fft_denoise(image, keep_fraction=0.1):
    image_float = image.astype(np.float32) / 255.0
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    keep_r, keep_c = int(rows * keep_fraction / 2), int(cols * keep_fraction / 2)
    mask[crow - keep_r:crow + keep_r, ccol - keep_c:ccol + keep_c] = 1.0
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.clip(np.abs(img_back) * 255.0, 0, 255).astype(np.uint8)

def apply_hist_eq_rgb(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cdf = hist.cumsum(); cdf /= cdf[-1]
    lut = np.uint8(255 * cdf)
    v_eq = cv2.LUT(v, lut)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2RGB)

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

def custom_canny(image, low_thresh=50, high_thresh=150):
    Gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    Gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag = np.hypot(Gx, Gy)
    mag = mag / (mag.max() + 1e-6) * 255
    return (mag > high_thresh).astype(np.uint8) * 255

def heuristic(a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
def get_neighbors(pos, rows, cols):
    r, c = pos; nb=[]
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(1,1)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<rows and 0<=nc<cols: nb.append((nr,nc))
    return nb

def astar_pathfinding(cost_map, start, goal):
    rows, cols = cost_map.shape
    open_set = [(0, start)]
    came_from, g_score, f_score = {}, {start:0.0}, {start:heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path=[current]
            while current in came_from:
                current=came_from[current]; path.append(current)
            return path[::-1]
        for n in get_neighbors(current, rows, cols):
            tentative=g_score[current]+cost_map[n]
            if tentative<g_score.get(n, np.inf):
                came_from[n]=current
                g_score[n]=tentative
                f_score[n]=tentative+heuristic(n,goal)
                heapq.heappush(open_set,(f_score[n],n))
    return []

def slope_to_clock_if(m_img):
    m_math=-m_img; vx,vy=(1.0,m_math) if m_math>=0 else (-1.0,-m_math)
    angle=np.degrees(np.arctan2(vy,vx)); angle=np.clip(angle,0,180)
    labels=[("3",7.5),("2:30",22.5),("2",37.5),("1:30",52.5),("1",67.5),
            ("12:30",82.5),("12",97.5),("11:30",112.5),("11",127.5),
            ("10:30",142.5),("10",157.5),("9:30",172.5)]
    for name,th in labels:
        if angle<th: return name
    return "9"

def load_tflite_midas(path):
    interpreter=tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def run_tflite_depth(interpreter,input_details,output_details,image_rgb):
    img_input=cv2.resize(image_rgb,(256,256))/255.0
    img_input=np.expand_dims(img_input.astype(np.float32),axis=0)
    interpreter.set_tensor(input_details[0]['index'],img_input)
    interpreter.invoke()
    depth=interpreter.get_tensor(output_details[0]['index'])[0]
    return cv2.normalize(cv2.resize(depth,(image_rgb.shape[1],image_rgb.shape[0])),
                         None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

# ---------- Initialize model once ----------
print("Loading MiDaS-V2.tflite ...")
interpreter,input_details,output_details=load_tflite_midas(MODEL_PATH)
print("✅ Model ready.")

# ---------- Flask Routes ----------
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

    start_total = time.time()
    timing = {}

    # Load image
    t=time.time()
    img=cv2.imread(img_path)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    timing["Image Load"]=time.time()-t

    # Histogram Equalization
    t=time.time()
    img_eq=apply_hist_eq_rgb(img_rgb)
    timing["Histogram Equalization"]=time.time()-t

    # FFT Denoise
    t=time.time()
    smooth1_gray=fft_denoise(cv2.cvtColor(img_eq,cv2.COLOR_RGB2GRAY),keep_fraction=0.1)
    timing["FFT Denoise"]=time.time()-t

    # Edge Detection
    t=time.time()
    log_conv=cv2.Laplacian(cv2.GaussianBlur(smooth1_gray,(9,9),1.0),cv2.CV_32F)
    edges_log=robust_laplacian_edge_detector_fast(img_gray,log_conv,150)
    edges_canny=custom_canny(edges_log,50,150)
    edges=cv2.bitwise_or(edges_canny,edges_log)
    timing["Edge Detection"]=time.time()-t

    # Depth Estimation
    t=time.time()
    depth=run_tflite_depth(interpreter,input_details,output_details,
                            cv2.cvtColor(smooth1_gray,cv2.COLOR_GRAY2RGB))
    timing["Depth Estimation"]=time.time()-t

    # Combine + A*
    t=time.time()
    combined=np.where(edges>0,255.0,depth).astype(np.float32)
    combined=cv2.normalize(combined,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    PATCH_ROWS,PATCH_COLS=15,15
    h,w=combined.shape
    patch_h,patch_w=h//PATCH_ROWS,w//PATCH_COLS
    combo_patch=np.zeros((PATCH_ROWS,PATCH_COLS),dtype=np.float32)
    for r in range(PATCH_ROWS):
        for c in range(PATCH_COLS):
            combo_patch[r,c]=np.mean(combined[r*patch_h:(r+1)*patch_h,c*patch_w:(c+1)*patch_w])
    combo_patch_f=combo_patch/255.0
    start=(PATCH_ROWS-1,PATCH_COLS//2)
    goal_region=combo_patch[:5,:]
    goal=np.unravel_index(np.argmin(goal_region),goal_region.shape)
    path=astar_pathfinding(combo_patch_f,start,goal)
    direction="No path found"
    if path:
        x_vals=np.array([c-start[1] for r,c in path])
        y_vals=np.array([r-start[0] for r,c in path])
        m=np.sum(x_vals*y_vals)/(np.sum(x_vals**2)+1e-6)
        m_img=(h/PATCH_ROWS)/(w/PATCH_COLS)*m
        direction=slope_to_clock_if(m_img)
    timing["Pathfinding + Direction"]=time.time()-t

    total_time=time.time()-start_total
    return jsonify({
        "filename": file.filename,
        "direction": direction,
        "timing": timing,
        "total_time_sec": round(total_time,3)
    })

if __name__ == '__main__':
    app.run(debug=True)
