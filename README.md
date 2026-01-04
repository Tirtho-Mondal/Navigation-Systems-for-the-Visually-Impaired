#  Navigation Systems for the Visually Impaired  

> **Course:** CSE 4128 – Image Processing and Computer Vision Laboratory  
> **Department:** Computer Science and Engineering  
> **University:** Khulna University of Engineering & Technology (KUET)

<p align="center">
  <img src="assets/logo.png" alt="KUET Logo" width="110">
</p>

---

## Submission

**Submitted to**  
Dr. Sk. Md. Masudul Ahsan  
Professor, Department of CSE, KUET  

Md Tajmilur Rahman  
Lecturer, Department of CSE, KUET  

**Submitted by**  
**Tirtho Mondal**  
Roll: 2007117  
4th Year, 1st Semester

---

## Table of Contents

- [Objectives](#objectives)
- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Contrast Enhancement](#contrast-enhancement)
  - [Denoising and Grayscale](#denoising-and-grayscale)
  - [Edge Detection](#edge-detection)
  - [Monocular Depth Estimation](#monocular-depth-estimation)
  - [Depth–Edge Fusion](#depthedge-fusion)
  - [Patchification and Cost Construction](#patchification-and-cost-construction)
  - [A* Path Planning](#a-path-planning)
    - [Worked Numerical Example](#worked-numerical-example)
  - [Near-Field Direction and Angle Estimation](#near-field-direction-and-angle-estimation)
  - [Clock Label Mapping](#clock-label-mapping)
- [Tools Used](#tools-used)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)
- [Figures](#figures)
- [How to Run (Prototype Guide)](#how-to-run-prototype-guide)
- [Repository Structure (Suggested)](#repository-structure-suggested)
- [License](#license)

---

## Objectives

- Robust scene perception using contrast enhancement, smoothing, edge detection, and depth estimation.  
- Grid-based path planning with A* on a fused depth-edge map.  
- Angle-to-clock-direction mapping for intuitive audio guidance.  
- Clean Flutter UI with text-to-speech for hands-free use.

---

## Introduction

Traditional mobility aids for blind and low-vision (BLV) users provide essential tactile feedback yet lack broader scene awareness in dynamic, unfamiliar environments. Leveraging the sensing and compute capabilities of commodity smartphones, this work develops a real-time navigation application that perceives obstacles and traversable space from the camera stream. The vision pipeline applies contrast enhancement, smoothing, and Canny edge detection alongside monocular depth estimation (MiDaS-V2), fusing depth and edges before down-sampling into a grid. An A* pathfinder selects a feasible route, whose near-field direction is mapped to intuitive “clock-face” prompts and delivered via text-to-speech within an accessible Flutter UI. The goal is practical, hands-free guidance that strengthens independence, safety, and confidence for BLV users.

<p align="center">
  <img src="assets/UI.png" alt="Main Interface and Settings of the BLV Navigation App" width="85%"><br>
  <em>Figure 1 — Main Interface and Settings of the BLV Navigation App</em>
</p>

---

## Methodology

A commodity smartphone is employed to deliver real-time navigation assistance by combining contrast-limited adaptive histogram equalization (CLAHE), Gaussian smoothing, grayscale conversion, Canny edge detection, monocular depth estimation, depth–edge fusion, grid-based cost construction, A* search, and angle-to–clock-direction mapping for audio guidance. The overall pipeline is depicted below.

<p align="center">
  <img src="assets/Group%2053.png" alt="Block Diagram of the App’s Image Processing Pipeline" width="85%"><br>
  <em>Figure 2 — Block Diagram of the App’s Image Processing Pipeline</em>
</p>

### Contrast Enhancement

Given an RGB frame $\mathbf{I}_{\mathrm{rgb}}$, conversion to CIE Lab space is followed by CLAHE on the luminance channel. Unless stated otherwise, the CLAHE clip limit is fixed to $\lambda=2.0$ with an $8\times8$ tile grid:

$$
\mathbf{I}_{\mathrm{lab}} = \mathrm{RGB2Lab}(\mathbf{I}_{\mathrm{rgb}}),\quad
\mathbf{I}_{\mathrm{eq}} = \mathrm{Lab2RGB}\!\left(\mathrm{CLAHE}(L;\,\lambda{=}2.0,\,T_x{=}8,\,T_y{=}8),\,a,\,b\right).
$$

<p align="center">
  <img src="assets/Histogram.png" alt="Input and Histogram Equalized Image" width="85%"><br>
  <em>Figure 3 — Input and Histogram Equalized Image</em>
</p>

### Denoising and Grayscale

Gaussian smoothing attenuates high-frequency sensor noise while preserving large-scale structure. A 2D isotropic Gaussian kernel $K_\sigma$ is applied to the contrast-enhanced frame, followed by luminance mapping to a single channel:

$$
\mathbf{I}_{\mathrm{blur}} = \mathbf{I}_{\mathrm{eq}} * K_\sigma.
$$

(Grayscale luminance mapping, commonly: $\mathbf{I}_{\mathrm{gray}} = 0.299\,R + 0.587\,G + 0.114\,B$.)

Gaussian kernel:

$$
K_\sigma(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right),\qquad \sum_{x,y}K_\sigma(x,y)=1.
$$

A $5\times5$ kernel with $\sigma=1.0$ is employed in practice.

<p align="center">
  <img src="assets/Gaussian.png" alt="Grayscale and Gaussian Smoothing Output" width="85%"><br>
  <em>Figure 4 — Conversion of Equalized Image to Grayscale and Smoothed Output</em>
</p>

### Edge Detection

Canny edge detection is applied to obtain a binary edge map $E$:

$$
E = \mathrm{Canny}\!\left(\mathbf{I}_{\mathrm{gray}};\,\tau_\ell,\tau_h\right),\qquad
E(u,v)\in\{0,1\}.
$$

Thresholds: $\tau_\ell = 100,\ \tau_h = 200$.

<p align="center">
  <img src="assets/canny%20edge.png" alt="Canny Edge Detection" width="85%"><br>
  <em>Figure 5 — Edge Detection Using the Canny Algorithm</em>
</p>

### Monocular Depth Estimation

A lightweight MiDaS-V2 model (TFLite) produces a raw depth $\hat{D}_{\mathrm{raw}}$, which is min–max normalized to 8-bit for fusion:

$$
\hat{D}_{\mathrm{raw}} = \mathcal{M}\!\left(\mathbf{I}_{\mathrm{blur}}\right),\qquad
D(u,v) = 255\,\frac{\hat{D}_{\mathrm{raw}}(u,v)-\min}{\max-\min}.
$$

<p align="center">
  <img src="assets/model.png" alt="Depth Image from MiDaS-V2" width="85%"><br>
  <em>Figure 6 — Depth Image Generated Using the MiDaS-V2 Model</em>
</p>

### Depth–Edge Fusion

To preserve geometric boundaries, edge pixels overwrite the normalized depth with a high constant:

$$
F(u,v)=
\begin{cases}
255,& \text{if } E(u,v)=1,\\
D(u,v),& \text{otherwise.}
\end{cases}
$$

<p align="center">
  <img src="assets/Merge.png" alt="Depth and Edge Fusion" width="85%"><br>
  <em>Figure 7 — Merging Depth and Edge Images for Enhanced Scene Understanding</em>
</p>

### Patchification and Cost Construction

The fused image $F\in\mathbb{R}^{H\times W}$ is partitioned into a $R\times C$ grid ($R=C=15$). Let $\Omega_{r,c}$ denote the pixel set of cell $(r,c)$. The mean intensity defines a traversability cost normalized to $[0,1]$:

$$
P[r,c] = \frac{1}{|\Omega_{r,c}|}\sum_{(u,v)\in\Omega_{r,c}} F(u,v),\qquad
\tilde{P}[r,c] = \frac{P[r,c]}{255}.
$$

Lower values in $\tilde{P}$ indicate more traversable regions.

<p align="center">
  <img src="assets/patch%20image.png" alt="15x15 Patch Division" width="85%"><br>
  <em>Figure 8 — Division of Merged Image into 15×15 Patches</em>
</p>

### A* Path Planning

Path computation is performed on $\tilde{P}$ using 8-connected neighbors. The start is fixed at the bottom-center cell $s=(R-1,\lfloor C/2\rfloor)$; the goal $g$ is selected as the lowest-cost cell among the top 5 rows. The A* evaluation is:

$$
f(n)=g(n)+h(n),
$$

with accumulated grid cost $g(n)$ and Euclidean heuristic

$$
h(n) = \| n - g \|_2 = \sqrt{(n_x - g_x)^2 + (n_y - g_y)^2}.
$$

<p align="center">
  <img src="assets/start%20node.png" alt="Start and Goal Node Selection" width="60%"><br>
  <em>Figure 9 — Start and Goal Node Selection in A* Algorithm</em>
</p>

<p align="center">
  <img src="assets/Picture4.png" alt="Path Generation Using A*" width="60%"><br>
  <em>Figure 10 — Path Generation Using the A* Algorithm</em>
</p>

#### Worked Numerical Example

Let Cost Matrix $C$ and normalized $P_{01}=C/255$:

$$
C =
\begin{bmatrix}
200 & 50 & 80 & 200\\
180 & 60 & 90 & 220\\
170 & 70 & 100 & 230\\
160 & 140 & 120 & 240
\end{bmatrix}
\qquad
P_{01} \approx
\begin{bmatrix}
0.7843 & 0.1961 & 0.3137 & 0.7843\\
0.7059 & 0.2353 & 0.3529 & 0.8627\\
0.6667 & 0.2745 & 0.3922 & 0.9020\\
0.6275 & 0.5490 & 0.4706 & 0.9412
\end{bmatrix}
$$

Start: $S=(3,2)$, Goal: $G=(0,1)$, Heuristic: $h(n)=\sqrt{(r-0)^2+(c-1)^2}$.

**Step 1: Expand $S=(3,2)$, $g(S)=0$**  
Neighbors: $\{(2,2),(3,1),(3,3),(2,1),(2,3)\}$

- $v=(2,1)$  
  $\text{cost} = 70/255 = 0.2745$  
  $g = 0 + 0.2745 = 0.2745$  
  $h = 2$  
  $f = 0.2745 + 2 = \boxed{2.2745}$

- $v=(2,2)$  
  $\text{cost} = 100/255 = 0.3922$  
  $g = 0 + 0.3922 = 0.3922$  
  $h = 2.2361$  
  $f = 0.3922 + 2.2361 = \boxed{2.6283}$

Smallest $f$: $(2,1)$.

**Step 2: Expand $(2,1)$, $g=0.2745$**  
Neighbors include $(1,1)$:

- $v=(1,1)$  
  $\text{cost}=60/255=0.2353$  
  $g=0.2745+0.2353=0.5098$  
  $h=1$  
  $f=\boxed{1.5098}$

Best $f$ remains $(1,1)$.

**Step 3: Expand $(1,1)$, $g=0.5098$**  
Neighbor includes goal $(0,1)$:

- $v=(0,1)$  
  $\text{cost}=50/255=0.1961$  
  $g=0.5098+0.1961=0.7059$  
  $h=0$  
  $f=0.7059$

**Goal reached.**

$$
\boxed{\text{Final Path: } (3,2)\rightarrow(2,1)\rightarrow(1,1)\rightarrow(0,1)}
$$

Total normalized cost: $g(G)=0.7059$  
Total raw cost: $70 + 60 + 50 = 180$

### Near-Field Direction and Angle Estimation

To obtain a stable and locally accurate heading direction, only the path points within the lower half of the $15\times15$ patch grid (rows $7$–$15$) are considered:

$$
\mathrm{lower}_{\mathrm{path}} \;=\; \{\, (r,c) \in \mathrm{path} \mid r \ge 7 \,\}.
$$

If no such points exist, all available path points are used as fallback.

Taking the start position $s=(r_s,c_s)$ as the origin, coordinate offsets are:

$$
\Delta x_i = c_i - c_s, \qquad \Delta y_i = r_i - r_s.
$$

A least-squares slope through the origin is computed with stabilizer $\varepsilon$:

$$
m = \frac{\sum_i \Delta x_i\,\Delta y_i}{\sum_i \Delta x_i^2 + \varepsilon}.
$$

To account for the aspect ratio between the patch grid and image pixels:

$$
m_{\mathrm{img}} = \frac{H / R}{W / C}\;m
$$

where $H,W$ denote image height and width (pixels), and $R,C$ the grid dimensions.

Finally, the near-field forward direction angle is:

$$
\theta = \arctan2(v_y, v_x), \qquad
(v_x, v_y) =
\begin{cases}
(1,\,-m_{\mathrm{img}}), & m_{\mathrm{img}} \geq 0,\\[4pt]
(-1,\,m_{\mathrm{img}}), & m_{\mathrm{img}} < 0,
\end{cases}
\quad \theta \in [0, \pi].
$$

> The negative sign in $-m_{\mathrm{img}}$ compensates for the inverted image $y$-axis (increases downward).

<p align="center">
  <img src="assets/line%20fitt.jpg" alt="Line Fitting and Angle Estimation" width="50%"><br>
  <em>Figure 11 — Line Fitting and Angle Estimation</em>
</p>

### Clock Label Mapping

To provide intuitive audio feedback, each estimated path angle $\theta$ is mapped to a corresponding clock direction label:

| **Angle $\theta$ (degrees)** | **Clock Label** |
|---|---|
| $0 \le \theta < 7.5$ | 3:00 |
| $7.5 \le \theta < 22.5$ | 2:30 |
| $22.5 \le \theta < 37.5$ | 2:00 |
| $37.5 \le \theta < 52.5$ | 1:30 |
| $52.5 \le \theta < 67.5$ | 1:00 |
| $67.5 \le \theta < 82.5$ | 12:30 |
| $82.5 \le \theta < 97.5$ | 12:00 |
| $97.5 \le \theta < 112.5$ | 11:30 |
| $112.5 \le \theta < 127.5$ | 11:00 |
| $127.5 \le \theta < 142.5$ | 10:30 |
| $142.5 \le \theta < 157.5$ | 10:00 |
| $157.5 \le \theta < 172.5$ | 9:30 |
| $172.5 \le \theta \le 180$ | 9:00 |

<p align="center">
  <img src="assets/Group%2054.png" alt="Clock Direction Mapping with Mobile Screen" width="50%"><br>
  <em>Figure 12 — Clock Direction Mapping with Mobile Screen</em>
</p>

<p align="center">
  <img src="assets/Response.png" alt="Application Response Indicating Safe Path Directions" width="85%"><br>
  <em>Figure 13 — Application Response Indicating Safe Path Directions</em>
</p>

---

## Tools Used

The proposed system integrates a combination of software frameworks, machine learning models, and image processing libraries to achieve real-time navigation assistance for visually impaired users.

### Programming Languages and Frameworks
1. **Python** – Backend, image processing, AI model integration.  
2. **Flask** – Lightweight Python web framework for server-side API.  
3. **Flutter** – Mobile frontend with accessible, voice-guided UI.

### Image Processing and Computer Vision Libraries
1. **OpenCV (cv2)** – CLAHE, Gaussian smoothing, grayscale, Canny edges.  
2. **NumPy** – Efficient numerical computation and matrix manipulation.  
3. **Base64 / BytesIO** – Image encoding and client–server data transfer.

### Depth Estimation
1. **TensorFlow Lite / TFLite Runtime** – Deploy MiDaS-V2 on mobile/edge.  
2. **MiDaS-V2** – Pre-trained monocular depth model for spatial understanding.

### Pathfinding and Algorithmic Components
1. **Pathfinding Algorithm (A\*)** – Optimal route on grid cost map.  
2. **`heapq` Module** – Priority queue for the A* open set.  
3. **`math` / `time` Modules** – Geometry, heuristic, performance timing.

### Additional Functional Components
I. **Text-to-Speech (TTS)** – Real-time spoken guidance.  
II. **Clock-Direction Mapping** – Converts angle to “clock-face” prompts.  
III. **CLAHE** – Robust local contrast enhancement.

---

## Limitations

The image processing module, while effective, exhibits several limitations. CLAHE may amplify noise or lose fine detail under poor or overexposed lighting. Gaussian smoothing can blur important edges, reducing precision. Canny detection is sensitive to lighting and threshold settings, potentially missing key edges or generating false positives. MiDaS-V2 provides relative depth (scale ambiguity), complicating absolute distance estimation—especially on flat or reflective surfaces. Minor misalignment during depth–edge fusion may introduce artifacts. Finally, the fixed $15\times15$ grid simplifies computation but may overlook small or narrow obstacles, limiting path-planning granularity.

---

## Conclusion

This project aims to bring independence and confidence to blind and visually impaired individuals through a smart, voice-guided mobile application. By combining image processing, depth estimation, and the A* pathfinding algorithm, the system can detect obstacles in real time and determine safe routes for navigation. With its intuitive, voice-controlled interface, users can move freely without constant physical assistance. Beyond the technical achievement, this reflects a compassionate use of technology—turning innovation into something that genuinely improves everyday life.

---

## References

> Use this section to list your sources or link to a `references.bib` file (the original LaTeX used `\bibliographystyle{ieeetr}` and `\bibliography{references}`).

---

## Figures

1. Main Interface and Settings — `assets/UI.png`  
2. Pipeline Block Diagram — `assets/Group 53.png`  
3. Histogram Equalization — `assets/Histogram.png`  
4. Grayscale & Gaussian Smoothing — `assets/Gaussian.png`  
5. Canny Edge Detection — `assets/canny edge.png`  
6. Depth (MiDaS-V2) — `assets/model.png`  
7. Depth + Edge Fusion — `assets/Merge.png`  
8. 15×15 Patch Division — `assets/patch image.png`  
9. Start/Goal Selection — `assets/start node.png`  
10. A* Path Generation — `assets/Picture4.png`  
11. Line Fitting & Angle — `assets/line fitt.jpg`  
12. Clock Mapping — `assets/Group 54.png`  
13. App Response — `assets/Response.png`  
14. KUET Logo — `assets/logo.png`

> **Note:** File paths with spaces are URL-encoded above (e.g., `Group%2053.png`). On disk, keep the original filenames in `assets/` or rename files to avoid spaces and update links accordingly.

---

## How to Run (Prototype Guide)

> This section complements the report with practical steps to reproduce a simple prototype.

### 1) Backend (Flask + Python)
- Python 3.10+ recommended
- Install deps:
  ```bash
  pip install flask opencv-python-headless numpy tflite-runtime
