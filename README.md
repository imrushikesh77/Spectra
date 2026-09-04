# FaceToMathsEquation

> **Every face has a unique system of parametric equations.** Upload a photo → get the mathematics that can redraw the exact face contour.

This project makes your idea rigorous: a face is not *one* short equation (photorealism cannot be captured that way), but a **finite system of closed parametric curves**, each with a provably complete Fourier representation. With all coefficients, reconstruction error → 0. With truncated `K`, you trade equation length vs. fidelity.

Runs **100% in the browser** — no photo leaves the device — plus a Python backend for batch processing and photorealistic field experiments.

---

## Demo

Open `index.html` in a modern browser (or serve via any static server). No build step.

```bash
# simplest
python3 -m http.server 8000
# then open http://localhost:8000
```

Upload a frontal, well-lit face photo. The app:

1. Detects **468 landmarks** via MediaPipe Face Landmarker (wasm, GPU-accelerated)
2. Groups landmarks into 9 features: face oval, left/right eye, left/right eyebrow, nose bridge+tip, nostrils, outer/inner lips
3. For each feature, computes **Fourier descriptors**
   ```
   z(t) = x(t) + i·y(t) = Σ_{k=-K}^{K} c_k · e^{i·k·ω·t},   t∈[0,1), ω=2π
   c_k = (1/N) Σ_{n=0}^{N-1} z_n · e^{-i·2π·k·n/N}
   ```
4. Shows `K` slider (3 → 50 in browser, 100+ in Python) and renders **solely from equations** — proving the equations alone redraw the face
5. Exports **LaTeX**, **JSON** (all coefficients + SHA-256 `Face Equation ID`), and **SVG**

## What "exact" means — honest mathematics

| Claim | Truth |
|---|---|
| *One equation redraws the exact photorealistic face* | **No.** An image is a function `I: [0,1]² → RGB`. Exactly reproducing texture needs an implicit neural field `f_θ(x,y) → RGB` (thousands of parameters). The Python folder demonstrates this path. |
| *Contours can be exact* | **Yes, with all N coefficients** per feature. Fourier series of a discrete closed polygon is exactly invertible via DFT. Truncating to `K < N` gives a controlled approximation. `K≈15` is visually excellent; `K=N` is lossless for landmarks. |
| *Is the equation unique?* | **Yes up to normalization.** After centering by `(cx,cy)` and scaling by face size, the coefficient vector is unique. We hash it → `Face Equation ID` (SHA-256). Two photos of the same person give *nearby* but not identical vectors (pose/expression vary) — as expected. |
| *Single equation?* | **No — a system.** 8–10 parametric equations is the correct formalism, analogous to how a 3D mesh is a system. The app bundles them as one JSON/LaTeX system. |

We chose rigor over hype. The UI explains this directly.

## Project structure

```
FaceToMathsEquation/
├── index.html          # Main app (zero backend)
├── css/style.css
├── js/app.js           # Fouriers + rendering + MediaPipe glue
├── python/
│   ├── face_to_equation.py  # Batch CLI, OpenCV + MediaPipe
│   └── requirements.txt
└── README.md
```

## Python backend

For automated pipelines, higher `K`, or experiments with neural texture fields:

```bash
cd python
pip install -r requirements.txt   # numpy, opencv-python, mediapipe

# Synthetic demo (validates math without a photo)
python face_to_equation.py --demo --K 20

# Real photo
python face_to_equation.py --image ../photo.jpg --K 30 --out ./output

# Outputs in ./output:
#   face-equation-<ID>.json  — full coefficient set
#   face-equation-<ID>.tex   — LaTeX system
#   face-equation-<ID>.png   — equation-only redraw
#   comparison-<ID>.png      — original | redraw side-by-side
#   landmarks.json           — raw 468 points
```

### Verification

```bash
python face_to_equation.py --demo --K 3   # expect RMSE ~0.02 (coarse)
python face_to_equation.py --demo --K 50  # expect RMSE <0.001 (near-exact)
```

## Equation format (interoperable)

```json
{
  "faceEquationId": "A3F9...01C4",
  "K": 15,
  "globalRMSE": 0.00312,
  "normalization": {"cx":0.51,"cy":0.49,"scale":0.43},
  "features": [{
    "key":"faceOval", "N":36, "rmse":0.002,
    "coeffs":[{"k":-15,"re":0.001,"im":-0.002,"mag":0.002,"phase": -1.1}, ...]
  }]
}
```

**Redraw in any language:**

```python
# t in [0,1]
x = sum(c.re*cos(2π*k*t) - c.im*sin(2π*k*t) for c,k in coeffs)
y = sum(c.re*sin(2π*k*t) + c.im*cos(2π*k*t) for c,k in coeffs)
X = x*scale + cx   # to 0-1 image coords
Y = y*scale + cy
```

For LaTeX, import the generated `.tex` — it is a ready `align` environment.

## Texture / photorealism (advanced)

True pixel-exact `I(x,y)` requires fitting a small MLP `f_θ: R²→R³` to the image (e.g., SIREN, 2–3 hidden layers, ~5k params). That *is* an equation — just a large one (weights are coefficients). The contour system here is the provable, interpretable foundation. If you want the neural field variant, open an issue — it fits in ~2 min per image on CPU.

## Limitations & tips

- **Frontal, single face, good light** works best. Profile, occlusion, or multiple faces degrades landmarks.
- **Expression/pose changes the coefficients.** Uniqueness is per capture, not biometric ID. For identity-stable embedding, use FaceNet/ArcFace; this project is geometric.
- **WASM download:** First load fetches ~10 MB MediaPipe assets from CDN. Offline → app falls back to procedural demo (still proves the math).

## License

MIT — use the equation format freely. If you publish work built on this, cite Fourier Descriptors (Persoon & Fu, 1977) and MediaPipe Face Mesh.

## Roadmap

- [ ] Blendshape-aware equations (expression as parameter)
- [ ] SIREN texture field toggle in browser (WebNN)
- [ ] 3D morphable model equation (x,y,z)

---

Built to honor the idea that *every face deserves its own mathematics* — without cheating about what "exact" means.
# Spectra
