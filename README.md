# Spectra — Photo ⇄ Equation

> **Every picture is a chord.** Drop a photo → get its real Fourier equation → rebuild it pixel-perfect from the numbers alone.

Open `index.html` (or `python3 -m http.server 8000` → `http://localhost:8000`). No build step, no backend — 100% on-device.

## How it works

1. Your photo is cover-cropped to a square (`64`, `128`, or `256` — powers of two, so the FFT is exact) and forward-Fourier-transformed per channel:
   ```
   F(u,v) = FFT₂(f),   f(x,y) = (1/N) Σ_u Σ_v F(u,v)·e^{+i2π(ux+vy)/S}
   ```
2. The **equation file** (`.json`) holds *only* those frequency coefficients — no image bytes. Verified: no PNG/data-URL markers in the payload.
3. Drag **K** from 1 → FULL: the picture rebuilds by genuinely evaluating partial sums, from flat drone (K=1 = channel means) through sketch to exact.
4. At FULL K the inverse FFT reproduces every pixel — round-trip MSE ≈ 1e-27 (floating-point zero).
5. **Equation → Photo**: drop the `.json` back in and it develops through the inverse FFT. No original pixels involved.

Falsifiable proof: truncate to K=8 and the output degrades (MSE ~10⁴). If hidden pixels were used, K couldn't matter.

## Project structure

```
Spectra/
├── index.html          # App (zero backend)
├── css/style.css
├── js/app.js           # FFT engine + UI (radix-2 FFT, synthesis, IFFT)
├── python/
│   ├── face_to_equation.py  # Legacy contour backend (predates Spectra; kept for reference)
│   └── requirements.txt
└── README.md
```

## Equation format (v3)

```json
{
  "app": "spectra-patchbay", "version": 3, "S": 128, "id": "A3F9…01C4",
  "form": "f(x,y) = (1/N) Σ_u Σ_v F(u,v)·exp(+i·2π(ux+vy)/S), F = FFT2(f)",
  "channels": {
    "R": {"re_b64": "…Float64 LE…", "im_b64": "…"},
    "G": {...}, "B": {...}
  }
}
```

Sizes: 64px ≈ 150KB · 128px ≈ 1MB · 256px ≈ 4MB of real numbers.

**Morph (expression as parameter):** patch photos A and B, move `t` — coefficients interpolate `F(t) = (1−t)·F_A + t·F_B`, endpoints exact by construction.

## Honest limits

- Exactness is *at the chosen resolution* (64/128/256). A short readable equation cannot hold a photo — information theory, not effort. The `.json` *is* the equation, all numbers included.
- Square cover-crop only (FFT needs powers of two).
- Legacy `CHITRA-SUTRA-v1` PNG strings still paste, but are re-analysed into real coefficients and labeled as such.

## Roadmap

- [x] Real FFT both ways with coefficient-only equation files
- [x] K-dial partial-sum proof (output obeys K)
- [x] Expression morph via coefficient interpolation (`Morph` tab)
- [ ] SIREN neural-field toggle — **dropped**: a meaningful in-browser fit needs minutes of training for blobby quality; would read as broken next to exact FFT. Revisit with WebGPU.
- [ ] 3D morphable model — **dropped**: needs 3D priors/data this project doesn't have; would be theater, not math.

## License

MIT.
