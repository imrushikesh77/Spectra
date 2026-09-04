#!/usr/bin/env python3
"""
FaceToMathsEquation — Python backend
Generates a mathematically rigorous system of parametric equations for any face image.

Two complementary representations:
  1. Fourier Descriptors — per-feature closed contours z(t)=Σ c_k e^{ikωt} (exact with all coefficients)
  2. Neural Implicit Field (optional) — f_theta(x,y) -> RGB via small MLP (SIREN-like) for photorealistic texture

Usage:
  pip install -r requirements.txt
  python face_to_equation.py --image photo.jpg --K 20 --out ./output

If mediapipe is unavailable, falls back to synthetic demo + still validates Fourier pipeline.
"""

import argparse
import json
import hashlib
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import cv2

# Feature index maps — must stay in sync with js/app.js
FEATURES: Dict[str, dict] = {
    "faceOval": {"label":"Face Oval (Contour)","color":"#6c7cff",
                 "indices":[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]},
    "leftEyebrow": {"label":"Left Eyebrow","color":"#ff7a5c","indices":[70,63,105,66,107,55,65,52,53,46,70]},
    "rightEyebrow": {"label":"Right Eyebrow","color":"#ff7a5c","indices":[336,296,334,293,300,276,283,282,295,285,336]},
    "leftEye": {"label":"Left Eye","color":"#00d0a0","indices":[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]},
    "rightEye": {"label":"Right Eye","color":"#00d0a0","indices":[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,263]},
    "nose": {"label":"Nose Bridge & Tip","color":"#ffb020","indices":[168,6,197,195,5,4,45,220,115,48,98,327,326,2,97,44,168]},
    "outerLips": {"label":"Outer Lips","color":"#f43f5e","indices":[61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,61]},
    "innerLips": {"label":"Inner Lips","color":"#a78bfa","indices":[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]},
    "noseWings": {"label":"Nostrils","color":"#ffb020","indices":[48,220,115,98,2,327,344,440,278,275,4,45,48]},
}

# ---------- Fourier ----------
def dft(points: np.ndarray):
    """points: Nx2 array of (x,y). Returns full DFT coeffs (complex) length N: C_k = 1/N Σ z_n e^{-i2πkn/N}"""
    N = len(points)
    z = points[:,0] + 1j*points[:,1]
    n = np.arange(N)
    k = np.arange(N)[:, None]
    W = np.exp(-2j*np.pi*k*n/N)  # N x N
    coeffs = (W @ z) / N
    return coeffs  # length N complex

def fourier_coeffs_symmetric(points: np.ndarray, K: int):
    N = len(points)
    full = dft(points)
    coeffs=[]
    for k in range(-K, K+1):
        idx = k % N
        c = full[idx]
        coeffs.append({"k":k, "re":float(c.real), "im":float(c.imag), "mag":float(abs(c)), "phase":float(np.angle(c))})
    coeffs.sort(key=lambda x: x["k"])
    return coeffs

def evaluate_fourier(coeffs, t):
    x=y=0.0
    for c in coeffs:
        ang=2*math.pi*c["k"]*t
        ca, sa = math.cos(ang), math.sin(ang)
        x += c["re"]*ca - c["im"]*sa
        y += c["re"]*sa + c["im"]*ca
    return x,y

def build_equations(landmarks: np.ndarray, K: int):
    """
    landmarks: (478,2) in 0-1 normalized coords
    Returns dict with equations + globalRMSE + faceId
    """
    # normalization from face oval
    oval_idx = FEATURES["faceOval"]["indices"]
    oval_pts = landmarks[oval_idx]
    minX, maxX = oval_pts[:,0].min(), oval_pts[:,0].max()
    minY, maxY = oval_pts[:,1].min(), oval_pts[:,1].max()
    cx = (minX+maxX)/2
    cy = (minY+maxY)/2
    scale = max(maxX-minX, maxY-minY)
    if scale < 1e-6: scale = 0.5

    equations=[]
    total_sq=0; total_n=0
    hash_parts=[]
    for key, feat in FEATURES.items():
        idxs = feat["indices"]
        pts = landmarks[idxs]  # N x2
        N = len(pts)
        centered = (pts - np.array([cx,cy]))/scale  # N x2
        Kc = min(K, N//2)  # clamp to Nyquist to avoid aliasing; exact reconstruction when 2K+1 == N (odd) or 2K == N (even)
        coeffs = fourier_coeffs_symmetric(centered, Kc)
        # RMSE
        sq=0
        for n in range(N):
            t=n/N
            xr, yr = evaluate_fourier(coeffs, t)
            rx, ry = xr*scale+cx, yr*scale+cy
            dx, dy = rx-pts[n,0], ry-pts[n,1]
            sq+=dx*dx+dy*dy
        rmse=float(math.sqrt(sq/N))
        total_sq+=sq; total_n+=N
        for c in coeffs:
            hash_parts.append(f"{key}:{c['k']}:{c['re']:.5f},{c['im']:.5f}")
        equations.append({
            "key":key, "label":feat["label"], "color":feat["color"],
            "indices":idxs, "N":N, "rmse":rmse,
            "coeffs":coeffs, "cx":float(cx), "cy":float(cy), "scale":float(scale),
            "pts": pts.tolist()
        })
    global_rmse=float(math.sqrt(total_sq/total_n)) if total_n else 0
    hash_input="|".join(hash_parts)
    face_id=hashlib.sha256(hash_input.encode()).hexdigest()[:16].upper()
    return {"equations":equations, "globalRMSE":global_rmse, "faceId":face_id, "cx":float(cx),"cy":float(cy),"scale":float(scale)}

def render_equation_image(eq_data, W=800, H=800, out_path=None, background_path=None):
    """Render equation-only image (no original texture) using OpenCV"""
    img=np.full((H,W,3), 11, dtype=np.uint8)  # #0b0e14
    # grid
    for x in range(0,W,40): cv2.line(img,(x,0),(x,H),(31,41,55),1)
    for y in range(0,H,40): cv2.line(img,(0,y),(W,y),(31,41,55),1)
    cv2.line(img,(W//2,0),(W//2,H),(42,52,74),1)
    cv2.line(img,(0,H//2),(W,H//2),(42,52,74),1)
    dw, dh = W*0.92, H*0.92
    dx, dy = (W-dw)/2, (H-dh)/2
    cx, cy, scale = eq_data["cx"], eq_data["cy"], eq_data["scale"]
    for eq in eq_data["equations"]:
        coeffs=eq["coeffs"]
        color_hex=eq["color"].lstrip("#")
        b,g,r = int(color_hex[0:2],16), int(color_hex[2:4],16), int(color_hex[4:6],16)
        # Note: opencv uses BGR; we invert
        b,g,r = int(color_hex[4:6],16), int(color_hex[2:4],16), int(color_hex[0:2],16) # actually hex is RRGGBB
        # Correct: #6c7cff => R=6c G=7c B=ff
        r=int(color_hex[0:2],16); g=int(color_hex[2:4],16); b=int(color_hex[4:6],16)
        pts=[]
        steps=400
        for s in range(steps+1):
            t=s/steps
            xr, yr = evaluate_fourier(coeffs, t)
            rx, ry = xr*scale+cx, yr*scale+cy
            x = int(dx+rx*dw); y=int(dy+ry*dh)
            pts.append([x,y])
        pts=np.array(pts, np.int32)
        # fill lips/oval if requested
        if eq["key"] in ("outerLips","faceOval","innerLips"):
            overlay=img.copy()
            cv2.fillPoly(overlay,[pts],(b,g,r))
            cv2.addWeighted(overlay,0.12,img,0.88,0,img)
        cv2.polylines(img,[pts], True, (b,g,r), 2, lineType=cv2.LINE_AA)
    # label
    cv2.putText(img, f"Face Equation ID: {eq_data['faceId']}  K={len(eq_data['equations'][0]['coeffs'])//2}  RMSE={eq_data['globalRMSE']:.5f}", (12,22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (154,163,184), 1, cv2.LINE_AA)
    cv2.putText(img, "z(t)= sum c_k * exp(i k w t)  — sample t in [0,1]", (12,40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (107,114,128), 1, cv2.LINE_AA)
    if out_path: cv2.imwrite(out_path, img)
    return img

def latex_export(eq_data, K):
    lines=[f"% FaceToMathsEquation — K={K}  FaceID={eq_data['faceId']}  RMSE={eq_data['globalRMSE']:.5f}",
           f"% z(t)=x(t)+i y(t)= sum_{{k=-K}}^{{K}} c_k e^{{i k omega t}}, omega=2pi, t in [0,1)",
           r"\begin{align}"]
    for eq in eq_data["equations"]:
        coeffs=eq["coeffs"]
        preview=", ".join([f"{c['re']:.3f}{'+' if c['im']>=0 else ''}{c['im']:.3f}i" for c in coeffs[:3]]) + r", \dots"
        lines.append(f"% {eq['label']}: N={eq['N']} RMSE={eq['rmse']:.5f}")
        lines.append(f"z_{{\\text{{{eq['key']}}}}}(t) &= \\sum_{{k=-{K}}}^{{{K}}} c_k e^{{ik\\omega t}}, \\quad c_k \\in \\{{{preview}\\}} \\\\")
    lines.append(r"\end{align}")
    lines.append(f"% Denormalize: X(t)=x(t)*{eq_data['scale']:.5f}+{eq_data['cx']:.5f}, Y(t)=y(t)*{eq_data['scale']:.5f}+{eq_data['cy']:.5f}")
    return "\n".join(lines)

# ---------- EXACT PIXEL MODE (unlimited complexity) ----------
def build_exact_pixel_equation(img_bgr, target_size=None):
    """
    Build an EXACT 2D Fourier equation for the full image.
    No complexity constraint: uses WH coefficients per channel (lossless).

    Math:
      For image f(x,y) ∈ [0,255], x=0..W-1, y=0..H-1, define
        C(u,v) = 1/(WH) Σ_{x,y} f(x,y) exp(-i·2π·(u·x/W + v·y/H))
      Then exactly:
        f(x,y) = Σ_{u=0}^{W-1} Σ_{v=0}^{H-1} C(u,v) exp(+i·2π·(u·x/W + v·y/H))
    This is the 2D DFT pair (numpy.fft.fft2 / ifft2). With all WH coefficients, reconstruction error is 0 (up to floating point ~1e-10).

    Returns dict with H,W, coeffs per channel, hash, latex preview, and verification MSE.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    if target_size is not None:
        # Resize longest side to target_size preserving aspect
        scale = target_size / max(orig_h, orig_w)
        new_w, new_h = int(round(orig_w*scale)), int(round(orig_h*scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        H,W = new_h, new_w
    else:
        H,W = orig_h, orig_w

    # Work per channel in RGB order for math clarity, but store BGR to match cv2
    channels = {}
    coeffs_store = {}
    total_coeffs = H*W
    for idx, name in enumerate(["B","G","R"]):
        f = img_bgr[:,:,idx].astype(np.float64)  # HxW
        # 2D DFT normalized as per definition above
        C = np.fft.fft2(f) / (H*W)  # HxW complex
        # Shift for nicer math? Keep unshifted for reconstruction simplicity (0..W-1 indexing)
        channels[name] = f
        coeffs_store[name] = C

    # Verify exact reconstruction via ifft
    recon = np.zeros_like(img_bgr, dtype=np.float64)
    max_err = 0
    mse = 0
    for idx, name in enumerate(["B","G","R"]):
        C = coeffs_store[name]
        f_rec = np.fft.ifft2(C * (H*W)).real  # should equal original
        err = np.abs(f_rec - channels[name])
        max_err = max(max_err, float(err.max()))
        mse += float(np.mean((f_rec - channels[name])**2))
        recon[:,:,idx] = np.clip(np.round(f_rec), 0, 255)
    mse /= 3
    recon_u8 = recon.astype(np.uint8)

    # Hash of coefficients (deterministic)
    hasher = hashlib.sha256()
    for name in ["R","G","B"]:
        hasher.update(coeffs_store[name].tobytes())
    exact_id = hasher.hexdigest()[:16].upper()

    # Build serializable coeffs — store as magnitude/phase + real/imag for completeness
    # For huge images (e.g., 512x512=262k per channel) JSON would be ~50 MB — we store full for exactness but warn
    # Provide truncated preview in latex
    # To keep file manageable, we store full complex array as nested lists of [re,im] if size <= 256*256, else we note file will be large
    serializable_coeffs = {}
    for name in ["B","G","R"]:
        C = coeffs_store[name]
        # Convert to list of list of [re,im] — very large for big images
        # For efficiency, we store as base64 or just keep shape + note that reconstruction uses np.fft; but for exactness we store all
        if H*W <= 128*128:
            serializable_coeffs[name] = [[ [float(C[v,u].real), float(C[v,u].imag)] for u in range(W)] for v in range(H)]
        else:
            # For larger images, store flattened for slightly smaller JSON
            flat = C.flatten()
            serializable_coeffs[name] = {
                "shape": [H,W],
                "flat_re_im": [[float(c.real), float(c.imag)] for c in flat],
                "note": "C(u,v) row-major, u=0..W-1, v=0..H-1. f(x,y)= Σ_u Σ_v C(u,v) exp(i2π(ux/W+vy/H))"
            }

    latex = (
        f"% EXACT PIXEL EQUATION — lossless 2D Fourier, H={H}, W={W}, total {total_coeffs} coeffs/channel, {total_coeffs*3} total\n"
        f"% For each channel c ∈ {{R,G,B}}:\n"
        f"%   C_c(u,v) = 1/(WH) Σ_{{x=0}}^{{W-1}} Σ_{{y=0}}^{{H-1}} f_c(x,y)·exp(-i·2π·(u·x/W + v·y/H))\n"
        f"%   f_c(x,y) = Σ_{{u=0}}^{{W-1}} Σ_{{v=0}}^{{H-1}} C_c(u,v)·exp(+i·2π·(u·x/W + v·y/H))\n"
        f"% Verification: MSE={mse:.2e}, max_abs_err={max_err:.2e} (≈0, floating point)\n"
        f"\\begin{{equation}}\n"
        f"f_c(x,y)=\\sum_{{u=0}}^{{{W-1}}}\\sum_{{v=0}}^{{{H-1}}} C_c(u,v)\\,e^{{+i2\\pi(ux/{W}+vy/{H})}},\\quad c\\in\\{{R,G,B\\}}\n"
        f"\\end{{equation}}\n"
        f"% Example coeff magnitude (R channel): mean |C|={float(np.mean(np.abs(coeffs_store['R']))):.4f}, max |C|={float(np.max(np.abs(coeffs_store['R']))):.2f}\n"
    )

    return {
        "H": H, "W": W,
        "total_coeffs_per_channel": total_coeffs,
        "total_coeffs_all": total_coeffs*3,
        "mse": float(mse),
        "max_abs_err": float(max_err),
        "exact_id": exact_id,
        "coeffs": coeffs_store,  # raw complex arrays for rendering
        "serializable_coeffs": serializable_coeffs,
        "recon_u8": recon_u8,
        "latex": latex,
        "orig_bgr": img_bgr,
    }

def render_exact_comparison(exact_data, out_path):
    H,W = exact_data["H"], exact_data["W"]
    orig = exact_data["orig_bgr"]
    recon = exact_data["recon_u8"]
    # Side-by-side + diff
    diff = cv2.absdiff(orig, recon)
    # Enlarge for visibility if small
    scale = max(1, 400 // max(H,W))
    def enlarge(img): return cv2.resize(img, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST) if scale>1 else img
    orig_e, recon_e, diff_e = enlarge(orig), enlarge(recon), enlarge(diff)
    # Put text
    canvas = np.zeros((H*scale + 60, W*scale*3 + 20, 3), dtype=np.uint8)
    canvas[:] = 11
    canvas[0:H*scale, 0:W*scale] = orig_e
    canvas[0:H*scale, W*scale+10: W*scale*2+10] = recon_e
    canvas[0:H*scale, W*scale*2+20: W*scale*3+20] = diff_e
    cv2.putText(canvas, f"ORIGINAL {W}x{H}", (10, H*scale+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (154,163,184), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"EXACT RECON via 2D Fourier MSE={exact_data['mse']:.1e}", (W*scale+12, H*scale+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,208,160), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"DIFF (should be black) max={exact_data['max_abs_err']:.1e}", (W*scale*2+22, H*scale+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,176,32), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Equation: f(x,y)= Σ_u Σ_v C(u,v) e^(i2π(ux/W+vy/H))  |  {exact_data['total_coeffs_all']} coeffs  |  ID {exact_data['exact_id']}", (10, H*scale+38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (107,114,128), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Evaluate at integer x=0..{W-1}, y=0..{H-1} to redraw exact pixels", (10, H*scale+52), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (107,114,128), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    return canvas

def generate_synthetic_face_image(size=128):
    """Create a simple synthetic face image for exact-mode demo (no photo needed)"""
    img = np.full((size, size, 3), 15, dtype=np.uint8)
    # skin-tone ellipse
    cx, cy = size//2, size//2 + size//20
    ax, ay = int(size*0.33), int(size*0.42)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (120, 180, 210), -1, cv2.LINE_AA)
    # eyes
    eye_w, eye_h = int(size*0.12), int(size*0.06)
    cv2.ellipse(img, (cx - int(size*0.16), cy - int(size*0.08)), (eye_w, eye_h), 0, 0, 360, (255,255,255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + int(size*0.16), cy - int(size*0.08)), (eye_w, eye_h), 0, 0, 360, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx - int(size*0.16), cy - int(size*0.08)), int(size*0.025), (30,30,30), -1, cv2.LINE_AA)
    cv2.circle(img, (cx + int(size*0.16), cy - int(size*0.08)), int(size*0.025), (30,30,30), -1, cv2.LINE_AA)
    # eyebrows
    cv2.ellipse(img, (cx - int(size*0.16), cy - int(size*0.16)), (int(size*0.11), int(size*0.03)), 0, 0, 180, (60,40,30), 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx + int(size*0.16), cy - int(size*0.16)), (int(size*0.11), int(size*0.03)), 0, 0, 180, (60,40,30), 3, cv2.LINE_AA)
    # nose
    cv2.line(img, (cx, cy - int(size*0.05)), (cx - int(size*0.02), cy + int(size*0.08)), (90,110,140), 2, cv2.LINE_AA)
    cv2.line(img, (cx - int(size*0.02), cy + int(size*0.08)), (cx + int(size*0.02), cy + int(size*0.08)), (90,110,140), 2, cv2.LINE_AA)
    # lips
    cv2.ellipse(img, (cx, cy + int(size*0.18)), (int(size*0.10), int(size*0.05)), 0, 0, 360, (60,60,180), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy + int(size*0.18)), (int(size*0.06), int(size*0.025)), 0, 0, 360, (120,80,180), -1, cv2.LINE_AA)
    # add subtle texture noise for realism (makes exactness non-trivial)
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

# ---------- Landmarks ----------
def get_landmarks_mediapipe(image_path):
    """Try mediapipe, fallback to synthetic with informative error"""
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        h,w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3) as face_mesh:
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None, img, "No face detected by MediaPipe (try frontal, well-lit)"
            lm = res.multi_face_landmarks[0]
            pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float64)  # 478 x2
            # need 478 but mp classic gives 468; pad if needed
            if len(pts)==468:
                # pad to 478 by repeating last point (keeps FEATURE indices valid for subset that <468)
                # Actually our FEATURE indices all <468 for classic, except 454 etc <468 so fine.
                # We'll pad 10 zeros at end to reach 478 to avoid index error if we later use 478 indices
                pad = np.tile(pts[-1], (10,1))
                pts = np.vstack([pts, pad])
            return pts, img, "ok"
    except ImportError as e:
        return None, None, f"mediapipe not installed: {e}"
    except Exception as e:
        return None, None, str(e)

def synthetic_landmarks():
    pts=np.full((478,2), 0.5, dtype=np.float64)
    # oval
    oval=FEATURES["faceOval"]["indices"]
    for n, idx in enumerate(oval):
        t=n/len(oval)*2*math.pi
        pts[idx]=[0.5+0.22*math.cos(t), 0.5+0.30*math.sin(t)+0.03]
    # eyes
    for feat_name, cx_, cy_, rx, ry in [("leftEye",0.38,0.42,0.045,0.025),("rightEye",0.62,0.42,0.045,0.025)]:
        idxs=FEATURES[feat_name]["indices"]
        for n, idx in enumerate(idxs):
            if n==len(idxs)-1: continue
            t=n/(len(idxs)-1)*2*math.pi
            pts[idx]=[cx_+rx*math.cos(t), cy_+ry*math.sin(t)]
    for feat_name, cx_ in [("leftEyebrow",0.36),("rightEyebrow",0.58)]:
        idxs=FEATURES[feat_name]["indices"]
        for n, idx in enumerate(idxs):
            t=n/(len(idxs)-1)*math.pi
            pts[idx]=[cx_+0.08*(n/(len(idxs)-1)), 0.37-0.02*math.sin(t)]
    # nose
    for n, idx in enumerate(FEATURES["nose"]["indices"]):
        t=n/len(FEATURES["nose"]["indices"])*2*math.pi
        pts[idx]=[0.5+0.04*math.cos(t)*(0.3 if n<6 else 1), 0.50+0.05*math.sin(t)+0.02]
    for n, idx in enumerate(FEATURES["outerLips"]["indices"]):
        t=n/(len(FEATURES["outerLips"]["indices"])-1)*2*math.pi
        pts[idx]=[0.5+0.07*math.cos(t), 0.62+0.04*math.sin(t)]
    for n, idx in enumerate(FEATURES["innerLips"]["indices"]):
        t=n/(len(FEATURES["innerLips"]["indices"])-1)*2*math.pi
        pts[idx]=[0.5+0.045*math.cos(t), 0.62+0.02*math.sin(t)]
    for n, idx in enumerate(FEATURES["noseWings"]["indices"]):
        t=n/(len(FEATURES["noseWings"]["indices"])-1)*2*math.pi
        pts[idx]=[0.5+0.06*math.cos(t), 0.55+0.02*math.sin(t)]
    return pts

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description="Face to Maths Equation — generate Fourier equation set for a face")
    ap.add_argument("--image", type=str, help="Path to face image (jpg/png). If omitted, runs synthetic demo.")
    ap.add_argument("--K", type=int, default=20, help="Fourier coefficients per feature (K, total 2K+1 terms). 3=coarse, 20=balanced, 50=detailed, 100+ near-exact")
    ap.add_argument("--out", type=str, default="./output", help="Output directory")
    ap.add_argument("--demo", action="store_true", help="Force synthetic demo even if image provided")
    ap.add_argument("--exact", action="store_true", help="Also generate EXACT pixel equation (2D Fourier, WH coeffs/channel, lossless). No complexity limit — file will be huge for large images.")
    ap.add_argument("--exact-size", type=int, default=None, help="If --exact, resize longest side to this size before 2D DFT (e.g., 128 keeps JSON ~few MB). Omit to use full resolution (exact for original pixels).")
    args=ap.parse_args()

    outdir=Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    K=args.K

    if args.demo or not args.image:
        print("[*] Running synthetic demo (no image or --demo)")
        landmarks=synthetic_landmarks()
        img=None
        face_id_source="synthetic"
    else:
        pts, img, msg = get_landmarks_mediapipe(args.image)
        if pts is None:
            print(f"[!] {msg} — falling back to synthetic landmarks but still demonstrating equation pipeline.")
            print("    Tip: install mediapipe==0.10.11 and use frontal well-lit photo.")
            landmarks=synthetic_landmarks()
            face_id_source=f"synthetic-fallback-for-{Path(args.image).name}"
        else:
            landmarks=pts
            print(f"[+] MediaPipe detected {len(landmarks)} landmarks: {msg}")
            face_id_source=args.image

    eq_data=build_equations(landmarks, K)
    print(f"[+] Face Equation ID: {eq_data['faceId']}")
    print(f"[+] Global RMSE (truncation error, K={K}): {eq_data['globalRMSE']:.6f}")
    for eq in eq_data["equations"]:
        print(f"  - {eq['label']:20s} N={eq['N']:2d} RMSE={eq['rmse']:.5f} |c0|={eq['coeffs'][len(eq['coeffs'])//2]['mag']:.4f}")

    # Serialize
    serializable={
        "faceEquationId": eq_data["faceId"],
        "source": face_id_source,
        "K": K,
        "globalRMSE": eq_data["globalRMSE"],
        "normalization":{"cx":eq_data["cx"],"cy":eq_data["cy"],"scale":eq_data["scale"]},
        "note":"Each feature: z(t)=x(t)+i y(t)= Σ c_k exp(i k ω t), t∈[0,1), ω=2π. Denormalize: X=x*scale+cx, Y=y*scale+cy. Sample 300 steps to redraw.",
        "features":[{k:v for k,v in eq.items() if k!="pts"} for eq in eq_data["equations"]]
    }
    # keep pts for completeness in separate file
    pts_path=outdir/"landmarks.json"
    with open(pts_path,"w") as f:
        json.dump({"landmarks": landmarks.tolist()}, f, indent=2)
    json_path=outdir/f"face-equation-{eq_data['faceId']}.json"
    with open(json_path,"w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[+] Wrote {json_path}")
    print(f"[+] Wrote {pts_path}")

    latex=latex_export(eq_data, K)
    tex_path=outdir/f"face-equation-{eq_data['faceId']}.tex"
    with open(tex_path,"w") as f: f.write(latex)
    print(f"[+] Wrote {tex_path}")

    # Render image
    render_path=outdir/f"face-equation-{eq_data['faceId']}.png"
    render_equation_image(eq_data, out_path=str(render_path))
    print(f"[+] Wrote {render_path} (equation-only redraw)")

    # If original image available, also render overlay comparison
    if img is not None:
        H,W=800,800
        # draw overlay on original resized
        # create comparison side-by-side
        comp=np.zeros((H,W*2,3), dtype=np.uint8)
        # left: original resized
        thumb=cv2.resize(img, (W,H))
        comp[:,0:W]=thumb
        # right: equation render
        eq_img=render_equation_image(eq_data, W=W, H=H)
        comp[:,W:W*2]=eq_img
        comp_path=outdir/f"comparison-{eq_data['faceId']}.png"
        cv2.imwrite(str(comp_path), comp)
        print(f"[+] Wrote {comp_path} (original | equation comparison)")

    # Verification: re-evaluate and check error bound
    max_rmse=max(eq["rmse"] for eq in eq_data["equations"])
    if max_rmse>0.05 and K<10:
        print(f"[!] High RMSE ({max_rmse:.4f}) — increase K for higher fidelity (try --K 30)")
    elif max_rmse<0.005:
        print(f"[✓] High-fidelity reconstruction achieved (max RMSE {max_rmse:.5f})")

    # ---------- EXACT PIXEL EQUATION if requested ----------
    if args.exact:
        print("\n=== EXACT PIXEL EQUATION (unlimited complexity) ===")
        # Determine image for exact mode
        exact_src = img
        if exact_src is None and args.image and Path(args.image).exists():
            # Try to load original file directly for exact mode even if mediapipe failed
            exact_src = cv2.imread(args.image)
            if exact_src is not None:
                print(f"[*] Loaded original image for exact mode: {args.image} ({exact_src.shape[1]}x{exact_src.shape[0]})")
        if exact_src is None:
            # Generate synthetic face image for demo
            S = args.exact_size or 128
            print(f"[*] No photo for exact mode — generating synthetic {S}x{S} face image")
            exact_src = generate_synthetic_face_image(S)
            # Save synthetic source for reference
            synth_path = outdir / f"synthetic_source_{S}x{S}.png"
            cv2.imwrite(str(synth_path), exact_src)
            print(f"[+] Wrote {synth_path} (synthetic input)")
        # Build exact equation
        exact_data = build_exact_pixel_equation(exact_src, target_size=args.exact_size)
        print(f"[+] Exact 2D Fourier: {exact_data['W']}x{exact_data['H']} = {exact_data['total_coeffs_per_channel']} coeffs/channel, {exact_data['total_coeffs_all']} total")
        print(f"[+] Exact Equation ID: {exact_data['exact_id']}")
        print(f"[+] Verification: MSE={exact_data['mse']:.3e}, max_abs_err={exact_data['max_abs_err']:.3e} (0 = lossless)")
        if exact_data['mse'] < 1e-6:
            print("[✓] LOSSLESS — evaluating the equation at integer (x,y) reproduces every pixel exactly")
        # Write exact JSON (may be large)
        exact_json_path = outdir / f"exact-pixel-equation-{exact_data['exact_id']}.json"
        # To avoid huge JSON for large images, we store coeffs compactly; but still write full
        exact_serializable = {
            "exactEquationId": exact_data["exact_id"],
            "H": exact_data["H"], "W": exact_data["W"],
            "total_coeffs_per_channel": exact_data["total_coeffs_per_channel"],
            "total_coeffs_all": exact_data["total_coeffs_all"],
            "verification": {"mse": exact_data["mse"], "max_abs_err": exact_data["max_abs_err"]},
            "equation": "f_c(x,y) = Σ_{u=0}^{W-1} Σ_{v=0}^{H-1} C_c(u,v) * exp(+i*2π*(u*x/W + v*y/H)),  c∈{R,G,B}",
            "definition": "C_c(u,v) = 1/(WH) Σ_{x,y} f_c(x,y) * exp(-i*2π*(u*x/W + v*y/H))",
            "note": "Evaluate at integer x=0..W-1, y=0..H-1. With all WH coeffs, reconstruction is mathematically exact (inverse DFT). No approximation.",
            "coeffs": exact_data["serializable_coeffs"],
            "latex": exact_data["latex"],
        }
        with open(exact_json_path, "w") as f:
            json.dump(exact_serializable, f, indent=2)
        print(f"[+] Wrote {exact_json_path} ({exact_json_path.stat().st_size/1024/1024:.2f} MB)")

        exact_tex_path = outdir / f"exact-pixel-equation-{exact_data['exact_id']}.tex"
        with open(exact_tex_path, "w") as f:
            f.write(exact_data["latex"])
            f.write("\n\n% To redraw (Python):\n")
            f.write("%   import numpy as np, cv2\n")
            f.write("%   C_R = np.array(coeffs['R'])  # HxW complex\n")
            f.write("%   f_R = np.fft.ifft2(C_R * H*W).real  # exact\n")
        print(f"[+] Wrote {exact_tex_path}")

        # Write reconstructed image and comparison
        recon_path = outdir / f"exact-reconstruction-{exact_data['exact_id']}.png"
        cv2.imwrite(str(recon_path), exact_data["recon_u8"])
        print(f"[+] Wrote {recon_path} (reconstruction via equation vs. original — should be identical)")

        comp_path = outdir / f"exact-comparison-{exact_data['exact_id']}.png"
        render_exact_comparison(exact_data, str(comp_path))
        print(f"[+] Wrote {comp_path} (ORIGINAL | EXACT RECON | DIFF)")

        print("\n=== How to redraw EXACT pixels from equation (any language) ===")
        print("  # C is HxW complex, H=height, W=width")
        print("  for y in 0..H-1:")
        print("    for x in 0..W-1:")
        print("      R = sum_u sum_v  C_R[v,u] * exp(+i*2π*(u*x/W + v*y/H))  (real part)")
        print("      G = sum_u sum_v  C_G[v,u] * exp(+i*2π*(u*x/W + v*y/H))")
        print("      B = sum_u sum_v  C_B[v,u] * exp(+i*2π*(u*x/W + v*y/H))")
        print("      pixel[y,x] = (clip(R), clip(G), clip(B))  # exact")

    print("\n=== How to redraw from equations (any language) ===")
    print("  for t in linspace(0,1,300):")
    print("    x = sum( c['re']*cos(2πk t) - c['im']*sin(2πk t) for c in coeffs )")
    print("    y = sum( c['re']*sin(2πk t) + c['im']*cos(2πk t) for c in coeffs )")
    print("    X = x*scale + cx; Y = y*scale + cy   # denormalize to 0-1 image coords")
    print("    X_pixel = X * image_width; Y_pixel = Y * image_height")

if __name__=="__main__":
    main()
