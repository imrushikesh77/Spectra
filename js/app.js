/* Spectra — photo ⇄ equation. Real FFT both ways; equation file holds coefficients only. */

const $ = (id) => document.getElementById(id);

const fileInput = $('fileInput');
const browseBtn = $('browseBtn');
const dropZone = $('dropZone');
const btnDemo = $('btnDemo');
const statusBox = $('statusBox');
const statusText = $('statusText');
const btnReset = $('btnReset');
const tabPhoto = $('tabPhoto');
const tabEq = $('tabEq');
const panelPhoto = $('panelPhoto');
const panelEq = $('panelEq');
const workArea = $('workArea');
const viewOrig = $('viewOrig');
const viewSpec = $('viewSpec');
const viewEq = $('viewEq');
const origDims = $('origDims');
const matchBadge = $('matchBadge');
const eqMeta = $('eqMeta');
const faceIdEl = $('faceId');
const katexBox = $('katexBox');
const coeffTable = $('coeffTable');
const coeffTotalNote = $('coeffTotalNote');
const btnDownload = $('btnDownload');
const btnExport = $('btnExport');
const btnCopy = $('btnCopy');
const kSlider = $('kSlider');
const kVal = $('kVal');
const kMax = $('kMax');
const kMin = $('kMin');
const kMid = $('kMid');
const kFull = $('kFull');
const eqFile = $('eqFile');
const btnLoadFile = $('btnLoadFile');
const btnSample = $('btnSample');
const btnDevelop = $('btnDevelop');
const eqInput = $('eqInput');
const eqStatus = $('eqStatus');
const eqCanvas = $('eqCanvas');
const eqEmpty = $('eqEmpty');
const btnDownloadRebuilt = $('btnDownloadRebuilt');

let S = 128;
let fwd = null;   // {S,N,FR,FG,FB,order,trig,src,id}
let K = 128;

/* ---------------- FFT core (iterative radix-2, power-of-two only) ---------------- */
function fft1d(re, im, invert) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI / len) * (invert ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = re[i + j], ui = im[i + j];
        const vr = re[i + j + len / 2] * cwr - im[i + j + len / 2] * cwi;
        const vi = re[i + j + len / 2] * cwi + im[i + j + len / 2] * cwr;
        re[i + j] = ur + vr; im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
function fft2d(re, im, S, invert) {
  const aR = new Float64Array(S), aI = new Float64Array(S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) { aR[x] = re[y * S + x]; aI[x] = im[y * S + x]; }
    fft1d(aR, aI, invert);
    for (let x = 0; x < S; x++) { re[y * S + x] = aR[x]; im[y * S + x] = aI[x]; }
  }
  for (let x = 0; x < S; x++) {
    for (let y = 0; y < S; y++) { aR[y] = re[y * S + x]; aI[y] = im[y * S + x]; }
    fft1d(aR, aI, invert);
    for (let y = 0; y < S; y++) { re[y * S + x] = aR[y]; im[y * S + x] = aI[y]; }
  }
}
function forwardFFT2D(pixels) {
  const S = Math.sqrt(pixels.length) | 0;
  const re = Float64Array.from(pixels), im = new Float64Array(S * S);
  fft2d(re, im, S, false);
  return { re, im };
}
function radiusOrder(S) {
  const list = [...Array(S * S).keys()];
  const cc = (v) => (v <= S / 2 ? v : v - S);
  list.sort((a, b) => {
    const ax = a % S, ay = (a / S) | 0, bx = b % S, by = (b / S) | 0;
    return (cc(ax) ** 2 + cc(ay) ** 2) - (cc(bx) ** 2 + cc(by) ** 2);
  });
  return Uint32Array.from(list);
}
function buildTrig(S) {
  const cX = new Float64Array(S * S), sX = new Float64Array(S * S);
  const cY = new Float64Array(S * S), sY = new Float64Array(S * S);
  for (let u = 0; u < S; u++) for (let x = 0; x < S; x++) {
    const a = 2 * Math.PI * u * x / S;
    cX[u * S + x] = Math.cos(a); sX[u * S + x] = Math.sin(a);
  }
  for (let v = 0; v < S; v++) for (let y = 0; y < S; y++) {
    const a = 2 * Math.PI * v * y / S;
    cY[v * S + y] = Math.cos(a); sY[v * S + y] = Math.sin(a);
  }
  return { cX, sX, cY, sY };
}
/* f_K(x,y) = (1/N) Σ_{k<K} Re[ F_k · e^{+i2π(ux/W+vy/H)} ] — genuine partial evaluation */
function synthesize(FR, FG, FB, order, trig, S, K) {
  const N = S * S;
  const { cX, sX, cY, sY } = trig;
  const outR = new Float64Array(N), outG = new Float64Array(N), outB = new Float64Array(N);
  const inv = 1 / N;
  for (let k = 0; k < K; k++) {
    const bi = order[k];
    const u = bi % S, v = (bi / S) | 0;
    const rr = FR.re[bi], ri = FR.im[bi];
    const gr = FG.re[bi], gi = FG.im[bi];
    const br = FB.re[bi], bi2 = FB.im[bi];
    for (let y = 0; y < S; y++) {
      const cy = cY[v * S + y], sy = sY[v * S + y];
      for (let x = 0; x < S; x++) {
        const cxp = cX[u * S + x], sxp = sX[u * S + x];
        const cosT = cxp * cy - sxp * sy;
        const sinT = sxp * cy + cxp * sy;
        const p = y * S + x;
        outR[p] += rr * cosT - ri * sinT;
        outG[p] += gr * cosT - gi * sinT;
        outB[p] += br * cosT - bi2 * sinT;
      }
    }
  }
  for (let i = 0; i < N; i++) { outR[i] *= inv; outG[i] *= inv; outB[i] *= inv; }
  return { outR, outG, outB };
}
function inverseFull(FR, FG, FB, S) {
  const mk = (F) => {
    const re = Float64Array.from(F.re), im = Float64Array.from(F.im);
    fft2d(re, im, S, true);
    return re;
  };
  return { outR: mk(FR), outG: mk(FG), outB: mk(FB) };
}

/* ---------------- helpers ---------------- */
function setStatus(msg, mode = 'idle') {
  statusText.textContent = msg;
  statusBox.className = 'status ' + mode;
}
function simpleHash(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(16).padStart(8, '0');
}
async function makeId(S, FR) {
  let s = S + ':';
  for (let i = 0; i < Math.min(256, FR.re.length); i++) s += FR.re[i].toFixed(2) + ',';
  s += FR.re.length;
  try {
    if (crypto?.subtle) {
      const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
      return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 12).toUpperCase();
    }
  } catch {}
  return (simpleHash(s) + simpleHash([...s].reverse().join(''))).slice(0, 12).toUpperCase();
}
function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !file.type.startsWith('image/')) return reject(new Error('Not an image — try JPG/PNG/WEBP'));
    if (file.size > 12 * 1024 * 1024) return reject(new Error('Image too large (>12MB)'));
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => reject(new Error('Could not read image — convert HEIC to JPG first'));
    img.src = url;
  });
}
function rasterizeSquare(img, S) {
  const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
  const side = Math.min(iw, ih);
  const c = document.createElement('canvas');
  c.width = S; c.height = S;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, (iw - side) / 2, (ih - side) / 2, side, side, 0, 0, S, S);
  return c;
}
function channelsOf(canvas) {
  const S = canvas.width;
  const d = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, S, S).data;
  const N = S * S;
  const R = new Float64Array(N), G = new Float64Array(N), B = new Float64Array(N);
  for (let i = 0; i < N; i++) { R[i] = d[i * 4]; G[i] = d[i * 4 + 1]; B[i] = d[i * 4 + 2]; }
  return { R, G, B };
}
function paintChannels(canvas, R, G, B) {
  const S = Math.sqrt(R.length) | 0;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    img.data[i * 4] = Math.max(0, Math.min(255, Math.round(R[i])));
    img.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(G[i])));
    img.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(B[i])));
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function mseOf(aR, aG, aB, bR, bG, bB) {
  let se = 0;
  for (let i = 0; i < aR.length; i++)
    se += (aR[i] - bR[i]) ** 2 + (aG[i] - bG[i]) ** 2 + (aB[i] - bB[i]) ** 2;
  return se / (aR.length * 3);
}
function f64ToB64(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
}
function b64ToF64(b64, n) {
  const bin = atob(b64.replace(/\s+/g, ''));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (bytes.length < n * 8) throw new Error('Truncated coefficient data');
  return new Float64Array(bytes.buffer, 0, n);
}
async function copyText(t) {
  try { await navigator.clipboard.writeText(t); return true; }
  catch {
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); return true; } catch { return false; }
    finally { ta.remove(); }
  }
}
function renderSpectrum(canvas, FR, FG, FB, S) {
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(S, S);
  const mags = new Float64Array(S * S);
  let mx = 0;
  for (let v = 0; v < S; v++) for (let u = 0; u < S; u++) {
    const i = v * S + u;
    const m = Math.log1p(Math.hypot(
      0.299 * FR.re[i] + 0.587 * FG.re[i] + 0.114 * FB.re[i],
      0.299 * FR.im[i] + 0.587 * FG.im[i] + 0.114 * FB.im[i]));
    mags[i] = m; if (m > mx) mx = m;
  }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const sx = (x + S / 2) % S, sy = (y + S / 2) % S;
    const t = Math.max(0, Math.min(1, mags[sy * S + sx] / (mx || 1)));
    const p = (y * S + x) * 4;
    const g = Math.round(t * 255);
    img.data[p] = g; img.data[p + 1] = g; img.data[p + 2] = Math.round(200 + t * 55); img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/* ---------------- forward flow ---------------- */
function renderCoeffTable() {
  if (!fwd) return;
  const { S, FR, FG, FB } = fwd, N = S * S;
  const mag = (F, i) => Math.hypot(F.re[i], F.im[i]);
  const idx = [...Array(N).keys()];
  idx.sort((a, b) => (mag(FR, b) + mag(FG, b) + mag(FB, b)) - (mag(FR, a) + mag(FG, a) + mag(FB, a)));
  const lines = [];
  for (let r = 0; r < Math.min(10, N); r++) {
    const i = idx[r], u = i % S, v = (i / S) | 0;
    lines.push(`#${String(r + 1).padStart(2)} (u=${String(u).padStart(3)},v=${String(v).padStart(3)}) `
      + `|R|=${mag(FR, i).toFixed(1).padStart(8)} |G|=${mag(FG, i).toFixed(1).padStart(8)} |B|=${mag(FB, i).toFixed(1).padStart(8)}`);
  }
  lines.push(`… ${(N * 3).toLocaleString()} numbers total — all in the .json`);
  coeffTable.textContent = lines.join('\n');
}

let synthTimer = null;
function rebuildAtK() {
  if (!fwd) return;
  const { S, N, FR, FG, FB, order, trig } = fwd;
  const full = K >= N;
  setStatus(full ? 'Evaluating full inverse FFT…' : `Summing ${K} frequencies…`, 'loading');
  clearTimeout(synthTimer);
  synthTimer = setTimeout(() => {
    const { outR, outG, outB } = full ? inverseFull(FR, FG, FB, S) : synthesize(FR, FG, FB, order, trig, S, K);
    paintChannels(viewEq, outR, outG, outB);
    let label, exact;
    if (fwd.src) {
      const mse = mseOf(fwd.src.R, fwd.src.G, fwd.src.B, outR, outG, outB);
      exact = mse < 0.5;
      label = exact ? `Exact · MSE ${mse.toExponential(1)}` : `MSE ${mse < 10 ? mse.toFixed(1) : Math.round(mse)}`;
    } else {
      const chk = forwardFFT2D(outR), chkG = forwardFFT2D(outG), chkB = forwardFFT2D(outB);
      let se = 0;
      for (const [A, B] of [[chk.re, FR.re], [chk.im, FR.im], [chkG.re, FG.re], [chkG.im, FG.im], [chkB.re, FB.re], [chkB.im, FB.im]])
        for (let i = 0; i < A.length; i++) se += (A[i] - B[i]) ** 2;
      const rel = se / (6 * N);
      exact = full && rel < 1e-6;
      label = full ? `ΔF ${rel.toExponential(1)} · exact` : `K=${K}/${N}`;
    }
    matchBadge.textContent = label;
    kVal.textContent = full ? 'FULL' : String(K);
    setStatus(exact ? `Exact — ${label}` : `Partial sum — push K to FULL`, exact ? 'ready' : 'idle');
  }, full ? 30 : 90);
}

async function adoptForward(S, FR, FG, FB, src, mode) {
  const N = S * S;
  const id = await makeId(S, FR);
  fwd = { S, N, FR, FG, FB, order: radiusOrder(S), trig: buildTrig(S), src, id, mode };
  if (src) {
    const tmp = document.createElement('canvas');
    paintChannels(tmp, src.R, src.G, src.B);
    drawToView(viewOrig, tmp);
    origDims.textContent = `${S}×${S}`;
  }
  renderSpectrum(viewSpec, FR, FG, FB, S);
  kSlider.max = N;
  kMax.textContent = N.toLocaleString();
  K = Math.min(S >= 128 ? 400 : 200, N);
  kSlider.value = K;
  rebuildAtK();
  const total = (N * 6).toLocaleString();
  eqMeta.textContent = `${S} × ${S} · ${total} real numbers · ${mode}`;
  coeffTotalNote.textContent = `${total} numbers`;
  faceIdEl.textContent = id;
  try {
    katex.render(
      `f(x,y)=\\frac{1}{${S * S}}\\sum_{u=0}^{${S - 1}}\\sum_{v=0}^{${S - 1}}F(u,v)e^{+i2\\pi(ux+vy)/${S}}`,
      katexBox, { throwOnError: false, displayMode: true });
  } catch {
    katexBox.textContent = `f(x,y) = (1/${S * S}) Σ F(u,v)·e^(+i2π(ux+vy)/${S})`;
  }
  renderCoeffTable();
  workArea.classList.remove('hidden');
}
function drawToView(view, srcCanvas) {
  view.width = srcCanvas.width;
  view.height = srcCanvas.height;
  const ctx = view.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(srcCanvas, 0, 0);
}

async function handleFile(file) {
  if (!file) return;
  setStatus('Reading photo…', 'loading');
  try {
    const img = await loadImageFile(file);
    setStatus(`Forward FFT on ${S}×${S}…`, 'loading');
    await new Promise(r => setTimeout(r, 30));
    const bmp = rasterizeSquare(img, S);
    const src = channelsOf(bmp);
    await adoptForward(S, forwardFFT2D(src.R), forwardFFT2D(src.G), forwardFFT2D(src.B), src, 'photo');
    setStatus(`Done — ${S}×${S} chord computed. Push K to FULL.`, 'ready');
  } catch (e) {
    console.error(e);
    setStatus(e.message || 'Could not process photo', 'error');
  }
}

function equationJSON() {
  if (!fwd) return null;
  const enc = (F) => ({ re_b64: f64ToB64(F.re), im_b64: f64ToB64(F.im) });
  return {
    app: 'spectra-patchbay', version: 3, S: fwd.S, id: fwd.id,
    form: 'f(x,y) = (1/N) Σ_u Σ_v F(u,v)·exp(+i·2π(ux+vy)/S), F = FFT2(f)',
    channels: { R: enc(fwd.FR), G: enc(fwd.FG), B: enc(fwd.FB) },
  };
}

/* ---------------- export plate (theorem page, real values) ---------------- */
function escXml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
async function exportCard() {
  if (!fwd) { setStatus('Upload a photo first', 'error'); return; }
  setStatus('Typesetting plate…', 'loading');
  try { await document.fonts.ready; } catch {}
  try {
    const { S, id, FR, FG, FB } = fwd;
    const N = S * S;
    const mag = (i) => Math.hypot(FR.re[i], FR.im[i]) + Math.hypot(FG.re[i], FG.im[i]) + Math.hypot(FB.re[i], FB.im[i]);
    const idx = [...Array(N).keys()].sort((a, b) => mag(b) - mag(a)).slice(0, 6);
    const f2 = (x) => (x >= 0 ? '+' : '') + x.toFixed(1);
    const rows = idx.map((i, r) => {
      const u = i % S, v = (i / S) | 0;
      return `#${r + 1} (u=${u},v=${v}) R=${FR.re[i].toFixed(1)}${f2(FR.im[i])}i G=${FG.re[i].toFixed(1)}${f2(FG.im[i])}i B=${FB.re[i].toFixed(1)}${f2(FB.im[i])}i`;
    });
    const rebuiltURL = viewEq.toDataURL('image/png');
    const specURL = viewSpec.toDataURL('image/png');
    const EW = 1600, imgW = 560, imgH = 560;
    const EH = 340 + imgH + 640, cx = EW / 2;
    const svg =
`<svg xmlns="http://www.w3.org/2000/svg" width="${EW}" height="${EH}" viewBox="0 0 ${EW} ${EH}">
<rect width="${EW}" height="${EH}" fill="#FFFDF6"/>
<rect x="22" y="22" width="${EW - 44}" height="${EH - 44}" fill="none" stroke="#141210" stroke-width="4"/>
<text x="${cx}" y="88" text-anchor="middle" font-family="Georgia,serif" font-size="30" letter-spacing="10" fill="#141210">SPECTRA · VISHVA SUTRA</text>
<text x="${cx}" y="126" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="25" fill="#6A6252">real coefficients · ${S}×${S} · ID ${escXml(id)}</text>
<line x1="120" y1="154" x2="${EW - 120}" y2="154" stroke="#141210" stroke-width="1.5"/>
<image href="${rebuiltURL}" x="${cx - imgW - 24}" y="190" width="${imgW}" height="${imgH}" preserveAspectRatio="xMidYMid meet"/>
<image href="${specURL}" x="${cx + 24}" y="190" width="${imgW}" height="${imgH}" preserveAspectRatio="xMidYMid meet"/>
<text x="${cx - imgW / 2 - 24}" y="${190 + imgH + 40}" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="24" fill="#4A4234">Fig.1 — rebuilt from all ${N} frequencies, MSE 0</text>
<text x="${cx + imgW / 2 + 24}" y="${190 + imgH + 40}" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="24" fill="#4A4234">Fig.2 — spectrum |C|, log scale</text>
<text x="120" y="${190 + imgH + 120}" font-family="Georgia,serif" font-weight="bold" font-size="36" fill="#141210">Theorem 1 <tspan font-weight="normal" font-style="italic">(picture = chord).</tspan></text>
<text x="${cx - 620}" y="${190 + imgH + 250}" font-family="Georgia,serif" font-size="54">
<tspan font-style="italic" font-weight="bold">f</tspan><tspan baseline-shift="sub" font-size="30" font-style="italic">c</tspan><tspan font-style="italic" font-size="48">(x,y) = (1/${S * S})&#8721;</tspan><tspan baseline-shift="sub" font-size="22">u,v</tspan><tspan font-style="italic" font-size="48"> F</tspan><tspan baseline-shift="sub" font-size="28" font-style="italic">c</tspan><tspan font-style="italic" font-size="44">(u,v)e</tspan><tspan baseline-shift="super" font-size="25" font-style="italic">+i2&#960;(ux+vy)/${S}</tspan>
</text>
<text x="${EW - 120}" y="${190 + imgH + 250}" text-anchor="middle" font-family="Georgia,serif" font-size="32">(1)</text>
${rows.map((r, i) => `<text x="140" y="${190 + imgH + 310 + i * 34}" font-family="monospace" font-size="21" fill="#3A3428">${escXml(r)}</text>`).join('')}
<text x="140" y="${190 + imgH + 310 + 6 * 34}" font-family="Georgia,serif" font-style="italic" font-size="23" fill="#6A6252">… all ${(N * 3).toLocaleString()} numbers in the .json · evaluate (1) at integer x,y.</text>
<text x="${cx}" y="${EH - 48}" text-anchor="middle" font-family="monospace" font-size="19" fill="#A39C8B">100% on-device · Spectra · ID ${escXml(id)}</text>
</svg>`;
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
    const c = document.createElement('canvas');
    c.width = EW; c.height = EH;
    const x2 = c.getContext('2d');
    x2.fillStyle = '#FFFDF6'; x2.fillRect(0, 0, EW, EH);
    x2.drawImage(img, 0, 0, EW, EH);
    URL.revokeObjectURL(url);
    const out = await new Promise((res) => c.toBlob(res, 'image/png'));
    if (!out) throw new Error('raster failed');
    const durl = URL.createObjectURL(out);
    const a = document.createElement('a');
    a.href = durl; a.download = `spectra-${id}.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(durl), 4000);
    setStatus('Plate exported ✓', 'ready');
  } catch (e) {
    console.error(e);
    setStatus('Export failed — try Download .json instead', 'error');
  }
}

/* ---------------- reverse: equation → photo ---------------- */
async function coerceToCoeffs(raw) {
  const t = raw.trim();
  if (!t) throw new Error('Empty equation');
  try {
    const j = JSON.parse(t);
    if (j && j.channels && (j.app === 'spectra-patchbay' || j.app === 'spectra-chitra-sutra')) {
      const Sq = j.S, N = Sq * Sq;
      if (![32, 64, 128, 256].includes(Sq)) throw new Error(`Unsupported size ${Sq}`);
      const dec = (c) => ({ re: b64ToF64(c.re_b64, N), im: b64ToF64(c.im_b64, N) });
      return { S: Sq, FR: dec(j.channels.R), FG: dec(j.channels.G), FB: dec(j.channels.B), label: 'equation JSON ' + (j.id || '') };
    }
  } catch (e) { if (String(e.message).startsWith('Unsupported')) throw e; }
  const m = t.match(/CHITRA-SUTRA-v1\s*:\s*W\s*=\s*(\d+)\s*:\s*H\s*=\s*(\d+)\s*:\s*PNG\s*:\s*([A-Za-z0-9+/=_\-\s]+)/)
    || t.match(/data:image\/png;base64,([A-Za-z0-9+/=\s]+)/);
  if (m) {
    const b64 = (m[3] || m[1]).replace(/\s+/g, '').replace(/-/g, '+').replace(/_/g, '/');
    const img = await new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im); im.onerror = () => rej(new Error('Legacy PNG payload would not decode'));
      im.src = 'data:image/png;base64,' + b64;
    });
    const Sq = [32, 64, 128, 256].reduce((a, b) => Math.abs(b - Math.max(img.naturalWidth, img.naturalHeight)) < Math.abs(a - Math.max(img.naturalWidth, img.naturalHeight)) ? b : a);
    const bmp = rasterizeSquare(img, Sq);
    const src = channelsOf(bmp);
    return { S: Sq, FR: forwardFFT2D(src.R), FG: forwardFFT2D(src.G), FB: forwardFFT2D(src.B), label: 'legacy PNG re-analysed' };
  }
  throw new Error('Unrecognised — drop the .json equation file.');
}

async function developFromInput(fillSample = false) {
  if (fillSample && !eqInput.value.trim()) {
    const Sq = 64, N = Sq * Sq;
    const R = new Float64Array(N), G = new Float64Array(N), B = new Float64Array(N);
    for (let y = 0; y < Sq; y++) for (let x = 0; x < Sq; x++) {
      const i = y * Sq + x;
      R[i] = 128 + 127 * Math.sin(2 * Math.PI * 3 * x / Sq) * Math.cos(2 * Math.PI * 2 * y / Sq);
      G[i] = 128 + 127 * Math.sin(2 * Math.PI * 5 * y / Sq);
      B[i] = 200 - (x / Sq) * 120;
    }
    const enc = (F) => ({ re_b64: f64ToB64(F.re), im_b64: f64ToB64(F.im) });
    const FR = forwardFFT2D(R), FG = forwardFFT2D(G), FB = forwardFFT2D(B);
    eqInput.value = JSON.stringify({ app: 'spectra-patchbay', version: 3, S: Sq, id: 'SAMPLE', form: 'sample chord', channels: { R: enc(FR), G: enc(FG), B: enc(FB) } });
  }
  eqStatus.textContent = 'Developing — inverse FFT…';
  btnDownloadRebuilt.disabled = true;
  try {
    const { S: Sq, FR, FG, FB, label } = await coerceToCoeffs(eqInput.value);
    const { outR, outG, outB } = inverseFull(FR, FG, FB, Sq);
    paintChannels(eqCanvas, outR, outG, outB);
    eqEmpty.style.display = 'none';
    eqStatus.textContent = `Developed ${Sq}×${Sq} · inverse FFT of ${label} ✓`;
    btnDownloadRebuilt.disabled = false;
    btnDownloadRebuilt.onclick = () => {
      const a = document.createElement('a');
      a.href = eqCanvas.toDataURL('image/png');
      a.download = 'rebuilt-from-equation.png';
      a.click();
    };
  } catch (e) { eqStatus.textContent = e.message; }
}

/* ---------------- wiring ---------------- */
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
dropZone.addEventListener('click', (e) => {
  if (e.target.closest('button')) return;
  fileInput.click();
});
['dragover', 'dragenter'].forEach(ev => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files?.[0];
  if (!f) return;
  if (/\.(json|txt)$/i.test(f.name)) { setStatus('Equation files go in Equation → Photo', 'error'); return; }
  handleFile(f);
});
document.querySelectorAll('.sizebtn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.sizebtn').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  S = parseInt(b.dataset.s, 10);
  setStatus(`Resolution ${S}×${S} — drop a photo`, 'idle');
}));
let kTimer = null;
kSlider.addEventListener('input', () => {
  if (!fwd) return;
  K = Math.max(1, Math.min(fwd.N, parseInt(kSlider.value, 10) || 1));
  clearTimeout(kTimer);
  kTimer = setTimeout(rebuildAtK, 90);
});
kMin.addEventListener('click', () => { if (!fwd) return; K = 1; kSlider.value = 1; rebuildAtK(); });
kMid.addEventListener('click', () => { if (!fwd) return; K = Math.min(200, fwd.N); kSlider.value = K; rebuildAtK(); });
kFull.addEventListener('click', () => { if (!fwd) return; K = fwd.N; kSlider.value = K; rebuildAtK(); });
btnDownload.addEventListener('click', () => {
  const j = equationJSON();
  if (!j) return setStatus('Upload a photo first', 'error');
  const txt = JSON.stringify(j);
  const blob = new Blob([txt], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `spectra-equation-${j.id}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  setStatus(`Downloaded — ${(txt.length / 1024).toFixed(0)}KB of real coefficients`, 'ready');
});
btnExport.addEventListener('click', exportCard);
btnCopy.addEventListener('click', async () => {
  if (!fwd) return setStatus('Upload a photo first', 'error');
  const latex = `\\begin{equation}\nf(x,y)=\\frac{1}{${fwd.S * fwd.S}}\\sum_{u=0}^{${fwd.S - 1}}\\sum_{v=0}^{${fwd.S - 1}}F(u,v)e^{+i2\\pi(ux+vy)/${fwd.S}} \\quad \\text{(${((fwd.N * 6)).toLocaleString()} numbers in .json, ID ${fwd.id})}\n\\end{equation}`;
  const ok = await copyText(latex);
  btnCopy.textContent = ok ? '✓ Formula copied' : 'Copy failed';
  setStatus(ok ? 'Display formula copied — rebuild needs the .json' : 'Copy failed', ok ? 'ready' : 'error');
  setTimeout(() => (btnCopy.textContent = 'Copy formula'), 2000);
});
btnReset.addEventListener('click', () => {
  fwd = null;
  fileInput.value = '';
  workArea.classList.add('hidden');
  for (const cv of [viewOrig, viewSpec, viewEq]) { try { cv.width = 2; cv.height = 2; } catch {} }
  matchBadge.textContent = '—';
  eqMeta.textContent = '—';
  faceIdEl.textContent = '—';
  katexBox.innerHTML = '';
  coeffTable.textContent = '—';
  btnCopy.textContent = 'Copy formula';
  setStatus('Waiting for a photo…', 'idle');
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
tabPhoto.addEventListener('click', () => {
  tabPhoto.classList.add('active'); tabEq.classList.remove('active');
  panelPhoto.classList.remove('hidden'); panelEq.classList.add('hidden');
});
tabEq.addEventListener('click', () => {
  tabEq.classList.add('active'); tabPhoto.classList.remove('active');
  panelEq.classList.remove('hidden'); panelPhoto.classList.add('hidden');
});
btnLoadFile.addEventListener('click', () => eqFile.click());
eqFile.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    eqInput.value = await f.text();
    eqStatus.textContent = `Loaded ${f.name} (${(f.size / 1024).toFixed(0)}KB) — press Develop`;
  } catch { eqStatus.textContent = 'Could not read file'; }
});
btnDevelop.addEventListener('click', () => developFromInput(false));
btnSample.addEventListener('click', () => developFromInput(true));

setStatus('Waiting for a photo…', 'idle');
