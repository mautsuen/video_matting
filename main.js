// main.js
import * as ort from "https://unpkg.com/onnxruntime-web@1.23.2/dist/ort.all.min.mjs";

const $ = (id) => document.getElementById(id);

const startBtn = $("start");
const stopBtn  = $("stop");
const video    = $("video");
const out      = $("out");
const statusEl = $("status");

// Your UI (some may be null depending on your HTML)
const drawFpsSel  = $("fps");        // "Draw FPS"
const inferFpsSel = $("inferFps");   // "Infer FPS"
const inferSizeSel= $("inferSize");  // "Infer short edge"
const bgSel       = $("bg");

// --------- Tuning ----------
const MIRROR = true; // if your <video> uses CSS scaleX(-1), keep this true so output matches
const FORCE_LANDSCAPE_AR_ON_PORTRAIT = true;
// This matches your Python pipeline's final tensor aspect (672x512)
const TARGET_AR = 672 / 512;

// Matte sanity thresholds (avoid "suddenly all white")
const MIN_MEAN_ALPHA = 0.02;  // too small -> basically empty matte
const MIN_MAX_ALPHA  = 0.10;  // max too small -> basically empty matte
const BAD_STREAK_TO_FALLBACK = 3;

// ORT paths
const WASM_BASE = "./wasm/"; // must contain ort-wasm*.wasm/.mjs files (your local setup)
const MODEL_DIR = "./models/Xenova/modnet/onnx/";
const MODEL_FP16 = MODEL_DIR + "model_fp16.onnx";
const MODEL_FP32 = MODEL_DIR + "model.onnx";

// --------- State ----------
let running = false;
let stream = null;

let session = null;
let inputName = "input";
let outputName = "output";
let usingEP = "none"; // "webgpu/fp16" | "webgpu/fp32" | "wasm/fp32"

const outCtx = out.getContext("2d", { willReadFrequently: false });

const inferCanvas = document.createElement("canvas");
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

const lastGoodMask = document.createElement("canvas");
const lastGoodCtx = lastGoodMask.getContext("2d", { willReadFrequently: false });

let matteReady = false;
let badMatteStreak = 0;
let lastInferTS = 0;

function logStatus(msg) {
  statusEl.textContent = msg;
  console.log(msg);
}

function isPortraitVideo() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  return vw > 0 && vh > 0 && vh > vw;
}

function getInferFPS() {
  // You asked draw fps == infer fps
  const v = inferFpsSel?.value ?? drawFpsSel?.value ?? "10";
  return Math.max(1, parseInt(v, 10) || 10);
}

function syncFPS() {
  if (!drawFpsSel || !inferFpsSel) return;
  // keep them identical
  const setBoth = (val) => {
    drawFpsSel.value = val;
    inferFpsSel.value = val;
  };
  setBoth(inferFpsSel.value || drawFpsSel.value || "10");

  inferFpsSel.addEventListener("change", () => setBoth(inferFpsSel.value));
  drawFpsSel.addEventListener("change", () => setBoth(drawFpsSel.value));
}

function getInferShortEdge() {
  const v = inferSizeSel?.value ?? "512";
  return Math.max(128, parseInt(v, 10) || 512);
}

function getBGMode() {
  return bgSel?.value ?? "white";
}

function setCanvasSizes() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;

  const outW = 480;

  let outH;
  if (FORCE_LANDSCAPE_AR_ON_PORTRAIT && vh > vw) {
    // portrait camera -> force a landscape-ish crop so face stays "normal size" for MODNet
    outH = Math.round(outW / TARGET_AR);
  } else {
    outH = Math.round(outW * (vh / vw));
  }

  out.width = outW;
  out.height = outH;

  // Make video element show the SAME crop region (so preview matches output)
  video.style.width = outW + "px";
  video.style.height = outH + "px";
  video.style.objectFit = "cover";
  if (MIRROR) {
    video.style.transform = "scaleX(-1)";
  } else {
    video.style.transform = "";
  }

  // Infer tensor size keeps aspect ratio of out canvas
  const shortEdge = getInferShortEdge();
  let inferW, inferH;
  if (outW >= outH) {
    inferH = shortEdge;
    inferW = Math.round(shortEdge * (outW / outH));
  } else {
    inferW = shortEdge;
    inferH = Math.round(shortEdge * (outH / outW));
  }

  // multiples of 32 helps MODNet/conv speed and some backends
  inferW = Math.max(32, Math.round(inferW / 32) * 32);
  inferH = Math.max(32, Math.round(inferH / 32) * 32);

  inferCanvas.width = inferW;
  inferCanvas.height = inferH;

  maskCanvas.width = inferW;
  maskCanvas.height = inferH;

  lastGoodMask.width = inferW;
  lastGoodMask.height = inferH;

  console.log(`out=${outW}x${outH}, infer=${inferW}x${inferH}, video=${vw}x${vh}`);
}

function drawVideoCover(ctx, w, h, mirror) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const sAR = vw / vh;
  const dAR = w / h;

  let sw, sh, sx, sy;
  if (sAR > dAR) {
    // source wider -> crop left/right
    sh = vh;
    sw = Math.round(vh * dAR);
    sx = Math.round((vw - sw) / 2);
    sy = 0;
  } else {
    // source taller -> crop top/bottom
    sw = vw;
    sh = Math.round(vw / dAR);
    sx = 0;
    sy = Math.round((vh - sh) / 2);
  }

  ctx.save();
  if (mirror) {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h);
  ctx.restore();
}

function buildInputTensorFloat32() {
  const { width: W, height: H } = inferCanvas;
  const img = inferCtx.getImageData(0, 0, W, H).data;

  // NCHW float32 normalized to [-1, 1]
  const data = new Float32Array(1 * 3 * H * W);
  const HW = H * W;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = (y * W + x);
      const i = p * 4;

      const r = img[i] / 255;
      const g = img[i + 1] / 255;
      const b = img[i + 2] / 255;

      data[p]         = (r - 0.5) / 0.5;
      data[p + HW]    = (g - 0.5) / 0.5;
      data[p + 2*HW]  = (b - 0.5) / 0.5;
    }
  }

  return new ort.Tensor("float32", data, [1, 3, H, W]);
}

function parseOutputToAlpha(outTensor) {
  const dims = outTensor.dims;
  const data = outTensor.data; // typed array

  // support NCHW or NHWC
  let H, W, layout;
  if (dims.length === 4) {
    const [N, d1, d2, d3] = dims;
    if (d1 <= 4 && d2 > 4 && d3 > 4) {
      layout = "NCHW";
      H = d2; W = d3;
    } else if (d3 <= 4 && d1 > 4 && d2 > 4) {
      layout = "NHWC";
      H = d1; W = d2;
    } else {
      // default assume NCHW
      layout = "NCHW";
      H = d2; W = d3;
    }
  } else if (dims.length === 3) {
    // [1, H, W]
    layout = "HW";
    H = dims[1]; W = dims[2];
  } else {
    throw new Error("Unexpected output dims: " + JSON.stringify(dims));
  }

  const alpha = new Float32Array(H * W);

  // sample a bunch to decide if we need sigmoid (usually MODNet matte is already 0..1)
  let outOf01 = 0;
  const S = 2048;
  for (let s = 0; s < S; s++) {
    const idx = (s * 9973) % (H * W);
    let v;
    if (layout === "NCHW") v = data[idx];               // c=0
    else if (layout === "NHWC") v = data[idx * dims[3]];// c=0
    else v = data[idx];

    if (!Number.isFinite(v) || v < -0.2 || v > 1.2) outOf01++;
  }
  const needSigmoid = (outOf01 / S) > 0.25;

  for (let i = 0; i < H * W; i++) {
    let v;
    if (layout === "NCHW") v = data[i];                // c=0
    else if (layout === "NHWC") v = data[i * dims[3]]; // c=0
    else v = data[i];

    if (!Number.isFinite(v)) v = 0;

    if (needSigmoid) v = 1 / (1 + Math.exp(-v));
    // clamp
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    alpha[i] = v;
  }

  return { alpha, H, W, needSigmoid };
}

function maybeInvertAlpha(alpha, W, H) {
  // decide invert by comparing center vs border
  let center = 0, border = 0, cN = 0, bN = 0;

  const cx0 = Math.floor(W * 0.35), cx1 = Math.floor(W * 0.65);
  const cy0 = Math.floor(H * 0.35), cy1 = Math.floor(H * 0.65);

  for (let y = cy0; y < cy1; y += 2) {
    for (let x = cx0; x < cx1; x += 2) {
      center += alpha[y * W + x];
      cN++;
    }
  }

  // border ring
  const ring = Math.max(8, Math.floor(Math.min(W, H) * 0.12));
  for (let y = 0; y < H; y += 3) {
    for (let x = 0; x < W; x += 3) {
      const isBorder = (x < ring || x >= W - ring || y < ring || y >= H - ring);
      if (!isBorder) continue;
      border += alpha[y * W + x];
      bN++;
    }
  }

  const cMean = cN ? center / cN : 0.5;
  const bMean = bN ? border / bN : 0.5;

  // if center is lower than border, likely inverted
  const invert = cMean < bMean;

  if (invert) {
    for (let i = 0; i < alpha.length; i++) alpha[i] = 1 - alpha[i];
  }

  return { invert, cMean, bMean };
}

function alphaStats(alpha) {
  // light sampling for speed
  const n = alpha.length;
  const S = Math.min(4096, n);
  let min = 1, max = 0, sum = 0;
  for (let s = 0; s < S; s++) {
    const i = (s * 7919) % n;
    const v = alpha[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  return { min, max, mean: sum / S };
}

function updateMaskCanvas(alpha, W, H) {
  // write RGBA mask (alpha channel)
  const img = maskCtx.createImageData(W, H);
  const d = img.data;

  for (let i = 0; i < W * H; i++) {
    const a = Math.round(alpha[i] * 255);
    d[i*4 + 0] = 0;
    d[i*4 + 1] = 0;
    d[i*4 + 2] = 0;
    d[i*4 + 3] = a;
  }
  maskCtx.putImageData(img, 0, 0);
}

function drawComposite() {
  const w = out.width, h = out.height;
  if (!w || !h) return;

  // background
  const bg = getBGMode();
  outCtx.save();
  outCtx.globalCompositeOperation = "source-over";
  outCtx.filter = "none";

  if (bg === "green") {
    outCtx.fillStyle = "#00ff00";
    outCtx.fillRect(0, 0, w, h);
  } else if (bg === "blur") {
    outCtx.filter = "blur(10px)";
    drawVideoCover(outCtx, w, h, MIRROR);
    outCtx.filter = "none";
  } else {
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, w, h);
  }

  // foreground video
  drawVideoCover(outCtx, w, h, MIRROR);

  // apply matte if ready
  if (matteReady) {
    outCtx.globalCompositeOperation = "destination-in";
    // scale mask to output size
    outCtx.drawImage(maskCanvas, 0, 0, w, h);
    outCtx.globalCompositeOperation = "source-over";
  }

  outCtx.restore();
}

async function initSessionWebGPU(modelPath) {
  // WebGPU EP
  // Note: Some devices have unstable/incorrect output on WebGPU; we will sanity-check & fallback.
  const sess = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["webgpu"],
  });
  return sess;
}

async function initSessionWASM(modelPath) {
  const hc = navigator.hardwareConcurrency || 4;

  ort.env.wasm.wasmPaths = WASM_BASE;
  ort.env.wasm.numThreads = Math.max(1, Math.min(8, hc));
  ort.env.wasm.simd = true;

  const sess = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["wasm"],
  });
  return sess;
}

async function initModel() {
  logStatus(
    `Loading model...\nsecureContext=${window.isSecureContext}\n` +
    `crossOriginIsolated=${window.crossOriginIsolated}\n` +
    `navigator.gpu=${!!navigator.gpu}\n`
  );

  // Always set wasm paths early (even WebGPU EP may internally reference wasm pieces on fallback paths)
  ort.env.wasm.wasmPaths = WASM_BASE;

  // Try WebGPU fp16 -> WebGPU fp32 -> WASM fp32
  // (fp16 often fails on many Android devices)
  if (navigator.gpu) {
    try {
      logStatus(statusEl.textContent + "\nLoading model… try WebGPU (fp16)…");
      session = await initSessionWebGPU(MODEL_FP16);
      usingEP = "webgpu/fp16";
    } catch (e1) {
      console.warn("WebGPU fp16 failed:", e1);
      try {
        logStatus(statusEl.textContent + "\nLoading model… try WebGPU (fp32)…");
        session = await initSessionWebGPU(MODEL_FP32);
        usingEP = "webgpu/fp32";
      } catch (e2) {
        console.warn("WebGPU fp32 failed:", e2);
      }
    }
  }

  if (!session) {
    logStatus(statusEl.textContent + `\nLoading model… fallback WASM (fp32)…`);
    session = await initSessionWASM(MODEL_FP32);
    usingEP = "wasm/fp32";
  }

  // names
  inputName = session.inputNames?.[0] ?? "input";
  outputName = session.outputNames?.[0] ?? "output";

  badMatteStreak = 0;
  matteReady = false;

  logStatus(
    `Model ready ✅ (${usingEP})\nthreads=${ort.env.wasm.numThreads ?? 1}\n` +
    `input=${inputName}\noutput=${outputName}\n`
  );
}

async function ensureCamera() {
  logStatus("Camera starting…");
  // Encourage a decent resolution (helps matte quality)
  const constraints = {
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();

  setCanvasSizes();
  logStatus(statusEl.textContent + "\nCamera ready.");
}

async function inferOnce() {
  // draw to infer canvas using same crop logic (but infer size)
  drawVideoCover(inferCtx, inferCanvas.width, inferCanvas.height, MIRROR);

  const inputTensor = buildInputTensorFloat32();

  const feeds = {};
  feeds[inputName] = inputTensor;

  const results = await session.run(feeds);
  const outTensor = results[outputName];

  const { alpha, H, W } = parseOutputToAlpha(outTensor);
  const inv = maybeInvertAlpha(alpha, W, H);
  const st = alphaStats(alpha);

  // sanity: if matte is basically empty for several frames, do NOT apply (prevents "all white")
  const isBad = (st.mean < MIN_MEAN_ALPHA && st.max < MIN_MAX_ALPHA);

  if (isBad) {
    badMatteStreak++;
    console.warn("Bad matte:", { ...st, ...inv, badMatteStreak, usingEP });
    if (badMatteStreak >= BAD_STREAK_TO_FALLBACK && usingEP.startsWith("webgpu")) {
      // Known: WebGPU can be unstable on some devices/GPUs; fallback to WASM for correctness.:contentReference[oaicite:2]{index=2}
      logStatus(statusEl.textContent + "\nWebGPU matte unstable → fallback to WASM…");
      session = await initSessionWASM(MODEL_FP32);
      usingEP = "wasm/fp32";
      inputName = session.inputNames?.[0] ?? "input";
      outputName = session.outputNames?.[0] ?? "output";
      badMatteStreak = 0;
      matteReady = false;
      return;
    }

    // If we have a last-good mask, keep using it. Otherwise disable matte (show raw video).
    if (lastGoodMask.width === maskCanvas.width && lastGoodMask.height === maskCanvas.height) {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.drawImage(lastGoodMask, 0, 0);
      matteReady = true;
    } else {
      matteReady = false;
    }
    return;
  }

  badMatteStreak = 0;

  updateMaskCanvas(alpha, W, H);

  // store last good
  lastGoodCtx.clearRect(0, 0, lastGoodMask.width, lastGoodMask.height);
  lastGoodCtx.drawImage(maskCanvas, 0, 0);

  matteReady = true;

  // optional: show debug stats in status (comment out if noisy)
  // logStatus(`Model ready ✅ (${usingEP})\nmean=${st.mean.toFixed(3)} max=${st.max.toFixed(3)} invert=${inv.invert}\nRunning…`);
}

function loop() {
  if (!running) return;

  // draw every frame (smooth preview), but inference throttled by Infer FPS
  drawComposite();

  const fps = getInferFPS();
  const now = performance.now();
  const interval = 1000 / fps;

  if (now - lastInferTS >= interval) {
    lastInferTS = now;
    inferOnce().catch((e) => {
      console.error("infer error:", e);
      // if inference fails, avoid leaving a broken matte applied
      matteReady = false;
    });
  }

  requestAnimationFrame(loop);
}

async function start() {
  try {
    syncFPS();

    await ensureCamera();
    await initModel();

    running = true;
    lastInferTS = 0;
    logStatus(statusEl.textContent + "Running…");
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    logStatus("Start failed ❌\n" + (e?.message || e));
  }
}

function stop() {
  running = false;
  matteReady = false;
  badMatteStreak = 0;

  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;

  outCtx.clearRect(0, 0, out.width, out.height);
  logStatus("stopped");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// If user changes infer size mid-run, resize tensors/canvases safely
inferSizeSel?.addEventListener("change", () => {
  if (!video.videoWidth) return;
  setCanvasSizes();
});

// Keep fps synced even if only one exists
inferFpsSel?.addEventListener("change", () => {});
drawFpsSel?.addEventListener("change", () => {});
