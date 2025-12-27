// main.js (FPS synced: infer + draw same tick, uniform zoom to avoid distortion on portrait)
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.all.min.mjs";

const $ = (id) => document.getElementById(id);

const startBtn = $("start");
const stopBtn  = $("stop");
const fpsSel   = $("fps");
const inferSizeSel = $("inferSize");
const bgSel    = $("bg");

const video  = $("video");
const out    = $("out");
const statusEl = $("status");

const outCtx = out.getContext("2d", { willReadFrequently: true });

const inferCanvas = document.createElement("canvas");
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

// ====== Tunables ======
const MIRROR = true;

// ✅ Uniform zoom only (NO ZOOM_X/ZOOM_Y)
// Portrait phones: keep zoom near 1.0 to avoid warping faces
const ZOOM_LANDSCAPE = 1.20; // mild zoom-in on PC
const ZOOM_PORTRAIT  = 1.00; // critical: no anisotropic zoom on phone portrait

const AUTO_INVERT = true;
const MASK_GAMMA  = 0.75;
const MASK_BLUR_PX = 1.2;

const MAX_WASM_THREADS = 8;

// ====== State ======
let running = false;
let loopTimer = null;

let session = null;
let inputName = null;
let outputName = null;
let using = "none";

let maskReady = false;

function setStatus(s) {
  statusEl.textContent = s;
  console.log(s);
}

function clamp01(x) {
  return x < 0 ? 0 : (x > 1 ? 1 : x);
}

function roundDownTo(n, d) {
  return Math.max(d, Math.floor(n / d) * d);
}

// Decide zoom based on current orientation (video or canvas)
function currentZoom() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  const portrait = (vh > vw) || (out.height > out.width);
  return portrait ? ZOOM_PORTRAIT : ZOOM_LANDSCAPE;
}

// ✅ Uniform "cover + zoom" (single zoom factor, no distortion)
function drawVideoCover(ctx, vid, dstW, dstH, mirror, zoom = 1.0) {
  const vw = vid.videoWidth || 1;
  const vh = vid.videoHeight || 1;

  // base cover scale
  const baseScale = Math.max(dstW / vw, dstH / vh);

  // uniform zoom => sw/sh shrink together => no aspect warping
  const sw = dstW / (baseScale * zoom);
  const sh = dstH / (baseScale * zoom);

  const sx = (vw - sw) * 0.5;
  const sy = (vh - sh) * 0.5;

  ctx.save();
  if (mirror) {
    ctx.translate(dstW, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(vid, sx, sy, sw, sh, 0, 0, dstW, dstH);
  ctx.restore();
}

function buildInputTensorFromCanvas(ctx, w, h) {
  const img = ctx.getImageData(0, 0, w, h);
  const data = img.data; // RGBA
  const chw = new Float32Array(3 * w * h);
  const area = w * h;

  // normalize: v/127.5 - 1
  for (let i = 0; i < area; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    chw[i] = r / 127.5 - 1.0;
    chw[area + i] = g / 127.5 - 1.0;
    chw[2 * area + i] = b / 127.5 - 1.0;
  }

  return new ort.Tensor("float32", chw, [1, 3, h, w]);
}

function updateMaskFromOutput(outTensor) {
  const dims = outTensor.dims;
  const data = outTensor.data;

  let H, W, C = 1;
  let layout;

  if (dims.length === 4) {
    C = dims[1];
    H = dims[2];
    W = dims[3];
    layout = "NCHW";
  } else if (dims.length === 3) {
    H = dims[1];
    W = dims[2];
    layout = "NHW";
  } else {
    throw new Error(`Unexpected output dims: ${JSON.stringify(dims)}`);
  }

  maskCanvas.width = W;
  maskCanvas.height = H;

  const img = maskCtx.createImageData(W, H);
  const rgba = img.data;

  const t = outTensor.type;
  const isUint8 = (t === "uint8");
  const isInt8  = (t === "int8");

  function getValAt(i) {
    const base = (layout === "NCHW" && C > 1) ? i : i;
    let v = data[base];
    if (isUint8) v = v / 255;
    else if (isInt8) v = (v + 128) / 255;
    return v;
  }

  // detect logits
  let needSigmoid = false;
  if (!isUint8 && !isInt8) {
    for (let k = 0; k < Math.min(W * H, 2000); k += 113) {
      const v = getValAt(k);
      if (v < -0.1 || v > 1.1) { needSigmoid = true; break; }
    }
  }

  function sigmoid(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    } else {
      const z = Math.exp(x);
      return z / (1 + z);
    }
  }

  // auto invert heuristic
  const step = Math.max(1, Math.floor(Math.min(W, H) / 64));
  const cx0 = Math.floor(W * 0.25), cx1 = Math.floor(W * 0.75);
  const cy0 = Math.floor(H * 0.25), cy1 = Math.floor(H * 0.75);

  let centerSum = 0, centerCnt = 0;
  let borderSum = 0, borderCnt = 0;

  for (let y = 0; y < H; y += step) {
    for (let x = 0; x < W; x += step) {
      const i = y * W + x;
      let v = getValAt(i);
      if (needSigmoid) v = sigmoid(v);
      v = clamp01(v);

      const inCenter = (x >= cx0 && x < cx1 && y >= cy0 && y < cy1);
      const inBorder = (x < W * 0.08 || x > W * 0.92 || y < H * 0.08 || y > H * 0.92);

      if (inCenter) { centerSum += v; centerCnt++; }
      if (inBorder) { borderSum += v; borderCnt++; }
    }
  }

  const centerMean = centerCnt ? centerSum / centerCnt : 0.5;
  const borderMean = borderCnt ? borderSum / borderCnt : 0.5;
  const doInvert = AUTO_INVERT && (centerMean < borderMean);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let v = getValAt(i);
      if (needSigmoid) v = sigmoid(v);
      v = clamp01(v);
      if (doInvert) v = 1.0 - v;

      v = Math.pow(v, MASK_GAMMA);
      const a = Math.round(clamp01(v) * 255);

      const p = i * 4;
      rgba[p + 0] = 0;
      rgba[p + 1] = 0;
      rgba[p + 2] = 0;
      rgba[p + 3] = a;
    }
  }

  maskCtx.putImageData(img, 0, 0);
  maskReady = true;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      // If you want front camera on phones, uncomment:
      // facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  const s = video.srcObject;
  if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

function configureCanvasSizes() {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;

  const outW = 480;
  const outH = Math.round(outW * (vh / vw));
  out.width = outW;
  out.height = outH;
}

async function initModel() {
  const hc = navigator.hardwareConcurrency || 4;
  ort.env.wasm.numThreads = Math.max(1, Math.min(MAX_WASM_THREADS, hc));

  // Absolute URL so it doesn't resolve to cdn.jsdelivr.net/wasm/...
  const ORIGIN = window.location.origin;
  ort.env.wasm.wasmPaths = `${ORIGIN}/wasm/`;

  const base = "/models/Xenova/modnet/onnx";
  const modelWebGPU = `${base}/model_fp16.onnx`;
  const modelFP32   = `${base}/model.onnx`;
  const modelUint8  = `${base}/model_uint8.onnx`;

  const canWebGPU = !!navigator.gpu && window.isSecureContext;

  session = null;
  using = "none";

  if (canWebGPU) {
    try {
      setStatus("Loading model… try WebGPU (fp16)…");
      session = await ort.InferenceSession.create(modelWebGPU, {
        executionProviders: ["webgpu"],
      });
      using = "webgpu/fp16";
    } catch (e1) {
      console.warn("WebGPU fp16 failed:", e1);
      try {
        setStatus("Loading model… try WebGPU (fp32)…");
        session = await ort.InferenceSession.create(modelFP32, {
          executionProviders: ["webgpu"],
        });
        using = "webgpu/fp32";
      } catch (e2) {
        console.warn("WebGPU fp32 failed:", e2);
      }
    }
  }

  if (!session) {
    try {
      setStatus(`Loading model… fallback WASM (fp32, threads=${ort.env.wasm.numThreads})…`);
      session = await ort.InferenceSession.create(modelFP32, {
        executionProviders: ["wasm"],
      });
      using = "wasm/fp32";
    } catch (e3) {
      console.warn("WASM fp32 failed:", e3);
      setStatus(`Loading model… fallback WASM (uint8, threads=${ort.env.wasm.numThreads})…`);
      session = await ort.InferenceSession.create(modelUint8, {
        executionProviders: ["wasm"],
      });
      using = "wasm/uint8";
    }
  }

  if (!session) throw new Error("Model init failed (WebGPU and WASM both failed).");

  inputName  = session.inputNames?.[0]  || "input";
  outputName = session.outputNames?.[0] || "output";

  setStatus(`Model ready ✅ (${using})\nthreads=${ort.env.wasm.numThreads}\ninput=${inputName}\noutput=${outputName}`);
}

async function inferOnce() {
  if (!running || !session) return;

  const outW = out.width, outH = out.height;
  const shortEdge = parseInt(inferSizeSel?.value || "256", 10);

  // keep aspect aligned to output
  let iw, ih;
  if (outW <= outH) {
    iw = shortEdge;
    ih = Math.round(shortEdge * (outH / outW));
  } else {
    ih = shortEdge;
    iw = Math.round(shortEdge * (outW / outH));
  }

  iw = roundDownTo(iw, 32);
  ih = roundDownTo(ih, 32);

  inferCanvas.width = iw;
  inferCanvas.height = ih;

  const zoom = currentZoom();
  drawVideoCover(inferCtx, video, iw, ih, MIRROR, zoom);

  const inputTensor = buildInputTensorFromCanvas(inferCtx, iw, ih);
  const feeds = { [inputName]: inputTensor };

  const results = await session.run(feeds);
  const outTensor = results[outputName] || results[session.outputNames[0]];
  if (!outTensor) throw new Error("No output tensor found.");

  updateMaskFromOutput(outTensor);
}

function drawOnce() {
  if (!running) return;

  const outW = out.width, outH = out.height;
  const bgMode = bgSel?.value || "white";
  const zoom = currentZoom();

  outCtx.save();
  outCtx.setTransform(1, 0, 0, 1, 0, 0);
  outCtx.clearRect(0, 0, outW, outH);

  // Background
  if (bgMode === "white") {
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, outW, outH);
  } else if (bgMode === "green") {
    outCtx.fillStyle = "#00ff00";
    outCtx.fillRect(0, 0, outW, outH);
  } else if (bgMode === "blur") {
    outCtx.filter = "blur(10px)";
    drawVideoCover(outCtx, video, outW, outH, MIRROR, zoom);
    outCtx.filter = "none";
  } else {
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, outW, outH);
  }

  // Foreground + mask
  outCtx.save();
  outCtx.globalCompositeOperation = "source-over";
  drawVideoCover(outCtx, video, outW, outH, MIRROR, zoom);

  if (maskReady && maskCanvas.width > 0 && maskCanvas.height > 0) {
    outCtx.globalCompositeOperation = "destination-in";
    if (MASK_BLUR_PX > 0) outCtx.filter = `blur(${MASK_BLUR_PX}px)`;
    outCtx.drawImage(maskCanvas, 0, 0, outW, outH);
    outCtx.filter = "none";
  }

  outCtx.restore();
  outCtx.restore();
}

function stopLoop() {
  if (loopTimer) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
}

function scheduleNextTick(delayMs) {
  loopTimer = setTimeout(tick, Math.max(0, delayMs));
}

// FPS synced loop: infer -> draw -> wait
async function tick() {
  if (!running) return;

  const fps = parseInt(fpsSel?.value || "10", 10);
  const targetMs = 1000 / Math.max(1, fps);

  const t0 = performance.now();
  try {
    await inferOnce();
    drawOnce();
  } catch (e) {
    console.error(e);
    setStatus(`Runtime error ❌\n${e?.message || e}`);
  }
  const t1 = performance.now();

  const elapsed = t1 - t0;
  const delay = targetMs - elapsed;
  scheduleNextTick(delay);
}

async function start() {
  if (running) return;
  running = true;

  startBtn.disabled = true;
  stopBtn.disabled = false;

  try {
    setStatus("Camera starting…");
    await startCamera();

    configureCanvasSizes();

    setStatus(
      `Camera ready.\nsecureContext=${window.isSecureContext}\n` +
      `crossOriginIsolated=${window.crossOriginIsolated}\n` +
      `navigator.gpu=${!!navigator.gpu}\n` +
      `out=${out.width}x${out.height}\n` +
      `zoom=${currentZoom().toFixed(2)}`
    );

    maskReady = false;

    setStatus("Loading model…");
    await initModel();

    setStatus(statusEl.textContent + "\nRunning…");
    stopLoop();
    scheduleNextTick(0);

  } catch (e) {
    console.error(e);
    setStatus(`Start failed ❌\n${e?.message || e}`);
    running = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stop() {
  running = false;
  stopLoop();
  stopCamera();

  session = null;
  maskReady = false;

  startBtn.disabled = false;
  stopBtn.disabled = true;

  setStatus("Stopped.");
}

stopBtn.disabled = true;
startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

[fpsSel, inferSizeSel, bgSel].forEach((el) => {
  if (!el) return;
  el.addEventListener("change", () => {
    if (!running) return;
    stopLoop();
    scheduleNextTick(0);
  });
});
