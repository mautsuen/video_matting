// main.js (complete replacement)
// Browser MODNet background removal using ONNX Runtime Web (WebGPU first, fallback to WASM)

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.all.min.mjs";

const $ = (id) => document.getElementById(id);

const startBtn = $("start");
const stopBtn  = $("stop");
const fpsSel   = $("fps");
const inferFpsSel  = $("inferFps");
const inferSizeSel = $("inferSize");
const bgSel    = $("bg");

const video  = $("video");
const out    = $("out");
const statusEl = $("status");

// ====== Tunables (quality vs speed) ======
const MIRROR = false;

// This mimics your Python "crop columns [120:792]" effect (zoom mostly in X).
// Bigger -> zoom more -> face bigger -> usually better matting.
const ZOOM_X = 1.35;  // ~ 910/672 = 1.354...
const ZOOM_Y = 1.00;

const AUTO_INVERT = true; // if mask seems inverted, auto fix
const MASK_GAMMA  = 0.75; // <1 makes foreground stronger (helps face)
const MASK_BLUR_PX = 1.2; // edge feather (small blur)

// ORT WASM threads (requires crossOriginIsolated=true)
const MAX_WASM_THREADS = 8;

// ====== State ======
let running = false;
let drawTimer = null;
let inferTimer = null;
let inferBusy = false;

let session = null;
let inputName = null;
let outputName = null;
let using = "none";

// canvases
const outCtx = out.getContext("2d", { willReadFrequently: true });

const inferCanvas = document.createElement("canvas");
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

// last mask is stored in maskCanvas (same size as inferCanvas)

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

// Draw video into dstW x dstH with cover + extra zoom (asymmetric zoomX/zoomY)
function drawVideoCover(ctx, vid, dstW, dstH, mirror, zoomX = 1.0, zoomY = 1.0) {
  const vw = vid.videoWidth || 1;
  const vh = vid.videoHeight || 1;

  // base cover scale
  const baseScale = Math.max(dstW / vw, dstH / vh);

  // apply asymmetric zoom by shrinking sampled source region
  const sw = dstW / (baseScale * zoomX);
  const sh = dstH / (baseScale * zoomY);

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

// Build Float32 NCHW tensor with (x/255 - 0.5)/0.5  (matches Xenova/modnet preprocessor_config.json)
function buildInputTensorFromCanvas(ctx, w, h) {
  const img = ctx.getImageData(0, 0, w, h);
  const data = img.data; // RGBA
  const chw = new Float32Array(3 * w * h);

  const area = w * h;
  // (v/255 - 0.5)/0.5 == v/127.5 - 1
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

// Convert output tensor -> alpha mask on maskCanvas
function updateMaskFromOutput(outTensor) {
  // Handle common shapes:
  // [1, 1, H, W] or [1, H, W] or [1, C, H, W]
  const dims = outTensor.dims;
  const data = outTensor.data;

  let H, W, C = 1;
  let layout = "HWC?";

  if (dims.length === 4) {
    // [N, C, H, W]
    C = dims[1];
    H = dims[2];
    W = dims[3];
    layout = "NCHW";
  } else if (dims.length === 3) {
    // [N, H, W]
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

  // Detect type/range
  const t = outTensor.type; // 'float32'/'uint8'/...
  const isUint8 = (t === "uint8");
  const isInt8  = (t === "int8");

  // sample stats + auto invert heuristic
  // center box vs border
  const step = Math.max(1, Math.floor(Math.min(W, H) / 64));
  const cx0 = Math.floor(W * 0.25), cx1 = Math.floor(W * 0.75);
  const cy0 = Math.floor(H * 0.25), cy1 = Math.floor(H * 0.75);

  let centerSum = 0, centerCnt = 0;
  let borderSum = 0, borderCnt = 0;

  function getValAt(i /* 0..H*W-1 */) {
    // first channel only if C>1
    const base = (layout === "NCHW" && C > 1) ? i : i;
    let v = data[base];

    if (isUint8) v = v / 255;
    else if (isInt8) v = (v + 128) / 255; // crude fallback
    // float types assumed already 0..1 (as model card suggests)
    return v;
  }

  // If float is not in [0,1], apply sigmoid (some exports output logits)
  let needSigmoid = false;
  if (!isUint8 && !isInt8) {
    // quick probe a few points
    for (let y = 0; y < H; y += Math.max(1, Math.floor(H / 16))) {
      for (let x = 0; x < W; x += Math.max(1, Math.floor(W / 16))) {
        const i = y * W + x;
        const v = getValAt(i);
        if (v < -0.1 || v > 1.1) { needSigmoid = true; break; }
      }
      if (needSigmoid) break;
    }
  }

  function sigmoid(x) {
    // stable-ish
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    } else {
      const z = Math.exp(x);
      return z / (1 + z);
    }
  }

  // Gather means for auto invert
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

  // If center is "more background" than border, likely inverted for a portrait.
  const doInvert = AUTO_INVERT && (centerMean < borderMean);

  // Write alpha
  // We keep RGB=0 and only alpha channel; later we use destination-in compositing.
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let v = getValAt(i);
      if (needSigmoid) v = sigmoid(v);
      v = clamp01(v);

      if (doInvert) v = 1.0 - v;

      // strengthen foreground a bit
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

  return { H, W, doInvert, needSigmoid, centerMean, borderMean, outType: t };
}

async function initModel() {
  const hc = navigator.hardwareConcurrency || 4;
  ort.env.wasm.numThreads = Math.max(1, Math.min(MAX_WASM_THREADS, hc));

  // If you host wasm files locally at /wasm, use this:
  //   /wasm/ort-wasm-simd-threaded.jsep.mjs
  //   /wasm/ort-wasm-simd-threaded.wasm
  //   /wasm/ort-wasm-simd.wasm
  //   /wasm/ort-wasm.wasm
  // Otherwise you can switch to CDN by commenting this out.
  // ✅ 一定要用絕對 URL，不能用 "/wasm/"
  const ORIGIN = window.location.origin; // http://localhost:8000
  ort.env.wasm.wasmPaths = `${ORIGIN}/wasm/`;


  const base = "/models/Xenova/modnet/onnx";
  const modelWebGPU = `${base}/model_fp16.onnx`;    // faster if available
  const modelFP32   = `${base}/model.onnx`;         // best quality
  const modelUint8  = `${base}/model_uint8.onnx`;   // small, often fastest on wasm

  const canWebGPU = !!navigator.gpu && window.isSecureContext;

  // Try WebGPU first (fp16, then fp32)
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

  // Fallback to WASM (prefer fp32 quality first, then uint8)
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

  console.log("ORT using =", using);
  console.log("inputs =", session.inputNames, session.inputMetadata);
  console.log("outputs =", session.outputNames, session.outputMetadata);

  setStatus(`Model ready ✅ (${using})\ninput=${inputName}\noutput=${outputName}`);
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
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

  // output size fixed to 480 wide
  const outW = 480;
  const outH = Math.round(outW * (vh / vw));

  out.width = outW;
  out.height = outH;

  // infer size set later by inferSizeSel; but keep aspect aligned to output.
}

async function inferOnce() {
  if (!running || !session) return;
  if (inferBusy) return;
  inferBusy = true;

  try {
    const outW = out.width, outH = out.height;

    // infer short edge from UI
    const shortEdge = parseInt(inferSizeSel?.value || "256", 10);

    // keep same aspect as output so mask aligns after scaling
    let iw, ih;
    if (outW <= outH) {
      iw = shortEdge;
      ih = Math.round(shortEdge * (outH / outW));
    } else {
      ih = shortEdge;
      iw = Math.round(shortEdge * (outW / outH));
    }

    // enforce size_divisibility=32 (from preprocessor_config.json)
    iw = roundDownTo(iw, 32);
    ih = roundDownTo(ih, 32);

    inferCanvas.width = iw;
    inferCanvas.height = ih;

    // draw frame for inference (same crop policy as output)
    drawVideoCover(inferCtx, video, iw, ih, MIRROR, ZOOM_X, ZOOM_Y);

    const inputTensor = buildInputTensorFromCanvas(inferCtx, iw, ih);
    const feeds = { [inputName]: inputTensor };

    const t0 = performance.now();
    const results = await session.run(feeds);
    const t1 = performance.now();

    const outTensor = results[outputName] || results[session.outputNames[0]];
    if (!outTensor) throw new Error("No output tensor found.");

    const info = updateMaskFromOutput(outTensor);

    // lightweight perf info in console
    console.log(`[infer] ${using} ${iw}x${ih} ${(t1 - t0).toFixed(1)}ms`,
      { ...info }
    );

  } catch (e) {
    console.error(e);
    setStatus(`Infer error ❌\n${e?.message || e}`);
  } finally {
    inferBusy = false;
  }
}

function drawOnce() {
  if (!running) return;

  const outW = out.width, outH = out.height;
  const bgMode = bgSel?.value || "white";

  // Background
  outCtx.save();
  outCtx.setTransform(1, 0, 0, 1, 0, 0);
  outCtx.clearRect(0, 0, outW, outH);

  if (bgMode === "white") {
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, outW, outH);
  } else if (bgMode === "green") {
    outCtx.fillStyle = "#00ff00";
    outCtx.fillRect(0, 0, outW, outH);
  } else if (bgMode === "blur") {
    outCtx.filter = "blur(10px)";
    drawVideoCover(outCtx, video, outW, outH, MIRROR, ZOOM_X, ZOOM_Y);
    outCtx.filter = "none";
  } else {
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, outW, outH);
  }

  // Foreground (masked)
  // Draw fg first
  outCtx.save();
  outCtx.globalCompositeOperation = "source-over";
  drawVideoCover(outCtx, video, outW, outH, MIRROR, ZOOM_X, ZOOM_Y);

  // Apply mask (destination-in)
  outCtx.globalCompositeOperation = "destination-in";
  if (MASK_BLUR_PX > 0) outCtx.filter = `blur(${MASK_BLUR_PX}px)`;
  outCtx.drawImage(maskCanvas, 0, 0, outW, outH);
  outCtx.filter = "none";
  outCtx.restore();

  outCtx.restore();
}

function stopLoops() {
  if (drawTimer) { clearInterval(drawTimer); drawTimer = null; }
  if (inferTimer) { clearInterval(inferTimer); inferTimer = null; }
}

function startLoops() {
  stopLoops();

  const drawFPS = parseInt(fpsSel?.value || "10", 10);
  const inferFPS = parseInt(inferFpsSel?.value || "5", 10);

  // Draw loop (smoothness)
  drawTimer = setInterval(drawOnce, Math.max(1, Math.round(1000 / Math.max(1, drawFPS))));

  // Infer loop (real speed bottleneck)
  inferTimer = setInterval(inferOnce, Math.max(1, Math.round(1000 / Math.max(1, inferFPS))));
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
      `out=${out.width}x${out.height}`
    );

    setStatus("Loading model…");
    await initModel();

    setStatus(statusEl.textContent + "\nRunning…");
    startLoops();

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
  stopLoops();
  stopCamera();

  startBtn.disabled = false;
  stopBtn.disabled = true;

  setStatus("Stopped.");
}

// ===== UI wiring =====
stopBtn.disabled = true;

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// Restart loops when user changes FPS/size/background
[fpsSel, inferFpsSel, inferSizeSel, bgSel].forEach((el) => {
  if (!el) return;
  el.addEventListener("change", () => {
    if (!running) return;
    startLoops();
  });
});
