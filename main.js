import { pipeline, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const startBtn = document.getElementById("start");
const stopBtn  = document.getElementById("stop");
const fpsSel   = document.getElementById("fps");
const statusEl = document.getElementById("status");
const video    = document.getElementById("video");
const outCanvas = document.getElementById("out");
const outCtx    = outCanvas.getContext("2d");

let stream = null;
let running = false;
let segmenter = null;

// hidden canvas for grabbing frames
const grab = document.createElement("canvas");
const gctx = grab.getContext("2d", { willReadFrequently: true });

async function loadModel() {
  statusEl.textContent = "Loading Xenova/modnet (first load can be slow)...";
  // 官方範例：background-removal + Xenova/modnet :contentReference[oaicite:4]{index=4}
  segmenter = await pipeline("background-removal", "Xenova/modnet", { dtype: "fp32" });
  statusEl.textContent = "Model ready.";
}

async function start() {
  if (!segmenter) await loadModel();

  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  video.srcObject = stream;
  await video.play();

  running = true;
  statusEl.textContent = "Running...";

  const loop = async () => {
    if (!running) return;

    // 解析度越小越快（可自行調）
    const W = 640, H = 480;
    grab.width = W; grab.height = H;
    outCanvas.width = W; outCanvas.height = H;

    gctx.drawImage(video, 0, 0, W, H);

    // RawImage.fromCanvas 是官方支援的入口 :contentReference[oaicite:5]{index=5}
    const input = RawImage.fromCanvas(grab);
    const output = await segmenter(input);

    // 背景移除 pipeline 通常回傳一張「帶 alpha 的影像」(channels=4) :contentReference[oaicite:6]{index=6}
    const outImg = output[0];
    const outC = outImg.toCanvas();

    // 你可以在這裡換背景：先鋪底色，再畫有 alpha 的結果
    outCtx.clearRect(0, 0, W, H);
    outCtx.fillStyle = "#ffffff";
    outCtx.fillRect(0, 0, W, H);
    outCtx.drawImage(outC, 0, 0, W, H);

    const fps = Number(fpsSel.value);
    setTimeout(loop, Math.round(1000 / fps));
  };

  loop();
}

function stop() {
  running = false;
  statusEl.textContent = "Stopped.";
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

startBtn.onclick = start;
stopBtn.onclick  = stop;
