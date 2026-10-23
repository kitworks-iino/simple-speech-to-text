const $ = (id) => document.getElementById(id);
const speak = $("speak"), stop = $("stop"), status = $("status"), indicator = $("indicator");
const transcript = $("transcript"), copy = $("copy");
const SESSION_KEY = "simple-stt-trial-token";
const fragment = new URLSearchParams(location.hash.slice(1));
let token = fragment.get("t") || "";
try {
  if (token) sessionStorage.setItem(SESSION_KEY, token);
  else token = sessionStorage.getItem(SESSION_KEY) || "";
} catch { /* Private browsing may disable storage; the current tab still works. */ }
if (location.hash) history.replaceState(null, "", location.pathname + location.search);

let phase = "connecting", ready = false, generation = 0;
let context, stream, source, worklet, recordingTimer, flushDone, controller;
let chunks = [], frames = 0, sampleRate = 0;

function show(message, state = phase) {
  phase = state;
  status.textContent = message;
  indicator.className = state === "recording" ? "recording" :
    ["preparing", "transcribing", "stopping"].includes(state) ? "working" : "";
  speak.disabled = !ready || state !== "idle";
  speak.textContent = state === "recording" ? "録音中…" : state === "transcribing" ? "文字起こし中…" : "話す";
  stop.hidden = state !== "recording";
}

async function cleanAudio() {
  clearTimeout(recordingTimer);
  recordingTimer = null;
  if (stream) for (const track of stream.getTracks()) track.stop();
  if (worklet) {
    worklet.port.onmessage = null;
    try { worklet.disconnect(); } catch { /* Already disconnected. */ }
  }
  if (source) { try { source.disconnect(); } catch { /* Already disconnected. */ } }
  const previousContext = context;
  worklet = source = stream = context = null;
  if (previousContext && previousContext.state !== "closed") {
    try { await previousContext.close(); } catch { /* Audio session may already be gone. */ }
  }
}

function cancel(message) {
  generation++;
  controller?.abort();
  controller = null;
  flushDone?.();
  flushDone = null;
  void cleanAudio();
  chunks = [];
  show(message, "idle");
}

function errorMessage(error) {
  if (error.name === "NotAllowedError") return "マイクを許可してから、もう一度お試しください。";
  if (error.name === "NotFoundError") return "マイクが見つかりません。接続を確認してください。";
  if (error.name === "NotReadableError") return "マイクを利用できません。他の録音を終了してお試しください。";
  return error.message || "接続できませんでした。もう一度お試しください。";
}

function wavBlob(blocks, count, rate) {
  const buffer = new ArrayBuffer(44 + count * 2);
  const view = new DataView(buffer);
  const label = (offset, text) => [...text].forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)));
  label(0, "RIFF"); view.setUint32(4, 36 + count * 2, true); label(8, "WAVE");
  label(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  label(36, "data"); view.setUint32(40, count * 2, true);
  let offset = 44;
  for (const block of blocks) for (const sample of block) {
    const value = Math.max(-1, Math.min(1, Number.isFinite(sample) ? sample : 0));
    view.setInt16(offset, Math.round(value < 0 ? value * 32768 : value * 32767), true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function acceptSamples(samples, run) {
  if (run !== generation || !["recording", "stopping"].includes(phase)) return;
  const available = 30 * sampleRate - frames;
  if (available <= 0) return;
  const block = samples.length > available ? samples.slice(0, available) : samples;
  chunks.push(block);
  frames += block.length;
  if (phase !== "recording") return;
  // Silence never ends recording. Only the stop button or the explicit
  // temporary-demo duration limit completes this capture.
  if (frames >= 30 * sampleRate) void finish(run);
}

async function begin() {
  if (!ready || phase !== "idle") return;
  const run = ++generation;
  chunks = []; frames = 0;
  show("マイクを準備しています…", "preparing");
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!window.isSecureContext || !AudioContextClass || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("iPhoneのSafariでHTTPSの試用リンクを開いてください。");
    }
    context = new AudioContextClass();
    if (!context.audioWorklet) throw new Error("このブラウザーは録音方式に対応していません。最新のSafariでお試しください。");
    // Resume in the original tap handler, before awaiting permission on iOS.
    const resumed = context.resume().then(() => null, (error) => error);
    const granted = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false,
    });
    if (run !== generation) { granted.getTracks().forEach((track) => track.stop()); return; }
    stream = granted;
    const resumeError = await resumed;
    if (resumeError) throw resumeError;
    if (run !== generation) return;
    sampleRate = context.sampleRate;
    if (sampleRate < 8000 || sampleRate > 96000) throw new Error("このマイクのサンプルレートには対応していません。");
    await context.audioWorklet.addModule("/pcm-worklet.js");
    if (run !== generation) return;
    source = context.createMediaStreamSource(stream);
    worklet = new AudioWorkletNode(context, "pcm-capture", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    worklet.port.onmessage = ({ data }) => {
      if (data?.type === "samples") acceptSamples(data.samples, run);
      else if (data?.type === "flushed") { flushDone?.(); flushDone = null; }
    };
    source.connect(worklet);
    // The worklet emits only zeros; connect it so Safari continues processing.
    worklet.connect(context.destination);
    for (const track of stream.getTracks()) track.addEventListener("ended", () => {
      if (run === generation && phase === "recording") cancel("マイクの接続が切れました。もう一度お試しください。");
    });
    show("録音中です。話し終わったら「停止して文字起こし」を押してください。", "recording");
    recordingTimer = setTimeout(() => { if (run === generation) void finish(run); }, 30000);
  } catch (error) {
    if (run === generation) cancel(errorMessage(error));
  }
}

async function finish(run) {
  if (run !== generation || phase !== "recording") return;
  show("録音を終了しています…", "stopping");
  clearTimeout(recordingTimer);
  await new Promise((resolve) => {
    const timer = setTimeout(() => { flushDone = null; resolve(); }, 400);
    flushDone = () => { clearTimeout(timer); resolve(); };
    worklet.port.postMessage({ type: "flush" });
  });
  if (run !== generation) return;
  await cleanAudio();
  if (run !== generation) return;
  if (!frames) { cancel("音声を録音できませんでした。もう一度お試しください。"); return; }
  const audio = wavBlob(chunks, frames, sampleRate);
  chunks = [];
  show("文字起こし中です。少しお待ちください…", "transcribing");
  const requestController = new AbortController();
  controller = requestController;
  const requestTimer = setTimeout(() => requestController.abort(), 180000);
  try {
    const response = await fetch("/api/transcribe", {
      method: "POST", headers: { Authorization: `Bearer ${token}`, "Content-Type": "audio/wav" },
      body: audio, signal: requestController.signal, cache: "no-store",
    });
    const result = await response.json();
    if (run !== generation) return;
    if (!response.ok) throw new Error(result.error || result.detail || "文字起こしできませんでした。もう一度お試しください。");
    if (typeof result.text !== "string" || !result.text.trim()) throw new Error("文字を認識できませんでした。もう一度お試しください。");
    transcript.value = result.text.trim();
    copy.disabled = false;
    show("できました。もう一度試すこともできます。", "idle");
  } catch (error) {
    if (run === generation) show(error.name === "AbortError" ? "処理がタイムアウトしました。短めに話してお試しください。" : errorMessage(error), "idle");
  } finally {
    clearTimeout(requestTimer);
    if (run === generation) controller = null;
  }
}

speak.addEventListener("click", begin);
stop.addEventListener("click", () => void finish(generation));
copy.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(transcript.value); copy.textContent = "コピー済み"; }
  catch { transcript.focus(); transcript.select(); copy.textContent = "選択しました"; }
  setTimeout(() => { copy.textContent = "コピー"; }, 1800);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && ["preparing", "recording", "stopping", "transcribing"].includes(phase)) cancel("中止しました。もう一度「話す」を押してください。");
});
window.addEventListener("pagehide", () => cancel("中止しました。"));

async function connect() {
  if (!token) { show("試用リンクが必要です。案内されたURLから開いてください。", "idle"); return; }
  try {
    const response = await fetch("/api/status", { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
    const result = await response.json();
    if (!response.ok || result.ready === false) throw new Error(result.error || "試用サーバーを利用できません。リンクの有効期限を確認してください。");
    ready = true;
    show("準備できました。ボタンを押して話してください。", "idle");
  } catch (error) { show(errorMessage(error), "idle"); }
}
void connect();
