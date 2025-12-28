const state = {
  config: null,
  audioContext: null,
  gainNode: null,
  sourceNode: null,
  workletNode: null,
  silentGainNode: null,
  mediaStream: null,
  syntheticOscillators: [],
  running: false,
  ringBufferFigure1: null,
  ringBufferFigure2: null,
  ringBufferFigure3: null,
  drawLoopId: null,
  lastDrawTimeFigure1: 0,
  lastDrawTimeFigure2: 0,
  lastDrawTimeFigure3: 0,
  params: {
    audioInput: {},
    figure1: {},
    figure2: {},
    figure3: {},
  },
  figure2Cache: {
    size: 0,
    windowFunc: "",
    window: null,
    real: null,
    imag: null,
    smoothed: null,
  },
  figure3Cache: {
    size: 0,
    windowFunc: "",
    window: null,
    real: null,
    imag: null,
    magnitudes: null,
  },
  figure3History: [],
  figure3HistoryLength: 0,
  figure3LastSegmentTime: 0,
  figure3Signature: "",
  figure3ImageData: null,
  figure3ImageCanvas: null,
  figure3ColorLUT: null,
  figure3ColorKey: "",
  plotFontSize: 48,
  restartTimer: null,
  userGestureReady: false,
};

const CANVAS_BG = "#151c26";
const DEFAULT_PLOT_FONT_SIZE = 48;
const DEFAULT_PARAM_FONT_SIZE = 14;

function plotFont() {
  const size = Number.isFinite(state.plotFontSize) ? state.plotFontSize : DEFAULT_PLOT_FONT_SIZE;
  return `${size}px 'Space Grotesk', sans-serif`;
}

function setParamFontSize(size) {
  const value = Number.isFinite(size) && size > 0 ? size : DEFAULT_PARAM_FONT_SIZE;
  document.documentElement.style.setProperty("--param-font-size", `${value}px`);
}

const elements = {
  inputSource: document.getElementById("inputSource"),
  gain: document.getElementById("gain"),
  sampleRateInput: document.getElementById("sampleRateInput"),
  channelsInput: document.getElementById("channelsInput"),
  framesPerChunk: document.getElementById("framesPerChunk"),
  sampleWidthBytes: document.getElementById("sampleWidthBytes"),
  syntheticFrequencies: document.getElementById("syntheticFrequencies"),
  timeWindow: document.getElementById("timeWindow"),
  maxPoints: document.getElementById("maxPoints"),
  refreshInterval: document.getElementById("refreshInterval"),
  lineWidth: document.getElementById("lineWidth"),
  markerSize: document.getElementById("markerSize"),
  maxAmplitudeInput: document.getElementById("maxAmplitudeInput"),
  fftTimeWindow: document.getElementById("fftTimeWindow"),
  fftSize: document.getElementById("fftSize"),
  windowFunc: document.getElementById("windowFunc"),
  fftScale: document.getElementById("fftScale"),
  minFrequency: document.getElementById("minFrequency"),
  maxFrequency: document.getElementById("maxFrequency"),
  smoothing: document.getElementById("smoothing"),
  fftRefreshInterval: document.getElementById("fftRefreshInterval"),
  fftLineWidth: document.getElementById("fftLineWidth"),
  fftMarkerSize: document.getElementById("fftMarkerSize"),
  surfaceWindow: document.getElementById("surfaceWindow"),
  surfaceSegment: document.getElementById("surfaceSegment"),
  surfaceFftSize: document.getElementById("surfaceFftSize"),
  surfaceTimeBins: document.getElementById("surfaceTimeBins"),
  surfaceFrequencyBins: document.getElementById("surfaceFrequencyBins"),
  surfaceMinFrequency: document.getElementById("surfaceMinFrequency"),
  surfaceMaxFrequency: document.getElementById("surfaceMaxFrequency"),
  surfaceColormap: document.getElementById("surfaceColormap"),
  surfaceNormalize: document.getElementById("surfaceNormalize"),
  surfaceRefreshInterval: document.getElementById("surfaceRefreshInterval"),
  surfaceInterpolation: document.getElementById("surfaceInterpolation"),
  statusText: document.getElementById("statusText"),
  sampleRate: document.getElementById("sampleRate"),
  channels: document.getElementById("channels"),
  figureMeta: document.getElementById("figureMeta"),
  canvas: document.getElementById("signalCanvas"),
  fftMeta: document.getElementById("fftMeta"),
  fftCanvas: document.getElementById("fftCanvas"),
  surfaceMeta: document.getElementById("surfaceMeta"),
  surfaceCanvas: document.getElementById("surfaceCanvas"),
  streamIndicator: document.getElementById("streamIndicator"),
  enableAudio: document.getElementById("enableAudio"),
  syntheticFrequencyControl: document.getElementById("syntheticFrequencyControl"),
};

class CircularBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
    this.writeIndex = 0;
    this.size = 0;
  }

  push(chunk) {
    const len = chunk.length;
    for (let i = 0; i < len; i += 1) {
      this.buffer[this.writeIndex] = chunk[i];
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      if (this.size < this.capacity) this.size += 1;
    }
  }

  toArray(latestCount) {
    const count = Math.min(latestCount, this.size);
    const start = (this.writeIndex - count + this.capacity) % this.capacity;
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      out[i] = this.buffer[(start + i) % this.capacity];
    }
    return out;
  }

  resize(newCapacity) {
    const current = this.toArray(Math.min(newCapacity, this.size));
    this.capacity = newCapacity;
    this.buffer = new Float32Array(newCapacity);
    this.buffer.set(current, 0);
    this.size = current.length;
    this.writeIndex = current.length % newCapacity;
  }
}

async function fetchConfig() {
  const candidates = ["./config.yaml", "../config.yaml"];
  for (const path of candidates) {
    try {
      const res = await fetch(path);
      if (!res.ok) continue;
      const text = await res.text();
      return window.jsyaml.load(text);
    } catch {
      /* try next */
    }
  }
  throw new Error("Unable to load config.yaml (checked ./ and ../)");
}

function setStatus(text) {
  if (elements.statusText) {
    elements.statusText.textContent = text;
  } else {
    console.info(text);
  }
}

function setStreamingIndicator(active) {
  if (!elements.streamIndicator) return;
  elements.streamIndicator.classList.toggle("hidden", !active);
}

function setAwaitingGesture(active) {
  document.body.classList.toggle("awaiting-gesture", active);
}

function syncUIFromConfig(config) {
  const audio = config.audio_input;
  const fig = config.figure1;
  const fig2 = config.figure2;
  const fig3 = config.figure3;
  const plotFont = Number(config.plot_font_size);
  const paramFont = Number(config.ui_param_font_size);
  state.plotFontSize = Number.isFinite(plotFont) && plotFont > 0 ? plotFont : DEFAULT_PLOT_FONT_SIZE;
  setParamFontSize(paramFont);
  elements.inputSource.value = audio.input_source || "mic";
  elements.gain.value = audio.gain ?? 0;
  elements.sampleRateInput.value = audio.sample_rate ?? 48000;
  elements.channelsInput.value = audio.channels ?? 1;
  elements.framesPerChunk.value = audio.frames_per_chunk ?? 1024;
  elements.sampleWidthBytes.value = audio.sample_width_bytes ?? 2;
  elements.syntheticFrequencies.value = (audio.synthetic?.frequencies || []).join(",");
  elements.timeWindow.value = fig.time_window_ms;
  elements.maxPoints.value = fig.max_points;
  elements.refreshInterval.value = fig.refresh_interval_ms;
  elements.lineWidth.value = fig.line_width;
  elements.markerSize.value = fig.marker_size;
  elements.maxAmplitudeInput.value = fig.max_amplitude ?? 1.0;
  elements.fftTimeWindow.value = fig2.time_window_ms;
  elements.fftSize.value = fig2.size;
  elements.windowFunc.value = fig2.window_func || "rect";
  elements.fftScale.value = fig2.scale || "linear";
  elements.minFrequency.value = fig2.min_frequency;
  elements.maxFrequency.value = fig2.max_frequency;
  elements.smoothing.value = fig2.smoothing ?? 0;
  elements.fftRefreshInterval.value = fig2.refresh_interval_ms;
  elements.fftLineWidth.value = fig2.line_width;
  elements.fftMarkerSize.value = fig2.marker_size;
  elements.surfaceWindow.value = fig3.displayed_time_window_ms;
  elements.surfaceSegment.value = fig3.fft_time_segment_ms;
  if (elements.surfaceFftSize) elements.surfaceFftSize.value = fig3.fft_size ?? "";
  elements.surfaceTimeBins.value = fig3.time_bins;
  elements.surfaceRefreshInterval.value = fig3.refresh_interval_ms;
  elements.surfaceFrequencyBins.value = fig3.frequency_bins;
  elements.surfaceMinFrequency.value = fig3.min_frequency ?? 0;
  elements.surfaceMaxFrequency.value = fig3.max_frequency ?? 4000;
  elements.surfaceColormap.value = fig3.colormap || "inferno";
  if (elements.surfaceNormalize) {
    elements.surfaceNormalize.value = fig3.per_column_normalize === false ? "false" : "true";
  }
  elements.surfaceInterpolation.value = fig3.interpolation_type || "linear";
  updateSourceVisibility();
}

function captureParams() {
  const parseList = (value) =>
    value
      .split(",")
      .map((s) => parseFloat(s.trim()))
      .filter((v) => Number.isFinite(v) && v > 0);

  const configAudio = state.config.audio_input;
  const sampleRateHz = Number(elements.sampleRateInput.value) || configAudio.sample_rate;
  const channels = Number(elements.channelsInput.value) || configAudio.channels || 1;
  const framesPerChunk = Number(elements.framesPerChunk.value) || configAudio.frames_per_chunk;
  const sampleWidthBytes = Number(elements.sampleWidthBytes.value) || configAudio.sample_width_bytes;

  state.params.audioInput = {
    sample_rate: Math.max(8000, sampleRateHz),
    channels: Math.max(1, Math.min(2, channels)),
    frames_per_chunk: Math.max(128, framesPerChunk),
    sample_width_bytes: Math.max(1, Math.min(4, sampleWidthBytes)),
    input_source: elements.inputSource.value,
    synthetic: {
      frequencies: parseList(elements.syntheticFrequencies.value),
    },
    gain: Number(elements.gain.value) || 0,
  };

  state.params.figure1 = {
    time_window_ms: Math.max(10, Number(elements.timeWindow.value) || 1000),
    max_points: Math.max(16, Number(elements.maxPoints.value) || 1200),
    refresh_interval_ms: Math.max(5, Number(elements.refreshInterval.value) || 30),
    line_width: Math.max(0.5, Number(elements.lineWidth.value) || 1),
    marker_size: Math.max(0, Number(elements.markerSize.value) || 0),
    max_amplitude: Math.max(0.1, Number(elements.maxAmplitudeInput.value) || 1),
  };

  const bufferRate = state.params.audioInput.sample_rate;
  const fftSizeInput = Number(elements.fftSize.value) || state.config.figure2.size;
  const fftSize = nextPowerOfTwo(Math.max(256, fftSizeInput));
  if (elements.fftSize.value && Number(elements.fftSize.value) !== fftSize) {
    elements.fftSize.value = `${fftSize}`;
  }

  state.params.figure2 = {
    time_window_ms: Math.max(10, Number(elements.fftTimeWindow.value) || state.config.figure2.time_window_ms),
    size: fftSize,
    window_func: elements.windowFunc.value || "rect",
    scale: elements.fftScale.value || "linear",
    min_frequency: Math.max(0, Number(elements.minFrequency.value) || 0),
    max_frequency: Math.max(10, Number(elements.maxFrequency.value) || state.config.figure2.max_frequency),
    smoothing: Math.min(0.99, Math.max(0, Number(elements.smoothing.value) || 0)),
    refresh_interval_ms: Math.max(5, Number(elements.fftRefreshInterval.value) || 30),
    line_width: Math.max(0.5, Number(elements.fftLineWidth.value) || 1),
    marker_size: Math.max(0, Number(elements.fftMarkerSize.value) || 0),
  };
  if (state.params.figure2.max_frequency <= state.params.figure2.min_frequency) {
    state.params.figure2.max_frequency = state.params.figure2.min_frequency + 10;
  }

  const fig3Window = Math.max(100, Number(elements.surfaceWindow.value) || state.config.figure3.displayed_time_window_ms);
  const fig3Segment = Math.max(10, Number(elements.surfaceSegment.value) || state.config.figure3.fft_time_segment_ms);
  const fig3FftSizeInput = Number(elements.surfaceFftSize?.value) || state.config.figure3.fft_size;
  const fig3TimeBins = Math.max(20, Number(elements.surfaceTimeBins.value) || state.config.figure3.time_bins);
  const fig3FreqBins = Math.max(20, Number(elements.surfaceFrequencyBins.value) || state.config.figure3.frequency_bins);
  const fig3MinFreq = Math.max(0, Number(elements.surfaceMinFrequency.value) || 0);
  const fig3MaxFreq = Math.max(10, Number(elements.surfaceMaxFrequency.value) || state.config.figure3.max_frequency);
  const fig3Interpolation = elements.surfaceInterpolation.value || "linear";
  const fig3Colormap = elements.surfaceColormap.value || "inferno";
  const fig3Normalize = elements.surfaceNormalize
    ? elements.surfaceNormalize.value !== "false"
    : state.config.figure3.per_column_normalize !== false;
  const fig3DefaultFftSize = nextPowerOfTwo(Math.ceil((bufferRate * fig3Segment) / 1000));
  const fig3FftSize = nextPowerOfTwo(Math.max(256, fig3FftSizeInput || fig3DefaultFftSize));
  if (elements.surfaceFftSize && elements.surfaceFftSize.value && Number(elements.surfaceFftSize.value) !== fig3FftSize) {
    elements.surfaceFftSize.value = `${fig3FftSize}`;
  }

  state.params.figure3 = {
    displayed_time_window_ms: fig3Window,
    fft_time_segment_ms: fig3Segment,
    time_bins: fig3TimeBins,
    frequency_bins: fig3FreqBins,
    min_frequency: fig3MinFreq,
    max_frequency: fig3MaxFreq,
    colormap: fig3Colormap,
    per_column_normalize: fig3Normalize,
    refresh_interval_ms: Math.max(5, Number(elements.surfaceRefreshInterval.value) || state.config.figure3.refresh_interval_ms),
    interpolation_type: fig3Interpolation,
    fft_size: fig3FftSize,
  };
  if (state.params.figure3.max_frequency <= state.params.figure3.min_frequency) {
    state.params.figure3.max_frequency = state.params.figure3.min_frequency + 10;
    elements.surfaceMaxFrequency.value = `${state.params.figure3.max_frequency}`;
  }
  const fig1Samples = Math.ceil((bufferRate * state.params.figure1.time_window_ms) / 1000);
  const fig1Capacity = Math.max(fig1Samples + bufferRate, bufferRate * 2);
  if (!state.ringBufferFigure1) {
    state.ringBufferFigure1 = new CircularBuffer(fig1Capacity);
  } else {
    state.ringBufferFigure1.resize(fig1Capacity);
  }

  const fig2Samples = Math.ceil((bufferRate * state.params.figure2.time_window_ms) / 1000);
  const fig2Needed = Math.max(fig2Samples, state.params.figure2.size);
  const fig2Capacity = Math.max(fig2Needed + bufferRate, bufferRate * 2);
  if (!state.ringBufferFigure2) {
    state.ringBufferFigure2 = new CircularBuffer(fig2Capacity);
  } else {
    state.ringBufferFigure2.resize(fig2Capacity);
  }

  const fig3Samples = Math.ceil((bufferRate * state.params.figure3.displayed_time_window_ms) / 1000);
  const fig3Needed = Math.max(fig3Samples, state.params.figure3.fft_size);
  const fig3Capacity = Math.max(fig3Needed + bufferRate, bufferRate * 2);
  if (!state.ringBufferFigure3) {
    state.ringBufferFigure3 = new CircularBuffer(fig3Capacity);
  } else {
    state.ringBufferFigure3.resize(fig3Capacity);
  }

  ensureFigure3History();
  updateMeta();
  updateFFTMeta();
  updateSurfaceMeta();
  updateSourceVisibility();
  if (state.gainNode) {
    state.gainNode.gain.value = dbToGain(state.params.audioInput.gain);
  }
}

function updateMeta() {
  elements.figureMeta.textContent = "points: —";
}

function updateFFTMeta() {
  if (!elements.fftMeta) return;
  elements.fftMeta.textContent = "points: —";
}

function updateSurfaceMeta() {
  const fig = state.params.figure3;
  if (!elements.surfaceMeta) return;
  elements.surfaceMeta.textContent = "";
}

function ensureFigure3History() {
  const fig = state.params.figure3;
  const historyLength = Math.max(1, Math.round(fig.displayed_time_window_ms / fig.refresh_interval_ms));
  const signature = [
    historyLength,
    fig.frequency_bins,
    fig.displayed_time_window_ms,
    fig.fft_time_segment_ms,
    fig.refresh_interval_ms,
    fig.min_frequency,
    fig.max_frequency,
  ].join("|");
  const needsReset =
    state.figure3Signature !== signature ||
    state.figure3History.length !== historyLength ||
    (state.figure3History[0] && state.figure3History[0].length !== fig.frequency_bins);
  if (needsReset) {
    state.figure3History = Array.from({ length: historyLength }, () => new Float32Array(fig.frequency_bins));
    state.figure3LastSegmentTime = 0;
    state.figure3ImageData = null;
    state.figure3Signature = signature;
    state.figure3HistoryLength = historyLength;
  }
}

function nextPowerOfTwo(value) {
  const safe = Math.max(1, value);
  return 2 ** Math.ceil(Math.log2(safe));
}

function dbToGain(db) {
  return Math.pow(10, db / 20);
}

function ensureSurfaceLUT(colormap) {
  if (state.figure3ColorLUT && state.figure3ColorKey === colormap) {
    return state.figure3ColorLUT;
  }
  const stops = {
    viridis: [
      [0, 68, 1, 84],
      [0.25, 59, 82, 139],
      [0.5, 33, 145, 140],
      [0.75, 94, 201, 98],
      [1, 253, 231, 37],
    ],
    inferno: [
      [0, 0, 0, 4],
      [0.25, 87, 15, 109],
      [0.5, 187, 55, 84],
      [0.75, 249, 142, 8],
      [1, 252, 255, 164],
    ],
    plasma: [
      [0, 12, 7, 134],
      [0.3, 126, 3, 168],
      [0.6, 203, 71, 119],
      [0.8, 248, 149, 64],
      [1, 240, 249, 33],
    ],
    magma: [
      [0, 0, 0, 4],
      [0.25, 70, 17, 99],
      [0.5, 142, 43, 126],
      [0.75, 216, 86, 82],
      [1, 252, 253, 191],
    ],
    turbo: [
      [0, 48, 18, 59],
      [0.25, 38, 120, 191],
      [0.5, 75, 200, 101],
      [0.75, 237, 196, 62],
      [1, 180, 4, 38],
    ],
    ice: [
      [0, 6, 18, 38],
      [0.25, 18, 59, 120],
      [0.5, 25, 120, 180],
      [0.75, 120, 210, 236],
      [1, 235, 249, 255],
    ],
    mono: [
      [0, 15, 20, 30],
      [1, 225, 232, 240],
    ],
  };
  const palette = stops[colormap] || stops.inferno;
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let left = palette[0];
    let right = palette[palette.length - 1];
    for (let j = 0; j < palette.length - 1; j += 1) {
      if (t >= palette[j][0] && t <= palette[j + 1][0]) {
        left = palette[j];
        right = palette[j + 1];
        break;
      }
    }
    const span = right[0] - left[0] || 1;
    const localT = (t - left[0]) / span;
    const r = Math.round(left[1] + (right[1] - left[1]) * localT);
    const g = Math.round(left[2] + (right[2] - left[2]) * localT);
    const b = Math.round(left[3] + (right[3] - left[3]) * localT);
    const offset = i * 3;
    lut[offset] = r;
    lut[offset + 1] = g;
    lut[offset + 2] = b;
  }
  state.figure3ColorLUT = lut;
  state.figure3ColorKey = colormap;
  return lut;
}

function updateSourceVisibility() {
  const showSynthetic = elements.inputSource.value === "synthetic";
  if (elements.syntheticFrequencyControl) {
    elements.syntheticFrequencyControl.classList.toggle("hidden", !showSynthetic);
  }
}

async function setupWorklet() {
  const ctx = state.audioContext;
  if (!ctx?.audioWorklet) {
    throw new Error("AudioWorklet not supported in this browser.");
  }
  await ctx.audioWorklet.addModule("./audio-worklet-processor.js");
  const node = new AudioWorkletNode(ctx, "capture-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1,
    channelCountMode: "explicit",
    channelInterpretation: "speakers",
    processorOptions: {
      chunkSize: state.params.audioInput.frames_per_chunk,
    },
  });
  node.port.onmessage = (event) => {
    if (!state.running) return;
    const chunk = event.data;
    if (chunk && chunk.length) {
      if (state.ringBufferFigure1) state.ringBufferFigure1.push(chunk);
      if (state.ringBufferFigure2) state.ringBufferFigure2.push(chunk);
      if (state.ringBufferFigure3) state.ringBufferFigure3.push(chunk);
    }
  };
  state.workletNode = node;
}

async function start() {
  if (state.running) return;
  if (!state.userGestureReady) {
    setStatus("Awaiting user gesture to start audio.");
    elements.enableAudio?.classList.remove("hidden");
    setAwaitingGesture(true);
    return;
  }
  try {
    const firstLoad = !state.config;
    const config = state.config || (await fetchConfig());
    state.config = config;
    if (firstLoad) {
      syncUIFromConfig(config);
    }
    captureParams();
    if (state.params.audioInput.input_source === "mic" && !window.isSecureContext) {
      setStatus("Mic capture needs https/localhost; enable a secure context or switch to Synthetic.");
    }

    const ctx = new AudioContext({
      sampleRate: state.params.audioInput.sample_rate,
      latencyHint: "interactive",
    });
    state.audioContext = ctx;

    await ctx.resume();

    state.gainNode = ctx.createGain();
    state.gainNode.gain.value = dbToGain(state.params.audioInput.gain);
    await setupWorklet();
    state.gainNode.connect(state.workletNode);
    state.silentGainNode = ctx.createGain();
    state.silentGainNode.gain.value = 0;
    state.workletNode.connect(state.silentGainNode);
    state.silentGainNode.connect(ctx.destination);

    setStatus(state.params.audioInput.input_source === "mic" ? "Requesting microphone…" : "Starting synthetic audio…");
    await connectSource();

    state.running = true;
    setStatus("Streaming…");
    if (elements.sampleRate) elements.sampleRate.textContent = `${ctx.sampleRate}`;
    if (elements.channels) elements.channels.textContent = `${state.params.audioInput.channels}`;
    setStreamingIndicator(true);
    setAwaitingGesture(false);
    kickOffLoops();
  } catch (err) {
    console.error(err);
    setStreamingIndicator(false);
    if (String(err?.message || "").toLowerCase().includes("not allowed")) {
      state.userGestureReady = false;
      elements.enableAudio?.classList.remove("hidden");
      setAwaitingGesture(true);
      setStatus("Browser blocked audio auto-start. Click Enable Audio to resume.");
      return;
    }
    if (state.params?.audioInput?.input_source === "mic") {
      setStatus(`Mic failed: ${err.message || err}. Check permissions/https, or switch source to Synthetic.`);
      return;
    }
    const hint =
      state.params?.audioInput?.input_source === "mic"
        ? "Mic capture needs a secure context (https/localhost) and browser support. You can switch source to Synthetic."
        : "";
    setStatus(`${err.message || "Failed to start audio."} ${hint}`.trim());
  }
}

function stop() {
  state.running = false;
  if (state.drawLoopId) {
    cancelAnimationFrame(state.drawLoopId);
    state.drawLoopId = null;
  }
  state.lastDrawTimeFigure1 = 0;
  state.lastDrawTimeFigure2 = 0;
  state.lastDrawTimeFigure3 = 0;
  state.figure3LastSegmentTime = 0;
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
  state.syntheticOscillators.forEach((osc) => {
    try {
      osc.stop();
    } catch (_) {
      /* ignore */
    }
  });
  state.syntheticOscillators = [];
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }
  if (state.workletNode) {
    try {
      state.workletNode.port.onmessage = null;
      state.workletNode.disconnect();
    } catch (_) {
      /* ignore */
    }
    state.workletNode = null;
  }
  if (state.silentGainNode) {
    try {
      state.silentGainNode.disconnect();
    } catch (_) {
      /* ignore */
    }
    state.silentGainNode = null;
  }
  state.gainNode = null;
  state.sourceNode = null;
  setStatus("Stopped.");
  setStreamingIndicator(false);
}

async function connectSource() {
  const { input_source } = state.params.audioInput;
  if (input_source === "synthetic") {
    connectSynthetic();
  } else {
    await connectMic();
  }
}

async function connectMic() {
  const audio = state.params.audioInput;
  if (!window.isSecureContext) {
    throw new Error("Mic capture requires a secure context (https or localhost).");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia not supported in this browser.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: { ideal: audio.channels },
      sampleRate: { ideal: audio.sample_rate },
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  state.mediaStream = stream;
  const ctx = state.audioContext;
  state.sourceNode = ctx.createMediaStreamSource(stream);
  state.sourceNode.connect(state.gainNode);
}

function connectSynthetic() {
  const ctx = state.audioContext;
  const { frequencies } = state.params.audioInput.synthetic;
  const gainNode = state.gainNode;
  gainNode.gain.value = dbToGain(state.params.audioInput.gain);
  const oscillators = frequencies.length ? frequencies : [220, 440, 880];
  oscillators.forEach((freq) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = freq;
    osc.connect(gainNode);
    osc.start();
    state.syntheticOscillators.push(osc);
  });
  gainNode.connect(ctx.destination);
}

function kickOffLoops() {
  const drawTick = (timestamp) => {
    if (!state.running) return;
    const fig1Interval = state.params.figure1.refresh_interval_ms;
    const fig2Interval = state.params.figure2.refresh_interval_ms;
    const fig3Interval = state.params.figure3.refresh_interval_ms;

    if (!state.lastDrawTimeFigure1) {
      state.lastDrawTimeFigure1 = timestamp - fig1Interval;
    }
    if (timestamp - state.lastDrawTimeFigure1 >= fig1Interval) {
      drawSignal();
      state.lastDrawTimeFigure1 = timestamp;
    }

    if (!state.lastDrawTimeFigure2) {
      state.lastDrawTimeFigure2 = timestamp - fig2Interval;
    }
    if (timestamp - state.lastDrawTimeFigure2 >= fig2Interval) {
      drawSpectrum();
      state.lastDrawTimeFigure2 = timestamp;
    }

    if (!state.lastDrawTimeFigure3) {
      state.lastDrawTimeFigure3 = timestamp - fig3Interval;
    }
    if (timestamp - state.lastDrawTimeFigure3 >= fig3Interval) {
      drawSurface();
      state.lastDrawTimeFigure3 = timestamp;
    }
    state.drawLoopId = requestAnimationFrame(drawTick);
  };
  state.drawLoopId = requestAnimationFrame(drawTick);
}

function drawSignal() {
  const canvas = elements.canvas;
  const ctx2d = canvas.getContext("2d");
  const { width, height } = canvas;

  ctx2d.fillStyle = CANVAS_BG;
  ctx2d.fillRect(0, 0, width, height);

  if (!state.ringBufferFigure1 || !state.audioContext) {
    ctx2d.fillStyle = "#9ba6b5";
    ctx2d.fillText("Awaiting audio…", 12, 24);
    return;
  }

  const fig = state.params.figure1;
  const sampleRate = state.audioContext.sampleRate || state.params.audioInput.sample_rate;
  const samplesNeeded = Math.ceil((sampleRate * fig.time_window_ms) / 1000);
  const samples = state.ringBufferFigure1.toArray(samplesNeeded);
  if (!samples.length) {
    ctx2d.fillStyle = "#9ba6b5";
    ctx2d.fillText("No samples yet.", 12, 24);
    return;
  }

  const targetPoints = Math.max(2, Math.min(fig.max_points, samples.length));
  if (elements.figureMeta) {
    elements.figureMeta.textContent = `points: ${targetPoints}`;
  }
  const stride = samples.length / targetPoints;
  const points = new Array(targetPoints);
  for (let i = 0; i < targetPoints; i += 1) {
    const value = samples[Math.floor(i * stride)];
    points[i] = value;
  }

  const padding = {
    left: 170,
    right: 32,
    top: 32,
    bottom: 120,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const mid = padding.top + plotHeight / 2;
  const amplitudeScale = plotHeight / 2 / fig.max_amplitude;

  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.35)";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(padding.left, padding.top);
  ctx2d.lineTo(padding.left, padding.top + plotHeight);
  ctx2d.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx2d.stroke();

  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.2)";
  ctx2d.beginPath();
  ctx2d.moveTo(padding.left, mid);
  ctx2d.lineTo(padding.left + plotWidth, mid);
  ctx2d.stroke();

  ctx2d.fillStyle = "rgba(155, 166, 181, 0.9)";
  ctx2d.font = plotFont();
  ctx2d.save();
  ctx2d.translate(40, padding.top + plotHeight / 2);
  ctx2d.rotate(-Math.PI / 2);
  ctx2d.fillText("Amplitude", 0, 0);
  ctx2d.restore();
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "bottom";
  ctx2d.fillText("Time (ms)", padding.left + plotWidth / 2, padding.top + plotHeight + 90);
  ctx2d.font = plotFont();
  const tickCount = 5;
  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.45)";
  ctx2d.fillStyle = "rgba(155, 166, 181, 0.85)";
  ctx2d.textAlign = "right";
  ctx2d.textBaseline = "middle";
  for (let i = 0; i < tickCount; i += 1) {
    const t = i / (tickCount - 1);
    const value = -fig.max_amplitude + t * (fig.max_amplitude * 2);
    const y = mid - value * amplitudeScale;
    ctx2d.beginPath();
    ctx2d.moveTo(padding.left - 8, y);
    ctx2d.lineTo(padding.left, y);
    ctx2d.stroke();
    ctx2d.fillText(value.toFixed(2), padding.left - 12, y);
  }
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "top";
  for (let i = 0; i < tickCount; i += 1) {
    const t = i / (tickCount - 1);
    const x = padding.left + t * plotWidth;
    const value = -fig.time_window_ms + t * fig.time_window_ms;
    const y = padding.top + plotHeight;
    ctx2d.beginPath();
    ctx2d.moveTo(x, y);
    ctx2d.lineTo(x, y + 8);
    ctx2d.stroke();
    ctx2d.fillText(Math.round(value), x, y + 36);
  }

  ctx2d.lineWidth = fig.line_width;
  ctx2d.strokeStyle = "#6dd5ed";
  ctx2d.beginPath();
  for (let i = 0; i < points.length; i += 1) {
    const x = padding.left + (i / (points.length - 1)) * plotWidth;
    const y = mid - points[i] * amplitudeScale;
    if (i === 0) {
      ctx2d.moveTo(x, y);
    } else {
      ctx2d.lineTo(x, y);
    }
  }
  ctx2d.stroke();

  if (fig.marker_size > 0) {
    ctx2d.fillStyle = "#2193b0";
    for (let i = 0; i < points.length; i += Math.max(1, Math.floor(points.length / 50))) {
      const x = padding.left + (i / (points.length - 1)) * plotWidth;
      const y = mid - points[i] * amplitudeScale;
      ctx2d.beginPath();
      ctx2d.arc(x, y, fig.marker_size, 0, Math.PI * 2);
      ctx2d.fill();
    }
  }
}

function ensureFFTCache(size, windowFunc, cache = state.figure2Cache) {
  if (cache.size !== size || !cache.real || !cache.imag) {
    cache.size = size;
    cache.real = new Float32Array(size);
    cache.imag = new Float32Array(size);
    if ("smoothed" in cache) cache.smoothed = new Float32Array(size / 2);
    if ("magnitudes" in cache) cache.magnitudes = new Float32Array(size / 2);
  }
  if (!cache.window || cache.window.length !== size || cache.windowFunc !== windowFunc) {
    const window = new Float32Array(size);
    if (windowFunc === "hann") {
      for (let i = 0; i < size; i += 1) {
        window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
      }
    } else if (windowFunc === "hamming") {
      for (let i = 0; i < size; i += 1) {
        window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (size - 1));
      }
    } else if (windowFunc === "blackman") {
      for (let i = 0; i < size; i += 1) {
        window[i] =
          0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (size - 1));
      }
    } else {
      window.fill(1);
    }
    cache.window = window;
    cache.windowFunc = windowFunc;
  }
  return cache;
}

function fftRadix2(real, imag) {
  const n = real.length;
  const levels = Math.log2(n);
  if (Math.floor(levels) !== levels) {
    throw new Error("FFT size must be a power of two.");
  }

  for (let i = 0; i < n; i += 1) {
    let j = 0;
    for (let bit = 0; bit < levels; bit += 1) {
      j = (j << 1) | ((i >>> bit) & 1);
    }
    if (j > i) {
      const tempReal = real[i];
      real[i] = real[j];
      real[j] = tempReal;
      const tempImag = imag[i];
      imag[i] = imag[j];
      imag[j] = tempImag;
    }
  }

  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const tableStep = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j += 1) {
        const angle = tableStep * j;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const k = i + j + halfSize;
        const tpre = real[k] * cos - imag[k] * sin;
        const tpim = real[k] * sin + imag[k] * cos;
        real[k] = real[i + j] - tpre;
        imag[k] = imag[i + j] - tpim;
        real[i + j] += tpre;
        imag[i + j] += tpim;
      }
    }
  }
}

function drawSpectrum() {
  const canvas = elements.fftCanvas;
  if (!canvas) return;
  const ctx2d = canvas.getContext("2d");
  const { width, height } = canvas;

  ctx2d.fillStyle = CANVAS_BG;
  ctx2d.fillRect(0, 0, width, height);

  if (!state.ringBufferFigure2 || !state.audioContext) {
    ctx2d.fillStyle = "#9ba6b5";
    ctx2d.fillText("Awaiting audio…", 12, 24);
    return;
  }

  const fig = state.params.figure2;
  const sampleRate = state.audioContext.sampleRate || state.params.audioInput.sample_rate;
  const timeSamples = Math.ceil((sampleRate * fig.time_window_ms) / 1000);
  const windowSamples = Math.min(timeSamples, fig.size);
  const samples = state.ringBufferFigure2.toArray(windowSamples);
  if (!samples.length) {
    ctx2d.fillStyle = "#9ba6b5";
    ctx2d.fillText("No samples yet.", 12, 24);
    return;
  }

  const cache = ensureFFTCache(fig.size, fig.window_func);
  const real = cache.real;
  const imag = cache.imag;
  real.fill(0);
  imag.fill(0);
  const offset = fig.size - samples.length;
  for (let i = 0; i < samples.length; i += 1) {
    const idx = offset + i;
    real[idx] = samples[i] * cache.window[idx];
  }

  fftRadix2(real, imag);

  const half = fig.size / 2;
  const nyquist = sampleRate / 2;
  const scale = fig.scale || "linear";
  const minFreq = Math.max(0, Math.min(fig.min_frequency, fig.max_frequency));
  const maxFreq = Math.max(minFreq + 1, Math.min(nyquist, Math.max(fig.min_frequency, fig.max_frequency)));
  const minBin = Math.max(0, Math.floor((minFreq / sampleRate) * fig.size));
  const maxBin = Math.min(half, Math.ceil((maxFreq / sampleRate) * fig.size));
  if (elements.fftMeta) {
    const points = Math.max(0, maxBin - minBin + 1);
    elements.fftMeta.textContent = `points: ${points}`;
  }

  const smoothing = fig.smoothing;
  const smoothed = cache.smoothed;
  let maxMag = 0;
  for (let i = 0; i < half; i += 1) {
    const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) / fig.size;
    const value = smoothing * smoothed[i] + (1 - smoothing) * mag;
    smoothed[i] = value;
    if (i >= minBin && i <= maxBin && value > maxMag) {
      maxMag = value;
    }
  }
  if (maxMag === 0) maxMag = 1;

  const padding = {
    left: 170,
    right: 32,
    top: 32,
    bottom: 120,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.35)";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(padding.left, padding.top);
  ctx2d.lineTo(padding.left, padding.top + plotHeight);
  ctx2d.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx2d.stroke();

  ctx2d.fillStyle = "rgba(155, 166, 181, 0.9)";
  ctx2d.font = plotFont();
  ctx2d.save();
  ctx2d.translate(40, padding.top + plotHeight / 2);
  ctx2d.rotate(-Math.PI / 2);
  ctx2d.fillText(scale === "log" ? "Magnitude (dB)" : "Magnitude", 0, 0);
  ctx2d.restore();
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "bottom";
  ctx2d.fillText("Frequency (Hz)", padding.left + plotWidth / 2, padding.top + plotHeight + 90);

  ctx2d.font = plotFont();
  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.45)";
  ctx2d.fillStyle = "rgba(155, 166, 181, 0.85)";
  ctx2d.textAlign = "right";
  ctx2d.textBaseline = "middle";
  if (scale === "log") {
    const ticks = [0, -20, -40, -60, -80];
    for (const tick of ticks) {
      const y = padding.top + plotHeight - ((tick - minDb) / dbSpan) * plotHeight;
      ctx2d.beginPath();
      ctx2d.moveTo(padding.left - 8, y);
      ctx2d.lineTo(padding.left, y);
      ctx2d.stroke();
      ctx2d.fillText(`${tick}`, padding.left - 12, y);
    }
  } else {
    const tickCount = 5;
    for (let i = 0; i < tickCount; i += 1) {
      const t = i / (tickCount - 1);
      const value = t * maxMag;
      const y = padding.top + plotHeight - t * plotHeight;
      ctx2d.beginPath();
      ctx2d.moveTo(padding.left - 8, y);
      ctx2d.lineTo(padding.left, y);
      ctx2d.stroke();
      ctx2d.fillText(value.toFixed(2), padding.left - 12, y);
    }
  }

  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "top";
  const tickCount = 5;
  for (let i = 0; i < tickCount; i += 1) {
    const t = i / (tickCount - 1);
    const freq = minFreq + t * (maxFreq - minFreq);
    const x = padding.left + t * plotWidth;
    const y = padding.top + plotHeight;
    ctx2d.beginPath();
    ctx2d.moveTo(x, y);
    ctx2d.lineTo(x, y + 8);
    ctx2d.stroke();
    ctx2d.fillText(Math.round(freq), x, y + 36);
  }

  const span = Math.max(1, maxBin - minBin);
  ctx2d.lineWidth = fig.line_width;
  ctx2d.strokeStyle = "#6dd5ed";
  ctx2d.beginPath();
  let first = true;
  const epsilon = 1e-12;
  const maxDb = 20 * Math.log10(maxMag + epsilon);
  const minDb = maxDb - 80;
  const dbSpan = maxDb - minDb || 1;
  for (let bin = minBin; bin <= maxBin; bin += 1) {
    const x = padding.left + ((bin - minBin) / span) * plotWidth;
    const magnitude = smoothed[bin];
    let y;
    if (scale === "log") {
      const db = 20 * Math.log10(magnitude + epsilon);
      y = padding.top + plotHeight - ((db - minDb) / dbSpan) * plotHeight;
    } else {
      y = padding.top + plotHeight - (magnitude / maxMag) * plotHeight;
    }
    if (first) {
      ctx2d.moveTo(x, y);
      first = false;
    } else {
      ctx2d.lineTo(x, y);
    }
  }
  ctx2d.stroke();

  if (fig.marker_size > 0) {
    ctx2d.fillStyle = "#2193b0";
    for (let bin = minBin; bin <= maxBin; bin += Math.max(1, Math.floor(span / 64))) {
      const x = padding.left + ((bin - minBin) / span) * plotWidth;
      const magnitude = smoothed[bin];
      let y;
      if (scale === "log") {
        const db = 20 * Math.log10(magnitude + epsilon);
        y = padding.top + plotHeight - ((db - minDb) / dbSpan) * plotHeight;
      } else {
        y = padding.top + plotHeight - (magnitude / maxMag) * plotHeight;
      }
      ctx2d.beginPath();
      ctx2d.arc(x, y, fig.marker_size, 0, Math.PI * 2);
      ctx2d.fill();
    }
  }

}

function computeSurfaceColumn() {
  if (!state.ringBufferFigure3 || !state.audioContext) return null;
  const fig = state.params.figure3;
  const sampleRate = state.audioContext.sampleRate || state.params.audioInput.sample_rate;
  const nyquist = sampleRate / 2;
  const minFreq = Math.max(0, Math.min(fig.min_frequency, fig.max_frequency));
  const maxFreq = Math.max(minFreq + 1, Math.min(nyquist, Math.max(fig.min_frequency, fig.max_frequency)));
  const samples = state.ringBufferFigure3.toArray(fig.fft_size);
  if (samples.length < fig.fft_size) return null;

  const cache = ensureFFTCache(fig.fft_size, "rect", state.figure3Cache);
  const real = cache.real;
  const imag = cache.imag;
  const magnitudes = cache.magnitudes;
  const window = cache.window;
  real.fill(0);
  imag.fill(0);
  for (let i = 0; i < fig.fft_size; i += 1) {
    real[i] = samples[i] * window[i];
  }

  fftRadix2(real, imag);

  const half = fig.fft_size / 2;
  for (let i = 0; i < half; i += 1) {
    magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) / fig.fft_size;
  }

  const freqBins = fig.frequency_bins;
  const output = new Float32Array(freqBins);
  const useLinear = ["linear", "quadratic", "spline"].includes(fig.interpolation_type);
  const denom = freqBins > 1 ? freqBins - 1 : 1;
  for (let i = 0; i < freqBins; i += 1) {
    const freq = minFreq + (i / denom) * (maxFreq - minFreq);
    const bin = (freq / sampleRate) * fig.fft_size;
    const clamped = Math.min(half - 1, Math.max(0, bin));
    if (useLinear) {
      const lo = Math.floor(clamped);
      const hi = Math.min(half - 1, lo + 1);
      const frac = clamped - lo;
      output[i] = magnitudes[lo] * (1 - frac) + magnitudes[hi] * frac;
    } else {
      output[i] = magnitudes[Math.round(clamped)];
    }
  }
  return output;
}

function drawSurface() {
  const canvas = elements.surfaceCanvas;
  if (!canvas) return;
  const ctx2d = canvas.getContext("2d");
  const { width, height } = canvas;

  ctx2d.fillStyle = CANVAS_BG;
  ctx2d.fillRect(0, 0, width, height);

  if (!state.ringBufferFigure3 || !state.audioContext) {
    ctx2d.fillStyle = "#9ba6b5";
    ctx2d.fillText("Awaiting audio…", 12, 24);
    return;
  }

  const fig = state.params.figure3;
  const sampleRate = state.audioContext.sampleRate || state.params.audioInput.sample_rate;
  const nyquist = sampleRate / 2;
  const minFreq = Math.max(0, Math.min(fig.min_frequency, fig.max_frequency));
  const maxFreq = Math.max(minFreq + 1, Math.min(nyquist, Math.max(fig.min_frequency, fig.max_frequency)));
  ensureFigure3History();
  const now = performance.now();
  const stepMs = fig.refresh_interval_ms;
  if (!state.figure3LastSegmentTime) {
    state.figure3LastSegmentTime = now - stepMs;
  }

  let iterations = 0;
  while (now - state.figure3LastSegmentTime >= stepMs && iterations < 4) {
    const column = computeSurfaceColumn();
    if (!column) break;
    state.figure3History.shift();
    state.figure3History.push(column);
    state.figure3LastSegmentTime += stepMs;
    iterations += 1;
  }

  let globalMax = 0;
  const historyLen = state.figure3History.length || 1;
  const columnMaxes = new Float32Array(historyLen);
  for (let c = 0; c < historyLen; c += 1) {
    const column = state.figure3History[c];
    let colMax = 0;
    if (column) {
      for (let i = 0; i < column.length; i += 1) {
        if (column[i] > colMax) colMax = column[i];
      }
    }
    columnMaxes[c] = colMax;
    if (colMax > globalMax) globalMax = colMax;
  }
  if (globalMax === 0) globalMax = 1;

  const padding = {
    left: 170,
    right: 32,
    top: 32,
    bottom: 120,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const timeBins = fig.time_bins;
  const freqBins = fig.frequency_bins;
  if (!state.figure3ImageData || state.figure3ImageData.width !== timeBins || state.figure3ImageData.height !== freqBins) {
    const imageCanvas = state.figure3ImageCanvas || document.createElement("canvas");
    imageCanvas.width = timeBins;
    imageCanvas.height = freqBins;
    state.figure3ImageCanvas = imageCanvas;
    const imageCtx = imageCanvas.getContext("2d");
    state.figure3ImageData = imageCtx.createImageData(timeBins, freqBins);
  }

  const lut = ensureSurfaceLUT(fig.colormap);
  const data = state.figure3ImageData.data;
  let offset = 0;
  const timeDenom = timeBins > 1 ? timeBins - 1 : 1;
  for (let y = 0; y < freqBins; y += 1) {
    const bin = freqBins - 1 - y;
    for (let x = 0; x < timeBins; x += 1) {
      const t = x / timeDenom;
      const historyPos = t * (historyLen - 1);
      const left = Math.floor(historyPos);
      const right = Math.min(historyLen - 1, left + 1);
      const frac = historyPos - left;
      const leftVal = state.figure3History[left]?.[bin] ?? 0;
      const rightVal = state.figure3History[right]?.[bin] ?? 0;
      const value = leftVal * (1 - frac) + rightVal * frac;
      let normalized;
      if (fig.per_column_normalize) {
        const leftMax = columnMaxes[left] || 0;
        const rightMax = columnMaxes[right] || 0;
        const columnMax = (leftMax * (1 - frac) + rightMax * frac) || 1;
        normalized = Math.min(1, Math.max(0, value / columnMax));
      } else {
        normalized = Math.min(1, Math.max(0, value / globalMax));
      }
      const index = Math.min(255, Math.floor(normalized * 255));
      const lutOffset = index * 3;
      data[offset] = lut[lutOffset];
      data[offset + 1] = lut[lutOffset + 1];
      data[offset + 2] = lut[lutOffset + 2];
      data[offset + 3] = 255;
      offset += 4;
    }
  }

  const imageCanvas = state.figure3ImageCanvas;
  const imageCtx = imageCanvas.getContext("2d");
  imageCtx.putImageData(state.figure3ImageData, 0, 0);
  ctx2d.imageSmoothingEnabled = true;
  ctx2d.drawImage(imageCanvas, padding.left, padding.top, plotWidth, plotHeight);

  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.35)";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(padding.left, padding.top);
  ctx2d.lineTo(padding.left, padding.top + plotHeight);
  ctx2d.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx2d.stroke();

  ctx2d.fillStyle = "rgba(155, 166, 181, 0.9)";
  ctx2d.font = plotFont();
  ctx2d.save();
  ctx2d.translate(40, padding.top + plotHeight / 2);
  ctx2d.rotate(-Math.PI / 2);
  ctx2d.fillText("Frequency (Hz)", 0, 0);
  ctx2d.restore();
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "bottom";
  ctx2d.fillText("Time (ms)", padding.left + plotWidth / 2, padding.top + plotHeight + 90);

  ctx2d.font = plotFont();
  ctx2d.strokeStyle = "rgba(155, 166, 181, 0.45)";
  ctx2d.fillStyle = "rgba(155, 166, 181, 0.85)";
  ctx2d.textAlign = "right";
  ctx2d.textBaseline = "middle";
  const freqTicks = 5;
  for (let i = 0; i < freqTicks; i += 1) {
    const t = i / (freqTicks - 1);
    const value = minFreq + t * (maxFreq - minFreq);
    const y = padding.top + plotHeight - t * plotHeight;
    ctx2d.beginPath();
    ctx2d.moveTo(padding.left - 8, y);
    ctx2d.lineTo(padding.left, y);
    ctx2d.stroke();
    ctx2d.fillText(Math.round(value), padding.left - 12, y);
  }

  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "top";
  const timeTicks = 5;
  for (let i = 0; i < timeTicks; i += 1) {
    const t = i / (timeTicks - 1);
    const value = -fig.displayed_time_window_ms + t * fig.displayed_time_window_ms;
    const x = padding.left + t * plotWidth;
    const y = padding.top + plotHeight;
    ctx2d.beginPath();
    ctx2d.moveTo(x, y);
    ctx2d.lineTo(x, y + 8);
    ctx2d.stroke();
    ctx2d.fillText(Math.round(value), x, y + 36);
  }
}

function bindUI() {
  const restartAudio = () => {
    if (state.restartTimer) clearTimeout(state.restartTimer);
    state.restartTimer = setTimeout(async () => {
      state.restartTimer = null;
      const wasRunning = state.running;
      if (wasRunning) stop();
      await start();
    }, 80);
  };

  elements.enableAudio?.addEventListener("click", async () => {
    state.userGestureReady = true;
    elements.enableAudio.classList.add("hidden");
    setAwaitingGesture(false);
    await start();
  });

  [
    elements.inputSource,
    elements.gain,
    elements.sampleRateInput,
    elements.channelsInput,
    elements.framesPerChunk,
    elements.sampleWidthBytes,
    elements.syntheticFrequencies,
    elements.windowFunc,
    elements.fftScale,
    elements.surfaceInterpolation,
    elements.surfaceColormap,
  ]
    .filter(Boolean)
    .forEach((el) =>
    el.addEventListener("change", () => {
      captureParams();
      restartAudio();
    })
  );

  [
    elements.timeWindow,
    elements.maxPoints,
    elements.refreshInterval,
    elements.lineWidth,
    elements.markerSize,
    elements.maxAmplitudeInput,
    elements.fftTimeWindow,
    elements.fftSize,
    elements.minFrequency,
    elements.maxFrequency,
    elements.smoothing,
    elements.fftRefreshInterval,
    elements.fftLineWidth,
    elements.fftMarkerSize,
    elements.surfaceWindow,
    elements.surfaceSegment,
    elements.surfaceFftSize,
    elements.surfaceTimeBins,
    elements.surfaceFrequencyBins,
    elements.surfaceRefreshInterval,
    elements.surfaceMinFrequency,
    elements.surfaceMaxFrequency,
  ]
    .filter(Boolean)
    .forEach((el) =>
    el.addEventListener("input", () => {
      captureParams();
      restartAudio();
    })
  );
}

async function bootstrap() {
  try {
    const config = await fetchConfig();
    state.config = config;
    syncUIFromConfig(config);
    captureParams();
    setStatus("Config loaded. Click Enable Audio to start.");
    elements.enableAudio?.classList.remove("hidden");
    setAwaitingGesture(true);
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Failed to load config.");
  }
}

bindUI();
bootstrap();
