const statusBar = document.getElementById('status');
const applyBtn = document.getElementById('applyBtn');
const signalCanvas = document.getElementById('signalCanvas');
const fftCanvas = document.getElementById('fftCanvas');
const channelSelectContainer = document.getElementById('channelSelect');
const channelsInput = document.getElementById('channels');
const timeWindowInput = document.getElementById('timeWindow');
const timeWindowResolutionLabel = document.getElementById('timeWindowResolution');
const refreshIntervalInput = document.getElementById('refreshInterval');
const signalPointLimitInput = document.getElementById('signalPointLimit');
const minFreqInput = document.getElementById('minFreq');
const maxFreqInput = document.getElementById('maxFreq');
const signalLineWidthInput = document.getElementById('signalLineWidth');
const signalMarkerSizeInput = document.getElementById('signalMarkerSize');
const fftLineWidthInput = document.getElementById('fftLineWidth');
const fftMarkerSizeInput = document.getElementById('fftMarkerSize');
const surfacePlotWindowInput = document.getElementById('surfacePlotWindow');
const surfaceFreqBinsInput = document.getElementById('surfaceFreqBins');
const surfaceTimeBinsInput = document.getElementById('surfaceTimeBins');
const surfacePublishIntervalInput = document.getElementById('surfacePublishInterval');
const surfaceCanvas = document.getElementById('surfaceCanvas');
const surfaceColorProfileInput = document.getElementById('surfaceColorProfile');

let selectedChannels = new Set(['Channel 1']);
let liveChannelCount = Number(channelsInput.value) || 1;
let socket;
let reconnectTimer;
let latestSignal = null;
let latestFft = null;
let hasHydrated = false;
let latestFrameTimestamp = Date.now();
let surfaceFrames = [];
let surfaceGrid = { history: [], frequency: [] };
let surfacePublishTimer = null;
let refreshQueued = false;
let surfaceBuildInFlight = false;
let surfaceConfigTimer = null;
const PLOT_BOTTOM_PADDING = 50;
const CHANNEL_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f97316', '#f472b6', '#facc15', '#f87171', '#0ea5e9'];
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const vocalHelp = document.getElementById('vocalHelp');
const DEFAULT_SURFACE_PROFILE = 'aurora';
const SURFACE_COLOR_PROFILES = {
  aurora: [
    { stop: 0, color: [15, 23, 42] },
    { stop: 0.35, color: [56, 189, 248] },
    { stop: 1, color: [244, 114, 182] },
  ],
  inferno: [
    { stop: 0, color: [0, 0, 0] },
    { stop: 0.4, color: [165, 21, 21] },
    { stop: 0.7, color: [249, 115, 22] },
    { stop: 1, color: [253, 224, 71] },
  ],
  ice: [
    { stop: 0, color: [5, 0, 37] },
    { stop: 0.5, color: [6, 182, 212] },
    { stop: 1, color: [224, 242, 254] },
  ],
};
const CHANNEL_RGB = CHANNEL_COLORS.map((hex) => {
  const safe = hex.replace('#', '');
  const value = parseInt(safe, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
});

function updateTimeWindowResolution() {
  if (!timeWindowResolutionLabel) return;
  const durationMs = Number(timeWindowInput.value);
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    timeWindowResolutionLabel.textContent = '--';
    return;
  }
  const hz = 1000 / durationMs;
  const decimals = hz >= 10 ? 1 : 2;
  timeWindowResolutionLabel.textContent = `${hz.toFixed(decimals)} Hz`;
}

function setStatus(message, tone = 'info') {
  statusBar.textContent = message;
  statusBar.dataset.tone = tone;
}

function channelFill(index, alpha) {
  const base = CHANNEL_RGB[index % CHANNEL_RGB.length];
  const clamped = Math.max(0, Math.min(1, alpha));
  return `rgba(${base.r}, ${base.g}, ${base.b}, ${clamped})`;
}

function getSurfacePalette() {
  const key = (surfaceColorProfileInput && surfaceColorProfileInput.value) || DEFAULT_SURFACE_PROFILE;
  if (SURFACE_COLOR_PROFILES[key]) return { key, stops: SURFACE_COLOR_PROFILES[key] };
  return { key: DEFAULT_SURFACE_PROFILE, stops: SURFACE_COLOR_PROFILES[DEFAULT_SURFACE_PROFILE] };
}

function sampleGradientColor(stops, value) {
  if (!stops || !stops.length) return `rgba(255,255,255,${Math.max(0, Math.min(1, value))})`;
  const clamped = Math.max(0, Math.min(1, value));
  if (stops.length === 1) {
    const [r, g, b] = stops[0].color;
    return `rgb(${r}, ${g}, ${b})`;
  }
  let prev = stops[0];
  for (let i = 1; i < stops.length; i += 1) {
    const curr = stops[i];
    if (clamped <= curr.stop || i === stops.length - 1) {
      const span = Math.max(1e-6, curr.stop - prev.stop);
      const localT = Math.max(0, Math.min(1, (clamped - prev.stop) / span));
      const [pr, pg, pb] = prev.color;
      const [cr, cg, cb] = curr.color;
      const r = Math.round(pr + (cr - pr) * localT);
      const g = Math.round(pg + (cg - pg) * localT);
      const b = Math.round(pb + (cb - pb) * localT);
      return `rgb(${r}, ${g}, ${b})`;
    }
    prev = curr;
  }
  const [r, g, b] = stops[stops.length - 1].color;
  return `rgb(${r}, ${g}, ${b})`;
}

function buildChannelTags(totalOverride, autoFill = false) {
  const baseValue = totalOverride !== undefined && totalOverride !== null
    ? totalOverride
    : channelsInput.value;
  const total = Math.max(1, Number(baseValue) || 1);
  liveChannelCount = total;
  if (autoFill) {
    selectedChannels = new Set(Array.from({ length: total }, (_, i) => `Channel ${i + 1}`));
  } else {
    selectedChannels = new Set(
      Array.from(selectedChannels).filter((label) => {
        const idx = parseInt(label.replace('Channel', '').trim(), 10) - 1;
        return Number.isInteger(idx) && idx >= 0 && idx < total;
      }),
    );
  }
  channelSelectContainer.innerHTML = '';
  for (let i = 1; i <= total; i += 1) {
    const channelName = `Channel ${i}`;
    const wrapper = document.createElement('label');
    wrapper.className = 'channel-checkbox';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selectedChannels.has(channelName);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        selectedChannels.add(channelName);
      } else {
        selectedChannels.delete(channelName);
      }
      refreshPlots();
    });
    const colorDot = document.createElement('span');
    colorDot.style.background = CHANNEL_COLORS[(i - 1) % CHANNEL_COLORS.length];
    colorDot.style.width = '12px';
    colorDot.style.height = '12px';
    colorDot.style.borderRadius = '50%';
    const labelText = document.createElement('span');
    labelText.textContent = channelName;
    wrapper.appendChild(checkbox);
    wrapper.appendChild(colorDot);
    wrapper.appendChild(labelText);
    channelSelectContainer.appendChild(wrapper);
  }
}

channelsInput.addEventListener('input', () => {
  buildChannelTags(Number(channelsInput.value), true);
  refreshPlots();
});
timeWindowInput.addEventListener('input', () => {
  updateTimeWindowResolution();
  refreshPlots();
});
signalPointLimitInput.addEventListener('input', refreshPlots);
minFreqInput.addEventListener('input', refreshPlots);
maxFreqInput.addEventListener('input', refreshPlots);
signalLineWidthInput.addEventListener('input', refreshPlots);
signalMarkerSizeInput.addEventListener('input', refreshPlots);
fftLineWidthInput.addEventListener('input', refreshPlots);
fftMarkerSizeInput.addEventListener('input', refreshPlots);
const scheduleSurfaceConfigApply = () => {
  if (surfaceConfigTimer) {
    clearTimeout(surfaceConfigTimer);
  }
  surfaceConfigTimer = setTimeout(() => {
    surfaceConfigTimer = null;
    trimSurfaceFrames();
    restartSurfacePublish();
    queueRefresh();
  }, 120);
};
surfacePlotWindowInput.addEventListener('input', scheduleSurfaceConfigApply);
surfaceFreqBinsInput.addEventListener('input', scheduleSurfaceConfigApply);
surfaceTimeBinsInput.addEventListener('input', scheduleSurfaceConfigApply);
surfacePublishIntervalInput.addEventListener('input', scheduleSurfaceConfigApply);
if (surfaceColorProfileInput) {
    surfaceColorProfileInput.addEventListener('input', scheduleSurfaceConfigApply);
}

if (vocalHelp) {
  const toggleHelp = () => {
    vocalHelp.classList.toggle('is-open');
  };
  vocalHelp.addEventListener('click', (event) => {
    event.stopPropagation();
    toggleHelp();
  });
  vocalHelp.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      event.stopPropagation();
      toggleHelp();
    }
  });
  document.addEventListener('click', (event) => {
    if (!vocalHelp.contains(event.target)) {
      vocalHelp.classList.remove('is-open');
    }
  });
}

async function sendConfig(payload) {
  setStatus('Applying configuration…');
  try {
    const response = await fetch('/api/audio-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error('Backend returned an error');
    }
    const data = await response.json();
    applyConfigToInputs(data);
    setStatus('Configuration synced with backend', 'success');
  } catch (err) {
    console.error(err);
    setStatus('Preview only: backend endpoint not reachable yet.', 'warning');
  }
}

function collectFormData() {
  const getValue = (id) => document.getElementById(id).value;
  return {
    app: {
      backend: getValue('backendSelect'),
      refresh_interval_ms: Number(refreshIntervalInput.value),
    },
    stream: {
      sample_rate: Number(getValue('sampleRate')),
      channels: Number(getValue('channels')),
      sample_width: Number(getValue('sampleWidth')),
      frames_per_chunk: Number(getValue('framesPerChunk')),
      input_device: getValue('inputDevice') || null,
      preamp_gain: Number(getValue('preampGain')),
    },
    fft: {
      time_window_ms: Number(getValue('timeWindow')),
      size: Number(getValue('fftSize')),
      window_func: getValue('windowType'),
      min_frequency: Number(getValue('minFreq')),
      max_frequency: Number(getValue('maxFreq')),
      smoothing: Number(getValue('smoothing')),
    },
    visualization: {
      max_signal_points: Number(signalPointLimitInput.value),
      signal_line_width: Number(signalLineWidthInput.value),
      signal_marker_size: Number(signalMarkerSizeInput.value),
      fft_line_width: Number(fftLineWidthInput.value),
      fft_marker_size: Number(fftMarkerSizeInput.value),
      peak_plot_window_ms: Number(surfacePlotWindowInput.value),
      surface_freq_bins: Number(surfaceFreqBinsInput.value),
      surface_color_profile: (surfaceColorProfileInput && surfaceColorProfileInput.value) || DEFAULT_SURFACE_PROFILE,
      surface_publish_interval_ms: Number(surfacePublishIntervalInput.value),
      surface_time_bins: Number(surfaceTimeBinsInput.value),
    },
    visualizationChannels: Array.from(selectedChannels),
  };
}

applyBtn.addEventListener('click', () => {
  const data = collectFormData();
  sendConfig(data);
});

function drawPlaceholder(canvas, renderFn) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth * dpr;
  const height = canvas.clientHeight * dpr;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  renderFn(ctx, width, height);
}

function drawSignalPlot(ctx, width, height, signal) {
  const frames = signal && Array.isArray(signal.frames) ? signal.frames : [];
  const totalChannels = signal && signal.channels ? signal.channels : liveChannelCount;
  const indices = getSelectedChannelIndices(totalChannels);
  const usableHeight = height - PLOT_BOTTOM_PADDING;
  const maxPerChannel = new Array(totalChannels).fill(1);
  frames.forEach((frame) => {
    frame.forEach((value, idx) => {
      maxPerChannel[idx] = Math.max(maxPerChannel[idx], Math.abs(value));
    });
  });
  if (frames.length && indices.length) {
    const baseline = usableHeight / 2;
    const lineWidth = Number(signalLineWidthInput.value) || 1.5;
    const markerSize = Math.max(0, Number(signalMarkerSizeInput.value) || 0);
    indices.forEach((channelIdx, index) => {
      ctx.beginPath();
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = CHANNEL_COLORS[channelIdx % CHANNEL_COLORS.length];
      const amplitude = maxPerChannel[channelIdx] || 1;
      const offset = (index - (indices.length - 1) / 2) * 6;
      frames.forEach((frame, frameIndex) => {
        const value = frame[channelIdx] !== undefined ? frame[channelIdx] : 0;
        const x = (frameIndex / (frames.length - 1 || 1)) * width;
        const normalized = Math.max(-1, Math.min(1, value / amplitude));
        const y = baseline - normalized * (usableHeight / 2 - 5) + offset;
        if (frameIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (markerSize > 0) {
        ctx.fillStyle = CHANNEL_COLORS[channelIdx % CHANNEL_COLORS.length];
        frames.forEach((frame, frameIndex) => {
          const value = frame[channelIdx] !== undefined ? frame[channelIdx] : 0;
          const x = (frameIndex / (frames.length - 1 || 1)) * width;
          const normalized = Math.max(-1, Math.min(1, value / amplitude));
          const y = baseline - normalized * (usableHeight / 2 - 5) + offset;
          ctx.beginPath();
          ctx.arc(x, y, markerSize, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
  }
  const durationMs = Number(timeWindowInput.value) || 1000;
  drawTimeAxis(ctx, width, height, durationMs);
  document.getElementById('signalPointCount').textContent = frames.length;
}

function drawFftPlot(ctx, width, height, fft, channels) {
  const freqs = fft && Array.isArray(fft.frequency) ? fft.frequency : [];
  const magnitude = fft && Array.isArray(fft.magnitude) ? fft.magnitude : [];
  const totalChannels = channels || liveChannelCount;
  const indices = getSelectedChannelIndices(totalChannels);
  const usableHeight = height - PLOT_BOTTOM_PADDING;
  if (freqs.length && magnitude.length && indices.length) {
    const maxMagnitude = Math.max(...magnitude.flat().map((v) => Math.abs(v) || 0), 1);
    const lineWidth = Number(fftLineWidthInput.value) || 1.2;
    const markerSize = Math.max(0, Number(fftMarkerSizeInput.value) || 0);
    indices.forEach((channelIdx) => {
      ctx.beginPath();
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = CHANNEL_COLORS[channelIdx % CHANNEL_COLORS.length];
      freqs.forEach((freq, i) => {
        const magRow = magnitude[i];
        const mag = magRow && magRow[channelIdx] !== undefined ? magRow[channelIdx] : 0;
        const x = (i / (freqs.length - 1 || 1)) * width;
        const y = usableHeight - (Math.min(mag, maxMagnitude) / maxMagnitude) * (usableHeight * 0.95);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (markerSize > 0) {
        ctx.fillStyle = CHANNEL_COLORS[channelIdx % CHANNEL_COLORS.length];
        freqs.forEach((freq, i) => {
          const magRow = magnitude[i];
          const mag = magRow && magRow[channelIdx] !== undefined ? magRow[channelIdx] : 0;
          const x = (i / (freqs.length - 1 || 1)) * width;
          const y = usableHeight - (Math.min(mag, maxMagnitude) / maxMagnitude) * (usableHeight * 0.95);
          ctx.beginPath();
          ctx.arc(x, y, markerSize, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
  }
  const freqRange = freqs.length ? { min: freqs[0], max: freqs[freqs.length - 1] } : null;
  drawFrequencyAxis(ctx, width, height, freqRange);
  document.getElementById('fftPointCount').textContent = freqs.length;
  updateFftPeakInfo(freqs, magnitude, indices);
}

function drawTimeAxis(ctx, width, height, durationMs) {
  const axisSpan = durationMs || Number(timeWindowInput.value) || 1000;
  const axisY = height - PLOT_BOTTOM_PADDING + 12;
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, axisY);
  ctx.lineTo(width - 10, axisY);
  ctx.stroke();

  ctx.fillStyle = '#cbd5f5';
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px 'Segoe UI', sans-serif`;
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const x = 20 + (i / ticks) * (width - 30);
    const value = Math.round((i / ticks) * axisSpan);
    ctx.beginPath();
    ctx.moveTo(x, axisY - 4);
    ctx.lineTo(x, axisY + 4);
    ctx.stroke();
    ctx.fillText(`${value} ms`, x - 15, axisY + 18);
  }
  ctx.fillText('Time (ms)', width / 2 - 30, axisY + 34);
}

function drawFrequencyAxis(ctx, width, height, range) {
  const minFreq = (range && range.min !== undefined ? range.min : Number(minFreqInput.value)) || 0;
  const maxFreq = (range && range.max !== undefined ? range.max : Number(maxFreqInput.value)) || 20000;
  const axisY = height - PLOT_BOTTOM_PADDING + 12;
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, axisY);
  ctx.lineTo(width - 10, axisY);
  ctx.stroke();

  ctx.fillStyle = '#cbd5f5';
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px 'Segoe UI', sans-serif`;
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const x = 20 + (i / ticks) * (width - 30);
    const freq = minFreq + (i / ticks) * (maxFreq - minFreq);
    ctx.beginPath();
    ctx.moveTo(x, axisY - 4);
    ctx.lineTo(x, axisY + 4);
    ctx.stroke();
    ctx.fillText(formatFrequency(freq), x - 20, axisY + 18);
  }
  ctx.fillText('Frequency (Hz)', width / 2 - 45, axisY + 34);
}

function drawSurfacePlot(ctx, width, height, surfaceData) {
  const durationMs = Math.max(200, Number(surfacePlotWindowInput.value) || 5000);
  const latestTs = latestFrameTimestamp || Date.now();
  const windowStart = latestTs - durationMs;
  const usableHeight = height - PLOT_BOTTOM_PADDING;
  const history = normalizeSurfaceHistory(surfaceData, windowStart, latestTs);
  const freqSpan = Math.max(history.maxFreq - history.minFreq, 1);
  const axisWidth = width - 30;
  const axisTop = 12;
  const axisHeight = usableHeight - axisTop;
  const columnWidth = history.entries.length
    ? Math.max(2, axisWidth / Math.max(1, history.entries.length))
    : axisWidth;
  const palette = getSurfacePalette();

  history.entries.forEach((entry, idx) => {
    if (idx < 0 || idx >= history.entries.length) return;
    const x = 20 + (idx / Math.max(1, history.entries.length - 1)) * axisWidth - columnWidth / 2;
    const magnitudes = collapseChannelMagnitudes(entry.channels);
    if (!magnitudes.length) return;
    for (let i = 0; i < magnitudes.length; i += 1) {
      const value = magnitudes[i];
      if (!Number.isFinite(value) || value <= 0) continue;
      const freqValue = history.frequency[i] !== undefined
        ? history.frequency[i]
        : history.minFreq + (i / Math.max(1, magnitudes.length - 1)) * freqSpan;
      if (freqValue < history.minFreq || freqValue > history.maxFreq) continue;
      const nextFreq = history.frequency[i + 1] !== undefined
        ? history.frequency[i + 1]
        : freqValue + freqSpan / Math.max(1, magnitudes.length);
      const clampedNext = Math.min(history.maxFreq, Math.max(freqValue, nextFreq));
      const freqRatio = (freqValue - history.minFreq) / freqSpan;
      const nextRatio = (clampedNext - history.minFreq) / freqSpan;
      const y = axisTop + (1 - freqRatio) * axisHeight;
      const nextY = axisTop + (1 - nextRatio) * axisHeight;
      const rectHeight = Math.max(1, Math.abs(nextY - y));
      const yTop = Math.min(y, nextY);
      const normalized = Math.pow(value / history.maxMagnitude, 0.7);
      ctx.fillStyle = sampleGradientColor(palette.stops, normalized);
      ctx.fillRect(x, yTop, columnWidth, rectHeight);
    }
  });

  drawSurfaceTimeAxis(ctx, width, height, durationMs);
  drawSurfaceFrequencyAxis(ctx, width, height, history.minFreq, history.maxFreq);
  const countEl = document.getElementById('surfaceFrameCount');
  if (countEl) {
    countEl.textContent = history.entries.length;
  }
}

function collapseChannelMagnitudes(channels) {
  if (!Array.isArray(channels) || channels.length === 0) return [];
  let minLength = Infinity;
  channels.forEach((row) => {
    if (Array.isArray(row) && row.length) {
      minLength = Math.min(minLength, row.length);
    }
  });
  if (!Number.isFinite(minLength) || minLength <= 0) return [];
  const combined = new Array(minLength).fill(0);
  channels.forEach((row) => {
    if (!Array.isArray(row)) return;
    for (let i = 0; i < minLength; i += 1) {
      const value = row[i];
      if (!Number.isFinite(value)) continue;
      combined[i] = Math.max(combined[i], value);
    }
  });
  return combined;
}

function normalizeSurfaceHistory(surfaceData, windowStart, latestTs) {
  const frequency = surfaceData && Array.isArray(surfaceData.frequency) ? surfaceData.frequency : [];
  const rawHistory = surfaceData && Array.isArray(surfaceData.history) ? surfaceData.history : [];
  const minFreqFallback = Number(minFreqInput.value) || 0;
  const maxFreqFallback = Number(maxFreqInput.value) || 20000;
  const entries = [];
  let maxMagnitude = 0;
  rawHistory.forEach((entry, idx) => {
    const channels = Array.isArray(entry.channels) ? entry.channels : [];
    if (!channels.length) return;
    // preserve arrival order; assign a synthetic timestamp for compatibility
    const timestampMs = (entry.timestamp !== undefined ? entry.timestamp * 1000 : idx);
    entries.push({ timestampMs, channels });
    channels.forEach((channelRow) => {
      if (!Array.isArray(channelRow)) return;
      channelRow.forEach((value) => {
        if (Number.isFinite(value)) {
          maxMagnitude = Math.max(maxMagnitude, value);
        }
      });
    });
  });
  const minFreq = frequency.length ? frequency[0] : minFreqFallback;
  const maxFreq = frequency.length ? frequency[frequency.length - 1] : maxFreqFallback;
  if (!Number.isFinite(maxMagnitude) || maxMagnitude <= 0) {
    maxMagnitude = 1;
  }
  return {
    entries: entries.sort((a, b) => a.timestampMs - b.timestampMs),
    frequency,
    minFreq,
    maxFreq,
    maxMagnitude,
  };
}

function appendSurfaceFrame(frame) {
  const fft = frame.fft || {};
  const frequency = Array.isArray(fft.frequency) ? fft.frequency : [];
  const magnitude = Array.isArray(fft.magnitude) ? fft.magnitude : [];
  if (!frequency.length || !magnitude.length) return;
  const channelCount = (frame.signal && frame.signal.channels) || liveChannelCount;
  surfaceFrames.push({ frequency, magnitude, channelCount });
  trimSurfaceFrames();
}

function trimSurfaceFrames() {
  const timeBinsValue = Number(surfaceTimeBinsInput.value);
  const timeBins = Number.isFinite(timeBinsValue) ? Math.max(8, timeBinsValue) : 120;
  const maxFrames = Math.max(timeBins * 2, timeBins + 16);
  if (surfaceFrames.length > maxFrames) {
    surfaceFrames = surfaceFrames.slice(-maxFrames);
  }
}

function buildSurfaceGrid() {
  const windowMsValue = Number(surfacePlotWindowInput.value);
  const windowMs = Number.isFinite(windowMsValue) ? Math.max(200, windowMsValue) : 5000;
  const freqBinsValue = Number(surfaceFreqBinsInput.value);
  const freqBins = Number.isFinite(freqBinsValue) ? Math.max(8, freqBinsValue) : 256;
  const timeBinsValue = Number(surfaceTimeBinsInput.value);
  const timeBins = Number.isFinite(timeBinsValue) ? Math.max(8, timeBinsValue) : 120;
  const minFreq = Number(minFreqInput.value) || 0;
  const maxFreq = Math.max(minFreq + 1, Number(maxFreqInput.value) || minFreq + 1);
  trimSurfaceFrames();
  const sortedFrames = [...surfaceFrames];
  const targetFreqs = buildSurfaceFrequencyAxis(minFreq, maxFreq, freqBins);
  const channelCount = resolveSurfaceChannelCount(sortedFrames);
  const emptyChannels = createEmptyChannels(channelCount, targetFreqs.length);
  if (!sortedFrames.length) {
    return { history: [], frequency: targetFreqs };
  }
  const entries = [];
  const frameCount = sortedFrames.length;
  for (let i = 0; i < timeBins; i += 1) {
    const relative = timeBins <= 1 ? 0 : i / (timeBins - 1);
    const framePos = relative * (frameCount - 1);
    const frameIdx = Math.round(framePos);
    const frame = sortedFrames[Math.max(0, Math.min(frameIdx, frameCount - 1))];
    const resampled = resampleFrameChannels(frame, targetFreqs, minFreq, maxFreq, channelCount);
    const syntheticTimestamp = (relative * windowMs) / 1000;
    entries.push({ timestamp: syntheticTimestamp, channels: resampled });
  }
  return { history: entries, frequency: targetFreqs };
}

function resolveSurfaceChannelCount(frames) {
  if (frames.length) {
    const value = frames[frames.length - 1].channelCount;
    if (Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  return liveChannelCount;
}

function createEmptyChannels(count, bins) {
  return Array.from({ length: count }, () => new Array(bins).fill(0));
}

function buildSurfaceFrequencyAxis(minFreq, maxFreq, bins) {
  const safeBins = Math.max(2, bins);
  const span = Math.max(1, maxFreq - minFreq);
  return Array.from({ length: safeBins }, (_, i) => minFreq + (i / (safeBins - 1)) * span);
}

function resampleFrameChannels(frame, targetFreqs, minFreq, maxFreq, channelCount) {
  const channels = [];
  for (let c = 0; c < channelCount; c += 1) {
    channels.push(targetFreqs.map((freq) => interpolateMagnitude(frame.frequency, frame.magnitude, c, freq, minFreq, maxFreq)));
  }
  return channels;
}

function interpolateMagnitude(freqs, magnitudes, channelIdx, targetFreq, minFreq, maxFreq) {
  if (!Array.isArray(freqs) || !freqs.length) return 0;
  const clamped = Math.min(maxFreq, Math.max(minFreq, targetFreq));
  const magAt = (idx) => {
    const row = magnitudes[idx];
    if (!row || row[channelIdx] === undefined) return 0;
    const value = row[channelIdx];
    return Number.isFinite(value) ? value : 0;
  };
  if (clamped <= freqs[0]) return magAt(0);
  if (clamped >= freqs[freqs.length - 1]) return magAt(freqs.length - 1);
  let left = 0;
  let right = freqs.length - 1;
  while (right - left > 1) {
    const mid = Math.floor((left + right) / 2);
    const value = freqs[mid];
    if (value === clamped) return magAt(mid);
    if (value < clamped) left = mid;
    else right = mid;
  }
  const span = Math.max(1e-6, freqs[right] - freqs[left]);
  const weight = (clamped - freqs[left]) / span;
  return magAt(left) * (1 - weight) + magAt(right) * weight;
}

function drawSurfaceTimeAxis(ctx, width, height, durationMs) {
  const axisY = height - PLOT_BOTTOM_PADDING + 12;
  const axisWidth = width - 30;
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, axisY);
  ctx.lineTo(width - 10, axisY);
  ctx.stroke();
  ctx.fillStyle = '#cbd5f5';
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px 'Segoe UI', sans-serif`;
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const ratio = i / ticks;
    const x = 20 + ratio * axisWidth;
    const age = (1 - ratio) * durationMs;
    const label = age >= 1000 ? `-${(age / 1000).toFixed(1)} s` : `-${Math.round(age)} ms`;
    ctx.beginPath();
    ctx.moveTo(x, axisY - 4);
    ctx.lineTo(x, axisY + 4);
    ctx.stroke();
    ctx.fillText(label, x - 24, axisY + 18);
  }
  ctx.fillText('Time history (ago)', width / 2 - 65, axisY + 34);
}

function drawSurfaceFrequencyAxis(ctx, width, height, minFreq, maxFreq) {
  const topPadding = 12;
  const usableHeight = height - PLOT_BOTTOM_PADDING;
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, topPadding);
  ctx.lineTo(20, usableHeight);
  ctx.stroke();
  ctx.fillStyle = '#cbd5f5';
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px 'Segoe UI', sans-serif`;
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const ratio = i / ticks;
    const freq = minFreq + (1 - ratio) * (maxFreq - minFreq);
    const y = topPadding + ratio * (usableHeight - topPadding);
    ctx.beginPath();
    ctx.moveTo(16, y);
    ctx.lineTo(24, y);
    ctx.stroke();
    ctx.fillText(formatFrequency(freq), 28, y + 4);
  }
  ctx.save();
  ctx.translate(8, usableHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Frequency (Hz)', 0, 0);
  ctx.restore();
}

function formatFrequency(freq) {
  if (freq >= 1000) {
    return `${(freq / 1000).toFixed(1)} kHz`;
  }
  return `${Math.round(freq)} Hz`;
}

function renderSignal(signal) {
  drawPlaceholder(signalCanvas, (ctx, width, height) => drawSignalPlot(ctx, width, height, signal));
}

function renderFft(fft, channels) {
  drawPlaceholder(fftCanvas, (ctx, width, height) => drawFftPlot(ctx, width, height, fft, channels));
}

function renderSurfacePlot(surfaceData) {
  if (!surfaceCanvas) return;
  drawPlaceholder(surfaceCanvas, (ctx, width, height) => drawSurfacePlot(ctx, width, height, surfaceData));
}

function buildAndRenderSurface() {
  if (surfaceBuildInFlight) return;
  surfaceBuildInFlight = true;
  try {
    surfaceGrid = buildSurfaceGrid();
    renderSurfacePlot(surfaceGrid);
    queueRefresh();
  } finally {
    surfaceBuildInFlight = false;
  }
}

function restartSurfacePublish() {
  stopSurfacePublish();
  const interval = Math.max(30, Number(surfacePublishIntervalInput.value) || 150);
  surfacePublishTimer = setInterval(buildAndRenderSurface, interval);
  buildAndRenderSurface();
}

function stopSurfacePublish() {
  if (surfacePublishTimer) {
    clearInterval(surfacePublishTimer);
    surfacePublishTimer = null;
  }
}

function refreshPlots() {
  refreshQueued = false;
  if (latestSignal) {
    renderSignal(latestSignal);
  } else {
    renderSignal({
      frames: [],
      channels: liveChannelCount,
      sampleRate: Number(document.getElementById('sampleRate').value),
    });
  }
  if (latestFft) {
    const channelCount = latestSignal ? latestSignal.channels : liveChannelCount;
    renderFft(latestFft, channelCount);
  } else {
    renderFft({ frequency: [], magnitude: [] }, liveChannelCount);
  }
  renderSurfacePlot(surfaceGrid);
}

function getSelectedChannelIndices(total) {
  const indices = [];
  selectedChannels.forEach((label) => {
    const idx = parseInt(label.replace('Channel', '').trim(), 10) - 1;
    if (!Number.isNaN(idx) && idx >= 0 && idx < total) {
      indices.push(idx);
    }
  });
  return indices;
}

function handleFrame(frame) {
  latestSignal = frame.signal;
  latestFft = frame.fft;
  latestFrameTimestamp = frame.timestamp ? frame.timestamp * 1000 : Date.now();
  try {
    appendSurfaceFrame(frame);
    if (!surfacePublishTimer) {
      surfaceGrid = buildSurfaceGrid();
    }
  } catch (err) {
    console.error('[voice-analyzer] failed to append surface frame', err);
  }
  queueRefresh();
}

function connectWebSocket() {
  if (socket) {
    socket.close();
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const hostname = window.location.hostname;
  let safeHost = (!hostname || hostname === '0.0.0.0') ? 'localhost' : hostname;
  if (safeHost.includes(':') && !safeHost.startsWith('[')) {
    safeHost = `[${safeHost}]`;
  }
  const defaultPort = !hostname ? '8000' : '';
  const hostWithPort = window.location.port
    ? `${safeHost}:${window.location.port}`
    : (defaultPort ? `${safeHost}:${defaultPort}` : safeHost);
  const wsUrl = `${protocol}://${hostWithPort}/ws/audio`;
  console.log('[voice-analyzer] connecting websocket', wsUrl);
  socket = new WebSocket(wsUrl);
  socket.addEventListener('open', () => {
    clearTimeout(reconnectTimer);
    setStatus('Live data connected', 'success');
  });
  socket.addEventListener('message', (event) => {
    console.log('[voice-analyzer] received frame');
    const payload = JSON.parse(event.data);
    console.log('[voice-analyzer] bins:', payload.fft.frequency.length);
    buildChannelTags(payload.signal.channels);
    handleFrame(payload);
  });
  socket.addEventListener('close', () => {
    setStatus('WebSocket disconnected, retrying…', 'warning');
    reconnectTimer = setTimeout(connectWebSocket, 2000);
  });
  socket.addEventListener('error', () => {
    setStatus('WebSocket error', 'warning');
  });
}

async function hydrateConfig() {
  try {
    const response = await fetch('/api/audio-config');
    if (!response.ok) throw new Error('Failed to load config');
    const data = await response.json();
    applyConfigToInputs(data);
  } catch (err) {
    console.warn('Unable to load backend config', err);
  }
}

function applyConfigToInputs(data) {
  const appConfig = data.app || {};
  const streamConfig = data.stream || {};
  const fftConfig = data.fft || {};
  const vizConfig = data.visualization || {};
  document.getElementById('backendSelect').value = appConfig.backend ?? 'pyaudio';
  timeWindowInput.value = fftConfig.time_window_ms ?? 1000;
  updateTimeWindowResolution();
  refreshIntervalInput.value = appConfig.refresh_interval_ms ?? 50;
  signalPointLimitInput.value = vizConfig.max_signal_points ?? 2000;
  signalLineWidthInput.value = vizConfig.signal_line_width ?? 1.5;
  signalMarkerSizeInput.value = vizConfig.signal_marker_size ?? 0;
  fftLineWidthInput.value = vizConfig.fft_line_width ?? 1.2;
  fftMarkerSizeInput.value = vizConfig.fft_marker_size ?? 0;
  surfacePlotWindowInput.value = vizConfig.peak_plot_window_ms ?? 5000;
  surfaceFreqBinsInput.value = vizConfig.surface_freq_bins ?? 256;
  if (surfaceColorProfileInput) {
    const profileValue = (vizConfig.surface_color_profile || DEFAULT_SURFACE_PROFILE).toLowerCase();
    const validValues = Array.from(surfaceColorProfileInput.options || []).map((opt) => opt.value);
    surfaceColorProfileInput.value = validValues.includes(profileValue) ? profileValue : DEFAULT_SURFACE_PROFILE;
  }
  document.getElementById('sampleRate').value = streamConfig.sample_rate ?? 44100;
  document.getElementById('channels').value = streamConfig.channels ?? 1;
  document.getElementById('sampleWidth').value = streamConfig.sample_width ?? 2;
  document.getElementById('framesPerChunk').value = streamConfig.frames_per_chunk ?? 1024;
  document.getElementById('inputDevice').value =
    streamConfig.input_device !== undefined && streamConfig.input_device !== null
      ? streamConfig.input_device
      : '';
  document.getElementById('preampGain').value = streamConfig.preamp_gain ?? 1.0;
  document.getElementById('fftSize').value = fftConfig.size ?? 2048;
  const windowValue = (() => {
    const raw = fftConfig.window_func ?? fftConfig.window ?? 'rect';
    return raw === 'none' ? 'rect' : raw;
  })();
  document.getElementById('windowType').value = windowValue;
  minFreqInput.value = fftConfig.min_frequency ?? 20;
  maxFreqInput.value = fftConfig.max_frequency ?? 20000;
  document.getElementById('smoothing').value = fftConfig.smoothing ?? 0;
  surfacePublishIntervalInput.value = vizConfig.surface_publish_interval_ms ?? 150;
  surfaceTimeBinsInput.value = vizConfig.surface_time_bins ?? 120;
  const shouldAutoFill = !hasHydrated;
  buildChannelTags(streamConfig.channels, shouldAutoFill);
  surfaceFrames = [];
  surfaceGrid = { history: [], frequency: [] };
  refreshPlots();
  restartSurfacePublish();
  hasHydrated = true;
}

buildChannelTags(liveChannelCount, true);
updateTimeWindowResolution();
refreshPlots();
setStatus('Loading configuration…');
hydrateConfig().catch((err) => {
  console.warn('Unable to load backend config', err);
}).finally(() => {
  connectWebSocket();
  restartSurfacePublish();
  setStatus('Ready to configure live stream.');
});

function queueRefresh() {
  if (refreshQueued) return;
  refreshQueued = true;
  requestAnimationFrame(refreshPlots);
}

function updateFftPeakInfo(freqs, magnitude, indices) {
  const infoEl = document.getElementById('fftPeakInfo');
  if (!freqs.length || !magnitude.length) {
    infoEl.textContent = 'Peak: --';
    infoEl.style.color = '#cbd5f5';
    return;
  }
  const channelIdx = (indices && indices.length) ? indices[0] : 0;
  let peakIndex = -1;
  let peakValue = -Infinity;
  for (let i = 0; i < magnitude.length; i += 1) {
    const row = magnitude[i];
    const val = row && row[channelIdx] !== undefined ? row[channelIdx] : -Infinity;
    if (val > peakValue) {
      peakValue = val;
      peakIndex = i;
    }
  }
  if (peakIndex === -1) {
    infoEl.textContent = 'Peak: --';
    infoEl.style.color = '#cbd5f5';
    return;
  }
  const peakFreq = freqs[peakIndex];
  if (!peakFreq || peakFreq <= 0) {
    infoEl.textContent = 'Peak: --';
    infoEl.style.color = '#cbd5f5';
    return;
  }
  const noteInfo = getClosestNote(peakFreq);
  if (!noteInfo) {
    infoEl.textContent = `Peak: ${peakFreq.toFixed(1)} Hz`;
    infoEl.style.color = '#cbd5f5';
    return;
  }
  const deviation = noteInfo.deviation;
  const deviationText = `${deviation > 0 ? '+' : ''}${deviation.toFixed(2)}%`;
  infoEl.textContent = `Peak: ${peakFreq.toFixed(1)} Hz · Note ${noteInfo.note} (${noteInfo.noteFreq.toFixed(2)} Hz) · Δ ${deviationText}`;
  if (Math.abs(deviation) <= 1) {
    infoEl.style.color = '#22c55e';
  } else if (deviation < 0) {
    infoEl.style.color = '#facc15';
  } else {
    infoEl.style.color = '#f87171';
  }
}

function getClosestNote(freq) {
  if (!freq || freq <= 0) return null;
  const noteNumber = Math.round(12 * Math.log2(freq / 440) + 69);
  if (!Number.isFinite(noteNumber)) return null;
  const noteFreq = 440 * 2 ** ((noteNumber - 69) / 12);
  const noteName = NOTE_NAMES[(noteNumber + 120) % 12];
  const octave = Math.floor(noteNumber / 12) - 1;
  const deviation = ((freq - noteFreq) / noteFreq) * 100;
  return { note: `${noteName}${octave}`, noteFreq, deviation };
}
