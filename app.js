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

let selectedChannels = new Set(['Channel 1']);
let liveChannelCount = Number(channelsInput.value) || 1;
let socket;
let reconnectTimer;
let latestSignal = null;
let latestFft = null;
let hasHydrated = false;
const PLOT_BOTTOM_PADDING = 50;
const CHANNEL_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f97316', '#f472b6', '#facc15', '#f87171', '#0ea5e9'];
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const vocalHelp = document.getElementById('vocalHelp');

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

function refreshPlots() {
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
  refreshPlots();
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
  const hostWithPort = window.location.port ? `${safeHost}:${window.location.port}` : safeHost;
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
  const shouldAutoFill = !hasHydrated;
  buildChannelTags(streamConfig.channels, shouldAutoFill);
  refreshPlots();
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
  setStatus('Ready to configure live stream.');
});

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
