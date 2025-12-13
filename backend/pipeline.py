from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import List, Optional

import numpy as np

from .audio_backends import AudioBackend, create_backend, SyntheticBackend
from .config import Settings

logger = logging.getLogger(__name__)


class AudioStreamManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._backend: Optional[AudioBackend] = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=8)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        if self._backend:
            return
        self._loop = asyncio.get_running_loop()
        backend = create_backend(self.settings.app.backend, self.settings.stream)
        try:
            backend.start(self._handle_chunk)
        except ModuleNotFoundError:
            logger.warning(
                "Module for backend %s missing, using synthetic generator",
                self.settings.app.backend,
            )
            backend = SyntheticBackend(self.settings.stream)
            backend.start(self._handle_chunk)
        except Exception as exc:
            logger.exception(
                "Failed to start backend %s, falling back to synthetic: %s",
                self.settings.app.backend,
                exc,
            )
            backend = SyntheticBackend(self.settings.stream)
            backend.start(self._handle_chunk)
        self._backend = backend
        logger.info("Audio stream started with backend %s", type(self._backend).__name__)

    async def stop(self) -> None:
        if self._backend:
            self._backend.stop()
            self._backend = None
        while not self._queue.empty():
            self._queue.get_nowait()
        self._loop = None

    async def restart(self, settings: Settings) -> None:
        await self.stop()
        self.settings = settings
        await self.start()

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        if self._loop is None:
            return

        def enqueue() -> None:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(chunk)

        try:
            self._loop.call_soon_threadsafe(enqueue)
        except RuntimeError:
            pass

    async def next_chunk(self) -> np.ndarray:
        return await self._queue.get()


class RollingBuffer:
    def __init__(self, max_samples: int, channels: int):
        self.max_samples = max_samples
        self.channels = channels
        self._buffer = np.zeros((0, channels), dtype=np.float32)

    def resize(self, max_samples: int, channels: int) -> None:
        self.max_samples = max_samples
        if channels != self.channels:
            self.channels = channels
            self._buffer = np.zeros((0, channels), dtype=np.float32)
        else:
            self._buffer = self._buffer[-self.max_samples :]

    def append(self, chunk: np.ndarray) -> None:
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        if chunk.shape[1] != self.channels:
            self.channels = chunk.shape[1]
            self._buffer = np.zeros((0, self.channels), dtype=np.float32)
        self._buffer = np.vstack((self._buffer, chunk))
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples :]

    def snapshot(self) -> np.ndarray:
        return self._buffer.copy()


class PeakTracker:
    def __init__(self, window_seconds: float, buffer_size: int, channels: int):
        self.window_seconds = max(0.1, window_seconds)
        self.buffer_size = max(1, buffer_size)
        self.channels = max(1, channels)
        self._history: list[deque[tuple[float, float]]] = [
            deque() for _ in range(self.channels)
        ]
        self._buffers: list[deque[float]] = [
            deque(maxlen=self.buffer_size) for _ in range(self.channels)
        ]

    def resize(
        self,
        channels: int,
        window_seconds: Optional[float] = None,
        buffer_size: Optional[int] = None,
    ) -> None:
        if window_seconds is not None:
            self.window_seconds = max(0.1, window_seconds)
        if buffer_size is not None and int(buffer_size) != self.buffer_size:
            self.buffer_size = max(1, int(buffer_size))
            self._buffers = [
                deque(list(buf)[-self.buffer_size :], maxlen=self.buffer_size)
                for buf in self._buffers
            ]
        if channels != self.channels:
            new_history: list[deque[tuple[float, float]]] = []
            new_buffers: list[deque[float]] = []
            for idx in range(channels):
                if idx < len(self._history):
                    new_history.append(self._history[idx])
                else:
                    new_history.append(deque())
                if idx < len(self._buffers):
                    new_buffers.append(
                        deque(list(self._buffers[idx])[-self.buffer_size :], maxlen=self.buffer_size)
                    )
                else:
                    new_buffers.append(deque(maxlen=self.buffer_size))
            self._history = new_history
            self._buffers = new_buffers
            self.channels = channels
        self._trim_history(time.time())

    def update(self, freq: np.ndarray, spectrum: np.ndarray, timestamp: float) -> dict:
        has_spectrum = freq.size > 0 and spectrum.size > 0
        if has_spectrum:
            total_channels = min(self.channels, spectrum.shape[1])
            for channel in range(total_channels):
                channel_spectrum = spectrum[:, channel]
                magnitudes = np.abs(channel_spectrum)
                if magnitudes.size == 0:
                    continue
                peak_index = int(np.argmax(magnitudes))
                if peak_index < 0 or peak_index >= len(freq):
                    continue
                peak_freq = float(freq[peak_index])
                refined_freq = _interpolate_peak_frequency(
                    freq, channel_spectrum, peak_index
                )
                if refined_freq is not None:
                    peak_freq = refined_freq
                if not np.isfinite(peak_freq) or peak_freq <= 0:
                    continue
                self._buffers[channel].append(peak_freq)
                median_freq = float(np.median(self._buffers[channel]))
                self._history[channel].append((timestamp, median_freq))
        self._trim_history(timestamp)
        return self.serialize()

    def serialize(self) -> dict:
        series = []
        for idx in range(self.channels):
            history = self._history[idx] if idx < len(self._history) else deque()
            timestamps = [point[0] for point in history]
            freqs = [point[1] for point in history]
            series.append(
                {
                    "channel": idx,
                    "timestamps": timestamps,
                    "frequencies": freqs,
                }
            )
        return {"channels": self.channels, "series": series}

    def _trim_history(self, current_time: float) -> None:
        cutoff = current_time - self.window_seconds
        for history in self._history:
            while history and history[0][0] < cutoff:
                history.popleft()


WINDOW_MAP = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect": lambda n: np.ones(n),
    "none": lambda n: np.ones(n),
}


def _interpolate_peak_frequency(
    freq: np.ndarray, spectrum: np.ndarray, peak_index: int
) -> Optional[float]:
    if (
        peak_index <= 0
        or peak_index >= len(spectrum) - 1
        or len(freq) < 2
    ):
        return None
    left = np.abs(spectrum[peak_index - 1]) ** 2
    center = np.abs(spectrum[peak_index]) ** 2
    right = np.abs(spectrum[peak_index + 1]) ** 2
    denominator = (left - 2 * center + right)
    if denominator == 0:
        return None
    offset = 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    left_spacing = freq[peak_index] - freq[peak_index - 1]
    right_spacing = freq[peak_index + 1] - freq[peak_index]
    bin_width = 0.5 * (left_spacing + right_spacing)
    refined = freq[peak_index] + offset * bin_width
    if not np.isfinite(refined):
        return None
    return float(refined)


class FftProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._previous: Optional[np.ndarray] = None  # stores the last spectrum for EMA smoothing
        self._detection_factor = 4  # zero-padding factor for more precise peak estimation

    def update(self, settings: Settings) -> None:
        self.settings = settings
        self._previous = None

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if samples.size == 0:
            empty = np.array([])
            return empty, np.array([[]]), empty, np.array([[]])
        fft_conf = self.settings.fft
        n = min(len(samples), fft_conf.size)
        if n <= 1:
            empty = np.array([])
            return empty, np.array([[]]), empty, np.array([[]])
        # Select window shape and convert newest samples into a windowed slice.
        window_key = (fft_conf.window_func or "rect").lower()
        window_fn = WINDOW_MAP.get(window_key, np.ones)
        window = window_fn(n)
        sliced = samples[-n:, :]
        windowed = sliced * window[:, None]
        freq, spectrum = self._compute_fft(windowed, n)
        detect_freq, detect_spectrum = self._compute_fft(
            windowed, self._resolve_detection_size(n), as_magnitude=False
        )
        smoothing = np.clip(fft_conf.smoothing, 0.0, 0.95)
        if (
            self._previous is not None
            and smoothing > 0
            and self._previous.shape == spectrum.shape
        ):
            # Exponential moving average over spectra: only the last frame is retained.
            spectrum = smoothing * self._previous + (1 - smoothing) * spectrum
        self._previous = spectrum
        return freq, spectrum, detect_freq, detect_spectrum

    def _resolve_detection_size(self, base_size: int) -> int:
        factor = max(1, int(self._detection_factor))
        padded = base_size * factor
        return min(max(base_size, padded), 1 << 17)  # cap at 131072 pts for safety

    def _compute_fft(
        self, windowed: np.ndarray, n_fft: int, as_magnitude: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        spectrum = np.fft.rfft(windowed, n=n_fft, axis=0)
        freq = np.fft.rfftfreq(n_fft, d=1.0 / self.settings.stream.sample_rate)
        mask = (freq >= self.settings.fft.min_frequency) & (freq <= self.settings.fft.max_frequency)
        spectrum = spectrum[mask, :]
        if as_magnitude:
            spectrum = np.abs(spectrum)
        return freq[mask], spectrum


def downsample(data: np.ndarray, max_points: int) -> np.ndarray:
    if len(data) <= max_points:
        return data
    step = int(np.ceil(len(data) / max_points))
    return data[::step]


def downsample_fft(freq: np.ndarray, spectrum: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(freq) <= max_points:
        return freq, spectrum
    step = int(np.ceil(len(freq) / max_points))
    return freq[::step], spectrum[::step, :]


class BroadcastHub:
    def __init__(self):
        self._connections: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._connections.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._connections.discard(queue)

    async def publish(self, message: dict) -> None:
        stale: List[asyncio.Queue] = []
        for queue in self._connections:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                stale.append(queue)
        for q in stale:
            self._connections.discard(q)


class AudioBroadcaster:
    def __init__(self, manager: AudioStreamManager, settings: Settings):
        self.manager = manager
        self.settings = settings
        self.buffer = RollingBuffer(settings.time_window_samples, settings.stream.channels)
        self.fft_processor = FftProcessor(settings)
        self.peak_tracker = PeakTracker(
            settings.peak_plot_window_seconds,
            settings.peak_buffer_size,
            settings.stream.channels,
        )
        self.hub = BroadcastHub()
        self.refresh_interval = settings.refresh_interval_seconds
        self._ingest_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ingest_task = asyncio.create_task(self._ingest_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        self._running = False
        for task in (self._ingest_task, self._broadcast_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._ingest_task = None
        self._broadcast_task = None

    async def restart(self, settings: Settings) -> None:
        self.settings = settings
        self.buffer.resize(settings.time_window_samples, settings.stream.channels)
        self.fft_processor.update(settings)
        self.peak_tracker.resize(
            settings.stream.channels,
            window_seconds=settings.peak_plot_window_seconds,
            buffer_size=settings.peak_buffer_size,
        )
        self.refresh_interval = settings.refresh_interval_seconds

    async def stream(self):
        queue = self.hub.subscribe()
        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            self.hub.unsubscribe(queue)

    async def _ingest_loop(self) -> None:
        try:
            while self._running:
                chunk = await self.manager.next_chunk()
                if chunk is not None:
                    self.buffer.append(chunk)
        except asyncio.CancelledError:
            pass

    async def _broadcast_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.refresh_interval)
                await self._publish_snapshot()
        except asyncio.CancelledError:
            pass

    async def _publish_snapshot(self) -> None:
        samples = self.buffer.snapshot()
        timestamp = time.time()
        freq, spectrum, detect_freq, detect_spectrum = self.fft_processor.process(samples)
        latest = downsample(samples, self.settings.signal_point_limit)
        peak_plot = self.peak_tracker.update(detect_freq, detect_spectrum, timestamp)
        message = {
            "timestamp": timestamp,
            "signal": {
                "channels": samples.shape[1] if samples.size else self.settings.stream.channels,
                "sampleRate": self.settings.stream.sample_rate,
                "frames": latest.tolist(),
            },
            "fft": {
                "frequency": freq.tolist(),
                "magnitude": spectrum.tolist(),
            },
            "peak_plot": peak_plot,
        }
        await self.hub.publish(message)
