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


class SpectrumSurface:
    def __init__(
        self,
        window_seconds: float,
        channels: int,
        freq_point_limit: int,
    ):
        self.window_seconds = max(0.1, window_seconds)
        self.channels = max(1, channels)
        self.freq_point_limit = self._sanitize_limit(freq_point_limit)
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._frequency = np.array([])
        self._channel_count = self.channels
        self._max_entries = self._estimate_max_entries()

    def resize(
        self,
        channels: int,
        window_seconds: Optional[float] = None,
        freq_point_limit: Optional[int] = None,
    ) -> None:
        if window_seconds is not None:
            self.window_seconds = max(0.1, window_seconds)
            self._max_entries = self._estimate_max_entries()
        if freq_point_limit is not None:
            self.freq_point_limit = self._sanitize_limit(freq_point_limit)
        if channels != self.channels:
            self.channels = max(1, channels)
            self._channel_count = min(self._channel_count, self.channels)
            self._history.clear()
        self._trim_history(time.time())

    def update(self, freq: np.ndarray, spectrum: np.ndarray, timestamp: float) -> dict:
        if freq.size > 0 and spectrum.size > 0:
            freq_axis, trimmed_spectrum = downsample_fft(
                freq, spectrum, self.freq_point_limit
            )
            if freq_axis.size and trimmed_spectrum.size:
                channel_count = min(self.channels, trimmed_spectrum.shape[1])
                if channel_count > 0:
                    frame = trimmed_spectrum[:, :channel_count].T  # (channel, freq)
                    self._history.append((timestamp, frame))
                    self._frequency = freq_axis
                    self._channel_count = channel_count
        self._trim_history(timestamp)
        return self.serialize()

    def serialize(self) -> dict:
        history_payload = [
            {"timestamp": ts, "channels": frame.tolist()} for ts, frame in self._history
        ]
        return {
            "channels": self._channel_count,
            "frequency": self._frequency.tolist(),
            "history": history_payload,
        }

    def _trim_history(self, current_time: float) -> None:
        cutoff = current_time - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        while len(self._history) > self._max_entries:
            self._history.popleft()

    def _estimate_max_entries(self) -> int:
        # assume up to 60 refreshes per second; keep small buffer extra
        return max(2, int(self.window_seconds * 60) + 2)

    @staticmethod
    def _sanitize_limit(limit: int) -> int:
        return max(8, int(limit))


WINDOW_MAP = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect": lambda n: np.ones(n),
    "none": lambda n: np.ones(n),
}


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
        self.surface_tracker = SpectrumSurface(
            settings.peak_plot_window_seconds,
            settings.stream.channels,
            settings.surface_freq_bins,
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
        self.surface_tracker.resize(
            settings.stream.channels,
            window_seconds=settings.peak_plot_window_seconds,
            freq_point_limit=settings.surface_freq_bins,
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
        freq, spectrum, _, _ = self.fft_processor.process(samples)
        latest = downsample(samples, self.settings.signal_point_limit)
        surface_plot = self.surface_tracker.update(freq, spectrum, timestamp)
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
            "surface_plot": surface_plot,
        }
        await self.hub.publish(message)
