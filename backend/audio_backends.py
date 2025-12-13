from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from .config import StreamConfig

ChunkCallback = Callable[[np.ndarray], None]


def dtype_from_sample_width(width: int) -> np.dtype:
    if width == 1:
        return np.int8
    if width == 2:
        return np.int16
    if width == 3:
        # 24-bit audio stored in 32-bit container
        return np.int32
    if width == 4:
        return np.int32
    raise ValueError(f"Unsupported sample width: {width}")


class AudioBackend(ABC):
    def __init__(self, config: StreamConfig):
        self.config = config

    @abstractmethod
    def start(self, on_chunk: ChunkCallback) -> None:
        """Start streaming and call callback for every chunk."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release resources."""


class PyAudioBackend(AudioBackend):
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self._pa = None
        self._stream = None

    def start(self, on_chunk: ChunkCallback) -> None:
        import pyaudio  # type: ignore

        self._pa = pyaudio.PyAudio()
        sample_format = self._format_from_width(pyaudio)
        dtype = dtype_from_sample_width(self.config.sample_width)

        def callback(in_data, _frame_count, _time_info, _status_flags):
            array = np.frombuffer(in_data, dtype=dtype)
            array = array.reshape(-1, self.config.channels).astype(np.float32)
            array *= self.config.preamp_gain
            on_chunk(array)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=sample_format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.frames_per_chunk,
            input_device_index=self._resolve_device_index(pyaudio),
            stream_callback=callback,
            start=True,
        )

    def _format_from_width(self, pyaudio_module):
        mapping = {
            1: pyaudio_module.paInt8,
            2: pyaudio_module.paInt16,
            3: pyaudio_module.paInt24,
            4: pyaudio_module.paInt32,
        }
        return mapping[self.config.sample_width]

    def _resolve_device_index(self, pyaudio_module) -> Optional[int]:
        if self.config.input_device is None:
            return None
        if isinstance(self.config.input_device, int):
            return self.config.input_device
        try:
            return int(self.config.input_device)
        except (ValueError, TypeError):
            pass
        count = self._pa.get_device_count()
        name = str(self.config.input_device).lower()
        for idx in range(count):
            info = self._pa.get_device_info_by_index(idx)
            if name in info.get("name", "").lower():
                return idx
        return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()
        self._stream = None
        self._pa = None


class SoundDeviceBackend(AudioBackend):
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self._stream = None

    def start(self, on_chunk: ChunkCallback) -> None:
        import sounddevice as sd  # type: ignore

        dtype = self._dtype_str()

        def callback(indata, _frames, _time_info, status):
            if status:
                print(f"sounddevice status: {status}")
            array = indata.copy().astype(np.float32)
            array *= self.config.preamp_gain
            on_chunk(array)

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=dtype,
            blocksize=self.config.frames_per_chunk,
            callback=callback,
            device=self.config.input_device,
        )
        self._stream.start()

    def _dtype_str(self) -> str:
        mapping = {
            1: "int8",
            2: "int16",
            3: "int32",
            4: "int32",
        }
        return mapping[self.config.sample_width]

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stream = None


class SyntheticBackend(AudioBackend):
    """Synthetic sine generator used when mic devices are unavailable or for testing."""

    CHORD_FREQUENCIES = np.array([440.0, 554.365, 659.255], dtype=np.float64)  # A4, C#5, E5

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self._phase = np.zeros(len(self.CHORD_FREQUENCIES), dtype=np.float64)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[ChunkCallback] = None

    def start(self, on_chunk: ChunkCallback) -> None:
        self._callback = on_chunk
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = self.config.frames_per_chunk / self.config.sample_rate
        while self._running:
            frames = self.config.frames_per_chunk
            t = (np.arange(frames) / self.config.sample_rate).reshape(-1, 1)
            phase = self._phase.reshape(1, -1)
            freqs = self.CHORD_FREQUENCIES.reshape(1, -1)
            chord = np.sin(2 * np.pi * (freqs * t + phase))
            mono = 0.2 * np.mean(chord, axis=1, keepdims=True)
            signal = np.tile(mono, (1, self.config.channels))
            phase_increment = (frames / self.config.sample_rate) * self.CHORD_FREQUENCIES
            self._phase = (self._phase + phase_increment) % 1.0
            if self._callback:
                self._callback(signal.astype(np.float32))
            time.sleep(interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        self._thread = None
        self._callback = None


BACKEND_MAP = {
    "pyaudio": PyAudioBackend,
    "sounddevice": SoundDeviceBackend,
    "synthetic": SyntheticBackend,
}


def create_backend(name: str, config: StreamConfig) -> AudioBackend:
    backend_cls = BACKEND_MAP.get(name.lower())
    if backend_cls is None:
        return SyntheticBackend(config)
    return backend_cls(config)
