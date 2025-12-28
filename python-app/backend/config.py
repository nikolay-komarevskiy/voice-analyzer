from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class StreamConfig:
    sample_rate: int
    channels: int
    sample_width: int
    frames_per_chunk: int
    input_device: str | None
    preamp_gain: float
    queue_maxsize: int


@dataclass
class FftConfig:
    time_window_ms: int
    size: int
    window_func: str
    min_frequency: float
    max_frequency: float
    smoothing: float


@dataclass
class AppConfig:
    backend: str
    refresh_interval_ms: int


@dataclass
class VisualizationConfig:
    max_signal_points: int
    signal_line_width: float
    signal_marker_size: float
    fft_line_width: float
    fft_marker_size: float
    peak_plot_window_ms: int
    surface_freq_bins: int
    surface_color_profile: str
    surface_publish_interval_ms: int
    surface_time_bins: int


@dataclass
class Settings:
    app: AppConfig
    stream: StreamConfig
    fft: FftConfig
    visualization: VisualizationConfig

    @property
    def time_window_samples(self) -> int:
        return max(1, int((self.fft.time_window_ms / 1000.0) * self.stream.sample_rate))

    @property
    def refresh_interval_seconds(self) -> float:
        return max(0.01, self.app.refresh_interval_ms / 1000.0)

    @property
    def signal_point_limit(self) -> int:
        return max(100, self.visualization.max_signal_points)

    @property
    def peak_plot_window_seconds(self) -> float:
        return max(0.1, self.visualization.peak_plot_window_ms / 1000.0)

    @property
    def surface_freq_bins(self) -> int:
        return max(8, self.visualization.surface_freq_bins)

    @property
    def surface_color_profile(self) -> str:
        value = (self.visualization.surface_color_profile or "aurora").strip().lower()
        return value or "aurora"

    def merge(self, payload: Dict[str, Any]) -> "Settings":
        data = self.to_dict()
        for key, value in payload.items():
            if key in data:
                if isinstance(value, dict) and isinstance(data[key], dict):
                    data[key].update(value)
                else:
                    data[key] = value
        return Settings.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app": {
                "backend": self.app.backend,
                "refresh_interval_ms": self.app.refresh_interval_ms,
            },
            "stream": {
                "sample_rate": self.stream.sample_rate,
                "channels": self.stream.channels,
                "sample_width": self.stream.sample_width,
            "frames_per_chunk": self.stream.frames_per_chunk,
            "input_device": self.stream.input_device,
            "preamp_gain": self.stream.preamp_gain,
            "queue_maxsize": self.stream.queue_maxsize,
        },
        "fft": {
            "time_window_ms": self.fft.time_window_ms,
            "size": self.fft.size,
            "window_func": self.fft.window_func,
                "min_frequency": self.fft.min_frequency,
                "max_frequency": self.fft.max_frequency,
                "smoothing": self.fft.smoothing,
            },
            "visualization": {
                "max_signal_points": self.visualization.max_signal_points,
                "signal_line_width": self.visualization.signal_line_width,
                "signal_marker_size": self.visualization.signal_marker_size,
            "fft_line_width": self.visualization.fft_line_width,
            "fft_marker_size": self.visualization.fft_marker_size,
            "peak_plot_window_ms": self.visualization.peak_plot_window_ms,
            "surface_freq_bins": self.visualization.surface_freq_bins,
            "surface_color_profile": self.visualization.surface_color_profile,
            "surface_publish_interval_ms": self.visualization.surface_publish_interval_ms,
            "surface_time_bins": self.visualization.surface_time_bins,
        },
    }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        app = data.get("app", {})
        stream = data.get("stream", {})
        fft = data.get("fft", {})
        visualization = data.get("visualization", {})
        return cls(
            app=AppConfig(
                backend=app.get("backend", "pyaudio"),
                refresh_interval_ms=int(app.get("refresh_interval_ms", 50)),
            ),
            stream=StreamConfig(
                sample_rate=int(stream.get("sample_rate", 44100)),
                channels=int(stream.get("channels", 1)),
                sample_width=int(stream.get("sample_width", 2)),
                frames_per_chunk=int(stream.get("frames_per_chunk", 1024)),
                input_device=stream.get("input_device"),
                preamp_gain=float(stream.get("preamp_gain", 1.0)),
                queue_maxsize=max(1, int(stream.get("queue_maxsize", 8))),
            ),
            fft=FftConfig(
                time_window_ms=int(fft.get("time_window_ms", 1000)),
                size=int(fft.get("size", 2048)),
                window_func=_resolve_window_func(fft),
                min_frequency=float(fft.get("min_frequency", 20.0)),
                max_frequency=float(fft.get("max_frequency", 20000.0)),
                smoothing=float(fft.get("smoothing", 0.0)),
            ),
            visualization=VisualizationConfig(
                max_signal_points=int(visualization.get("max_signal_points", 2000)),
                signal_line_width=float(visualization.get("signal_line_width", 1.5)),
                signal_marker_size=float(visualization.get("signal_marker_size", 0.0)),
                fft_line_width=float(visualization.get("fft_line_width", 1.2)),
                fft_marker_size=float(visualization.get("fft_marker_size", 0.0)),
                peak_plot_window_ms=int(visualization.get("peak_plot_window_ms", 5000)),
                surface_freq_bins=int(visualization.get("surface_freq_bins", 256)),
                surface_color_profile=str(visualization.get("surface_color_profile", "aurora")),
                surface_publish_interval_ms=max(10, int(visualization.get("surface_publish_interval_ms", 120))),
                surface_time_bins=max(8, int(visualization.get("surface_time_bins", 120))),
            ),
        )


def _resolve_window_func(fft: Dict[str, Any]) -> str:
    raw_value = fft.get("window_func", fft.get("window", "rect"))
    if not isinstance(raw_value, str):
        return "rect"
    if raw_value.lower() == "none":
        return "rect"
    return raw_value


def load_settings(path: Path) -> Settings:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return Settings.from_dict(data)
