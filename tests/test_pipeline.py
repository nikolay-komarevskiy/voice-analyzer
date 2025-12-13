import unittest

import numpy as np

from backend.config import Settings
from backend.pipeline import FftProcessor, PeakTracker


class FftProcessorTests(unittest.TestCase):
    def test_process_matches_expected_peak_frequency(self):
        sample_rate = 16384
        fft_size = 2048
        target_freq = 440.0  # Aligns exactly with a discrete FFT bin for this SR/size.
        samples = _make_tone(sample_rate, fft_size, target_freq)

        settings = _build_settings(sample_rate, fft_size)
        processor = FftProcessor(settings)
        freqs, spectrum, _, _ = processor.process(samples)
        self.assertGreater(freqs.size, 0)
        self.assertEqual(spectrum.shape[1], 1)

        peak_index = int(np.argmax(spectrum[:, 0]))
        self.assertAlmostEqual(freqs[peak_index], target_freq, places=6)

    def test_peak_tracker_interpolates_between_bins(self):
        sample_rate = 44100
        fft_size = 8192
        target_freq = 554.365
        samples = _make_tone(sample_rate, fft_size, target_freq)

        settings = _build_settings(sample_rate, fft_size)
        processor = FftProcessor(settings)
        tracker = PeakTracker(
            settings.peak_plot_window_seconds,
            settings.peak_buffer_size,
            settings.stream.channels,
        )

        _, _, detect_freq, detect_spectrum = processor.process(samples)
        peak_payload = tracker.update(detect_freq, detect_spectrum, timestamp=0.0)
        series = peak_payload["series"][0]
        measured_freq = series["frequencies"][-1]
        self.assertAlmostEqual(measured_freq, target_freq, places=2)


def _build_settings(sample_rate: int, fft_size: int) -> Settings:
    return Settings.from_dict(
        {
            "app": {"backend": "synthetic", "refresh_interval_ms": 50},
            "stream": {
                "sample_rate": sample_rate,
                "channels": 1,
                "sample_width": 2,
                "frames_per_chunk": fft_size,
                "input_device": None,
                "preamp_gain": 1.0,
            },
            "fft": {
                "time_window_ms": 200,
                "size": fft_size,
                "window_func": "rect",
                "min_frequency": 0.0,
                "max_frequency": sample_rate / 2,
                "smoothing": 0.0,
            },
            "visualization": {
                "max_signal_points": 1000,
                "signal_line_width": 1.5,
                "signal_marker_size": 0.0,
                "fft_line_width": 1.2,
                "fft_marker_size": 0.0,
                "peak_plot_window_ms": 5000,
                "peak_buffer_size": 5,
            },
        }
    )


def _make_tone(sample_rate: int, fft_size: int, frequency: float) -> np.ndarray:
    t = np.arange(fft_size) / sample_rate
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)[:, np.newaxis]


if __name__ == "__main__":
    unittest.main()
