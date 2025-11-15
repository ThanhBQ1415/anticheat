import logging
import time
from typing import Optional, Tuple

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)


class VoiceDetector:
    """
    Voice detector that combines WebRTC VAD with a lightweight spectral classifier
    to better distinguish human speech from other sounds (music, background noise).
    """

    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode: 0-3, 3 is most aggressive
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

        # Lightweight logistic model parameters (hand-tuned using sample data)
        # Features: [log_energy, spectral_centroid_khz, spectral_rolloff_khz, spectral_flatness, zcr]
        self.feature_mean = np.array([2.1, 1.7, 3.2, 0.25, 0.12], dtype=np.float32)
        self.feature_std = np.array([0.9, 0.6, 0.9, 0.15, 0.08], dtype=np.float32)
        self.feature_weights = np.array([1.2, -0.8, -0.4, -1.1, -0.6], dtype=np.float32)
        self.feature_bias = 0.35
        self.human_threshold = 0.72  # probability threshold tuned for fewer false positives
        self.energy_bounds: Tuple[float, float] = (-1.5, 5.0)  # acceptable log-energy range
        self.max_spectral_flatness = 0.3
        self.centroid_range_khz: Tuple[float, float] = (0.15, 4.5)
        self.rolloff_max_khz = 4.8
        self.max_zcr = 0.22
        self.required_consecutive_frames = 5  # ~90ms at 30ms/frame

        self.speech_detected = False
        self.last_speech_time: Optional[float] = None
        self.speech_duration = 0.0
        self.last_human_probability = 0.0
        self._consecutive_human_frames = 0
        self._consecutive_non_human_frames = 0

    def release(self):
        """Release resources."""
        # Reset state
        self.speech_detected = False
        self.last_speech_time = None
        self.speech_duration = 0.0
        self._consecutive_human_frames = 0
        self._consecutive_non_human_frames = 0
        logger.info("VoiceDetector resources released")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_audio_frame(self, audio_data: bytes) -> bool:
        """
        Process audio frame from external source (e.g., from frontend).
        Returns True if human speech is detected.
        """
        try:
            if len(audio_data) < self.frame_size * 2:  # 2 bytes per sample
                # Not enough data for a full frame; rely on existing state
                return self._recent_speech_detected()

            # Process chunk frame-by-frame (~30ms per frame)
            frame_bytes = self.frame_size * 2
            speech_confirmed = False
            frames_processed = 0

            for offset in range(0, len(audio_data) - frame_bytes + 1, frame_bytes):
                chunk = audio_data[offset : offset + frame_bytes]
                frames_processed += 1

                if self._evaluate_frame(chunk):
                    if self._consecutive_human_frames >= self.required_consecutive_frames:
                        speech_confirmed = True
                        break
                elif self._consecutive_non_human_frames >= self.required_consecutive_frames:
                    # Reset speech state when enough non-human frames observed
                    self.speech_detected = False
                    self.speech_duration = 0.0

            if speech_confirmed:
                now = time.time()
                self.last_speech_time = now
                self.speech_detected = True
                self.speech_duration += frames_processed * (self.frame_duration_ms / 1000.0)
                return True

            return self._recent_speech_detected()
        except Exception as e:
            logger.exception("VoiceDetector: error processing audio frame: %s", e)
            return False

    def _recent_speech_detected(self) -> bool:
        """Return True if speech was detected recently to avoid flicker."""
        if not self.speech_detected or self.last_speech_time is None:
            return False

        if time.time() - self.last_speech_time <= 0.3:
            return True

        # Cooldown elapsed; reset state
        self.speech_detected = False
        self.speech_duration = 0.0
        self.last_speech_time = None
        return False

    # ------------------------------------------------------------------
    # Core detection helpers
    # ------------------------------------------------------------------
    def _evaluate_frame(self, audio_frame: bytes) -> bool:
        """Evaluate a single frame and update running counters."""
        is_human = self._is_human_frame(audio_frame)
        logger.info("VoiceDetector: is_human=%s", is_human)
        if is_human:
            self._consecutive_human_frames += 1
            self._consecutive_non_human_frames = 0
        else:
            self._consecutive_non_human_frames += 1
            self._consecutive_human_frames = 0
        return is_human

    def _is_human_frame(self, audio_frame: bytes) -> bool:
        """Combine WebRTC VAD and spectral classifier for one frame."""
        vad_result = self.vad.is_speech(audio_frame[: self.frame_size * 2], self.sample_rate)
        if not vad_result:
            self.last_human_probability = 0.0
            return False

        samples = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            self.last_human_probability = 0.0
            return False

        features = self._extract_features(samples)
        probability = self._predict_probability(features)

        log_energy = features[0]
        spectral_flatness = features[3]
        centroid = features[1]
        rolloff = features[2]
        zcr = features[4]

        conditions = [
            probability >= self.human_threshold,
            self.energy_bounds[0] <= log_energy <= self.energy_bounds[1],
            spectral_flatness <= self.max_spectral_flatness,
            self.centroid_range_khz[0] <= centroid <= self.centroid_range_khz[1],
            rolloff <= self.rolloff_max_khz,
            zcr <= self.max_zcr,
        ]

        self.last_human_probability = probability if all(conditions) else 0.0
        logger.debug(
            "VoiceDetector: features=%s prob=%.3f vad=%s conditions=%s",
            np.round(features, 3),
            probability,
            vad_result,
            conditions,
        )
        return all(conditions)

    def _extract_features(self, samples: np.ndarray) -> np.ndarray:
        """Extract lightweight spectral features for classification."""
        # Normalize to [-1, 1]
        samples = samples / 32768.0
        samples = np.clip(samples, -1.0, 1.0)

        window = np.hanning(samples.size)
        windowed = samples * window

        n_fft = int(2 ** np.ceil(np.log2(max(256, windowed.size))))
        spectrum = np.fft.rfft(windowed, n=n_fft)
        magnitude = np.abs(spectrum) + 1e-8
        energy = np.sum(magnitude ** 2)

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        weighted_sum = np.sum(freqs * magnitude)
        centroid = (weighted_sum / (np.sum(magnitude) + 1e-8)) / 1000.0  # kHz

        cumulative = np.cumsum(magnitude)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1]) if cumulative[-1] > 0 else 0
        rolloff = (freqs[min(rolloff_idx, freqs.size - 1)] if freqs.size else 0.0) / 1000.0  # kHz

        spectral_flatness = np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude) + 1e-8)
        zcr = np.mean(np.abs(np.diff(np.sign(samples)))) * 0.5

        log_energy = np.log10(energy + 1e-8)
        return np.array([log_energy, centroid, rolloff, spectral_flatness, zcr], dtype=np.float32)

    def _predict_probability(self, features: np.ndarray) -> float:
        """Apply small logistic model."""
        normalized = (features - self.feature_mean) / (self.feature_std + 1e-6)
        score = float(np.dot(self.feature_weights, normalized) + self.feature_bias)
        probability = 1.0 / (1.0 + np.exp(-score))
        return probability

    # ------------------------------------------------------------------

