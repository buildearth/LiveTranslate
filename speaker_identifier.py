# speaker_identifier.py
"""Real-time speaker identification via embedding clustering."""

import logging
import threading

import numpy as np

log = logging.getLogger("LiveTranslate.Speaker")


class SpeakerIdentifier:
    """Identifies speakers by comparing audio segment embeddings.

    Uses resemblyzer (GE2E model) to extract d-vector embeddings,
    then clusters them online using cosine similarity.
    """

    def __init__(self, similarity_threshold: float = 0.75, device: str = "cpu"):
        self._threshold = similarity_threshold
        self._device = device
        self._lock = threading.Lock()

        # Speaker centroids: list of (label, embedding_sum, count)
        self._speakers: list[tuple[str, np.ndarray, int]] = []
        self._encoder = None
        self._ready = False

    def load(self):
        """Load the speaker embedding model. Call once at startup."""
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder(device=self._device)
            self._ready = True
            log.info("Speaker encoder loaded on %s", self._device)
        except ImportError:
            log.warning("resemblyzer not installed, speaker identification disabled")
        except Exception as e:
            log.error("Failed to load speaker encoder: %s", e)

    @property
    def ready(self) -> bool:
        return self._ready

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Identify speaker from audio segment. Returns label like '说话人1'.

        Thread-safe. Returns '未知' if model not loaded.
        """
        if not self._ready or self._encoder is None:
            return "未知"

        try:
            # resemblyzer expects float32 audio at 16kHz
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Extract embedding
            embed = self._encoder.embed_utterance(audio)

            with self._lock:
                return self._match_or_create(embed)
        except Exception as e:
            log.error("Speaker identification error: %s", e)
            return "未知"

    def _match_or_create(self, embed: np.ndarray) -> str:
        """Match embedding to known speaker or create new one."""
        best_sim = -1.0
        best_idx = -1

        for i, (label, centroid_sum, count) in enumerate(self._speakers):
            centroid = centroid_sum / count
            sim = self._cosine_similarity(embed, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self._threshold and best_idx >= 0:
            # Update centroid with running average
            label, centroid_sum, count = self._speakers[best_idx]
            self._speakers[best_idx] = (label, centroid_sum + embed, count + 1)
            return label
        else:
            # New speaker
            idx = len(self._speakers) + 1
            label = f"说话人{idx}"
            self._speakers.append((label, embed.copy(), 1))
            log.info("New speaker detected: %s (total: %d)", label, idx)
            return label

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def reset(self):
        """Clear all known speakers."""
        with self._lock:
            self._speakers.clear()
        log.info("Speaker identities reset")

    def speaker_count(self) -> int:
        with self._lock:
            return len(self._speakers)
