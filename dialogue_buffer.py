# dialogue_buffer.py
"""Thread-safe dialogue buffer for dual-channel ASR results."""

import threading
import time
from dataclasses import dataclass


@dataclass
class Utterance:
    speaker: str        # "对方" | "我方"
    text: str
    timestamp: float    # time.time()


class DialogueBuffer:
    """Collects utterances from dual ASR channels, manages pending analysis queue and rolling summary."""

    def __init__(self):
        self._lock = threading.Lock()
        self._utterances: list[Utterance] = []
        self._pending: list[Utterance] = []      # not yet consumed by analyzer
        self._summary: str = ""
        self._summary_cursor: int = 0            # utterances[:cursor] covered by summary
        self._listeners: list = []               # callable(Utterance) callbacks

    # -- write side (called from ASR threads) --

    def add(self, speaker: str, text: str) -> Utterance:
        """Add a new utterance. Thread-safe. Notifies listeners."""
        u = Utterance(speaker=speaker, text=text, timestamp=time.time())
        with self._lock:
            self._utterances.append(u)
            self._pending.append(u)
        for fn in self._listeners:
            try:
                fn(u)
            except Exception:
                pass
        return u

    def on_utterance(self, callback):
        """Register a callback(Utterance) fired on each new utterance."""
        self._listeners.append(callback)

    # -- read side (called from analyzer / compressor) --

    def take_pending(self) -> list[Utterance]:
        """Atomically drain and return all pending utterances."""
        with self._lock:
            batch = self._pending[:]
            self._pending.clear()
            return batch

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def get_recent(self, n: int) -> list[Utterance]:
        """Return last n utterances (for display)."""
        with self._lock:
            return self._utterances[-n:] if n else []

    # -- summary management (called from compressor) --

    @property
    def summary(self) -> str:
        with self._lock:
            return self._summary

    @property
    def summary_cursor(self) -> int:
        with self._lock:
            return self._summary_cursor

    def unsummarized_utterances(self) -> list[Utterance]:
        """Return utterances not yet covered by the summary."""
        with self._lock:
            return self._utterances[self._summary_cursor:]

    def unsummarized_count(self) -> int:
        with self._lock:
            return len(self._utterances) - self._summary_cursor

    def update_summary(self, new_summary: str):
        """Replace current summary and advance cursor to cover all current utterances."""
        with self._lock:
            self._summary = new_summary
            self._summary_cursor = len(self._utterances)

    # -- control --

    def clear(self):
        """Reset all state."""
        with self._lock:
            self._utterances.clear()
            self._pending.clear()
            self._summary = ""
            self._summary_cursor = 0

    def all_utterances(self) -> list[Utterance]:
        with self._lock:
            return self._utterances[:]
