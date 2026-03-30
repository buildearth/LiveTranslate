# summary_compressor.py
"""Async rolling summary compressor for dialogue context management."""

import logging
import threading

from openai import OpenAI

from analysis_presets import SUMMARY_COMPRESS_PROMPT
from dialogue_buffer import DialogueBuffer, Utterance

log = logging.getLogger(__name__)


def format_utterances(utterances: list[Utterance]) -> str:
    """Format utterances into timestamped text block."""
    from datetime import datetime
    lines = []
    for u in utterances:
        ts = datetime.fromtimestamp(u.timestamp).strftime("%H:%M:%S")
        lines.append(f"[{u.speaker} {ts}] {u.text}")
    return "\n".join(lines)


class SummaryCompressor:
    """Monitors DialogueBuffer and compresses history into rolling summaries.

    Runs compression in a background thread. Does not block the analyzer.
    """

    def __init__(self, buffer: DialogueBuffer, threshold: int = 15):
        self._buffer = buffer
        self._threshold = threshold
        self._client: OpenAI | None = None
        self._model: str = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._event = threading.Event()  # signaled when new utterances arrive
        self._lock = threading.Lock()

    def set_client(self, client: OpenAI, model: str):
        with self._lock:
            self._client = client
            self._model = model

    def start(self):
        if self._running:
            return
        self._running = True
        self._buffer.on_utterance(lambda _u: self._event.set())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _loop(self):
        while self._running:
            self._event.wait(timeout=5.0)
            self._event.clear()
            if not self._running:
                break
            if self._buffer.unsummarized_count() >= self._threshold:
                self._compress()

    def _compress(self):
        with self._lock:
            client = self._client
            model = self._model
        if not client or not model:
            return

        old_summary = self._buffer.summary
        new_utterances = self._buffer.unsummarized_utterances()
        if not new_utterances:
            return

        user_content = ""
        if old_summary:
            user_content += f"## 已有摘要\n{old_summary}\n\n"
        user_content += f"## 新增对话\n{format_utterances(new_utterances)}"

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_COMPRESS_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            new_summary = resp.choices[0].message.content.strip()
            self._buffer.update_summary(new_summary)
            log.info("Summary compressed: %d utterances → %d chars",
                     len(new_utterances), len(new_summary))
        except Exception as e:
            log.error("Summary compression failed: %s", e)
