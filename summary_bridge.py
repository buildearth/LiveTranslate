# summary_bridge.py
"""Background bridge between DialogueBuffer and LiveSummarizer."""

import logging
import threading
import time

from dialogue_buffer import DialogueBuffer, Utterance

log = logging.getLogger(__name__)


class SummaryBridge:
    """Wraps LiveSummarizer in a background thread.

    Consumes utterances from DialogueBuffer, batches them, and calls
    LiveSummarizer.summarize() periodically. Stores latest SummaryOutput
    for AnalysisScheduler and UI to read.
    """

    BATCH_THRESHOLD = 5
    TIME_INTERVAL = 30.0

    def __init__(self, buffer: DialogueBuffer):
        self._buffer = buffer
        self._summarizer = None
        self._session_id = ""
        self._latest_output = None
        self._lock = threading.Lock()

        self._running = False
        self._thread = None
        self._event = threading.Event()
        self._pending: list[Utterance] = []
        self._pending_lock = threading.Lock()
        self._last_call_time = 0.0

        # Callbacks (set by main.py)
        self.on_summary_updated = None  # callable(SummaryOutput)

    @property
    def latest_output(self):
        return self._latest_output

    def set_summarizer(self, summarizer, session_id: str):
        """Set or replace the LiveSummarizer instance."""
        with self._lock:
            self._summarizer = summarizer
            self._session_id = session_id
            self._latest_output = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._buffer.on_utterance(self._on_new_utterance)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _on_new_utterance(self, u: Utterance):
        with self._pending_lock:
            self._pending.append(u)
            count = len(self._pending)
        if count >= self.BATCH_THRESHOLD:
            self._event.set()

    def _take_pending(self) -> list[Utterance]:
        with self._pending_lock:
            batch = self._pending[:]
            self._pending.clear()
            return batch

    def _loop(self):
        while self._running:
            self._event.wait(timeout=self.TIME_INTERVAL)
            self._event.clear()
            if not self._running:
                break

            pending = self._take_pending()
            if not pending:
                continue

            with self._lock:
                summarizer = self._summarizer
                session_id = self._session_id
            if not summarizer or not session_id:
                continue

            self._do_summarize(summarizer, session_id, pending)

    def _do_summarize(self, summarizer, session_id, utterances):
        try:
            from live_summary import Message
            messages = [
                Message(role=u.speaker, content=u.text, timestamp=u.timestamp)
                for u in utterances
            ]
            output = summarizer.summarize(session_id, messages)
            self._latest_output = output
            self._last_call_time = time.time()
            log.info(
                "LiveSummary updated: topics=%d, overview=%d chars, tips=%d",
                output.meta.total_topics, len(output.overview), len(output.host_tips),
            )
            if self.on_summary_updated:
                self.on_summary_updated(output)
        except Exception as e:
            log.error("LiveSummary error: %s", e)
