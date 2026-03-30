# analyzer.py
"""Analysis scheduler: debounce + latest-wins + manual trigger with streaming output."""

import logging
import threading

from openai import OpenAI

from analysis_presets import AnalysisPreset
from dialogue_buffer import DialogueBuffer, Utterance
from summary_compressor import format_utterances

log = logging.getLogger(__name__)


class AnalysisScheduler:
    """Manages AI analysis timing and API calls.

    - Auto-trigger: debounce 800ms or batch >= 3 sentences
    - Latest-wins: if API busy, queue accumulates, fires immediately after current finishes
    - Manual trigger: skips debounce, interrupts stale requests
    - Streaming output via callback
    """

    DEBOUNCE_MS = 800
    BATCH_THRESHOLD = 3

    def __init__(self, buffer: DialogueBuffer):
        self._buffer = buffer
        self._client: OpenAI | None = None
        self._model: str = ""
        self._preset: AnalysisPreset | None = None
        self._lock = threading.Lock()

        self._running = False
        self._request_event = threading.Event()
        self._thread: threading.Thread | None = None

        # State
        self._api_busy = False
        self._stale = False  # True = current request result should be discarded
        self._debounce_timer: threading.Timer | None = None
        self._pending_count = 0  # utterances arrived since last analysis

        # Callbacks (set by main.py)
        self.on_stream_chunk = None    # callable(str) - partial analysis text
        self.on_stream_done = None     # callable(str) - final analysis text
        self.on_analysis_start = None  # callable()

        # Stats
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.analysis_count = 0

    def set_client(self, client: OpenAI, model: str):
        with self._lock:
            self._client = client
            self._model = model

    def set_preset(self, preset: AnalysisPreset):
        with self._lock:
            self._preset = preset

    def start(self):
        if self._running:
            return
        self._running = True
        self._buffer.on_utterance(self._on_new_utterance)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._cancel_debounce()
        self._request_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def trigger_manual(self):
        """Manual trigger: skip debounce, mark current request stale, fire immediately."""
        self._cancel_debounce()
        if self._api_busy:
            self._stale = True
        self._request_event.set()

    # -- internal --

    def _on_new_utterance(self, _u: Utterance):
        self._pending_count += 1
        if self._api_busy:
            return  # will fire after current request completes
        if self._pending_count >= self.BATCH_THRESHOLD:
            self._cancel_debounce()
            self._request_event.set()
        else:
            self._restart_debounce()

    def _restart_debounce(self):
        self._cancel_debounce()
        self._debounce_timer = threading.Timer(
            self.DEBOUNCE_MS / 1000, self._request_event.set
        )
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _cancel_debounce(self):
        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None

    def _loop(self):
        while self._running:
            self._request_event.wait(timeout=2.0)
            self._request_event.clear()
            if not self._running:
                break
            pending = self._buffer.take_pending()
            if not pending:
                continue
            self._pending_count = 0
            self._do_analysis(pending)

            # After analysis completes, check if more arrived during API call
            if self._buffer.pending_count() > 0:
                self._request_event.set()

    def _do_analysis(self, new_utterances: list[Utterance]):
        with self._lock:
            client = self._client
            model = self._model
            preset = self._preset
        if not client or not model or not preset:
            return

        self._api_busy = True
        self._stale = False
        if self.on_analysis_start:
            self.on_analysis_start()

        system_prompt = preset.build_prompt()
        summary = self._buffer.summary

        user_parts = []
        if summary:
            user_parts.append(f"## 对话摘要\n{summary}")
        user_parts.append(f"## 最新对话\n{format_utterances(new_utterances)}")
        user_parts.append("请基于以上对话给出分析和建议。")
        user_content = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = []
            for event in stream:
                if self._stale:
                    log.debug("Analysis marked stale, discarding")
                    break
                if not event.choices:
                    if event.usage:
                        self.prompt_tokens += event.usage.prompt_tokens or 0
                        self.completion_tokens += event.usage.completion_tokens or 0
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)
                    if self.on_stream_chunk and not self._stale:
                        self.on_stream_chunk("".join(chunks))

            if not self._stale:
                final = "".join(chunks)
                self.analysis_count += 1
                if self.on_stream_done:
                    self.on_stream_done(final)
        except Exception as e:
            log.error("Analysis API error: %s", e)
        finally:
            self._api_busy = False
