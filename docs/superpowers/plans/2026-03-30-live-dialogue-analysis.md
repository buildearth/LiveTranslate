# Live Dialogue Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform LiveTranslate from real-time translation to live dialogue recognition + AI analysis/guidance for streamers.

**Architecture:** Dual-channel audio (system + mic) → independent VAD + async ASR queues → DialogueBuffer → Analysis Scheduler (debounce + latest-wins + manual trigger) + Summary Compressor → streaming AI analysis panel. Six worker threads + main Qt thread.

**Tech Stack:** Python 3.10+, PyQt6, pyaudiowpatch, faster-whisper/SenseVoice/FunASR/Qwen3-ASR, OpenAI-compatible API, threading + queue.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `dialogue_buffer.py` | **Create** | Utterance dataclass, DialogueBuffer (thread-safe sentence queue + summary storage) |
| `analysis_presets.py` | **Create** | ANALYSIS_PRESETS dict, AnalysisPreset dataclass, `build_system_prompt()` template assembler |
| `analyzer.py` | **Create** | AnalysisScheduler class (debounce, latest-wins, manual trigger, streaming API calls) |
| `summary_compressor.py` | **Create** | SummaryCompressor class (async summary compression, fixed prompt, non-blocking) |
| `audio_capture.py` | **Modify** | Split mic from mix-in to independent channel: `read_system()` / `read_mic()` |
| `vad_processor.py` | **Modify** | Add `hard_max_duration` (3s) forced split using existing backtrack logic |
| `main.py` | **Major rewrite** | Replace single pipeline thread with 6 worker threads, wire new components |
| `subtitle_overlay.py` | **Modify** | Add analysis panel (lower half), replace target language combo with scene preset combo, speaker labels |
| `control_panel.py` | **Modify** | Replace Translation tab with AI Analysis tab, add Scene Preset tab with structured editor |
| `config.yaml` | **Modify** | Add `analysis` section with defaults |
| `i18n/en.yaml` | **Modify** | Add analysis-related UI strings |
| `i18n/zh.yaml` | **Modify** | Add analysis-related UI strings |

---

### Task 1: Create DialogueBuffer

Core data structure that all other components depend on.

**Files:**
- Create: `dialogue_buffer.py`

- [ ] **Step 1: Create Utterance dataclass and DialogueBuffer class**

```python
# dialogue_buffer.py
"""Thread-safe dialogue buffer for dual-channel ASR results."""

import threading
import time
from dataclasses import dataclass, field


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
```

- [ ] **Step 2: Verify with ruff**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 dialogue_buffer.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add dialogue_buffer.py
git commit -m "feat: 新增 DialogueBuffer 对话缓冲区组件"
```

---

### Task 2: Create Analysis Presets

Scene prompt system with structured templates and advanced mode.

**Files:**
- Create: `analysis_presets.py`

- [ ] **Step 1: Create preset definitions and template assembler**

```python
# analysis_presets.py
"""Analysis scene presets and structured prompt template assembly."""

from dataclasses import dataclass, field

# Tags available for structured template editor
FOCUS_TAGS = [
    "情绪变化", "关键诉求", "矛盾点", "报价", "承诺",
    "让步信号", "关键信息", "未回答问题", "互动节奏",
]

OUTPUT_TAGS = [
    "局势判断", "建议话术", "风险提醒", "价格对比",
    "情绪分析", "问题归类", "话题建议", "信息提取",
]


@dataclass
class AnalysisPreset:
    name: str
    role: str = ""
    focus_tags: list[str] = field(default_factory=list)
    output_tags: list[str] = field(default_factory=list)
    extra_instructions: str = ""
    is_advanced: bool = False
    advanced_prompt: str = ""
    builtin: bool = False  # True for built-in presets (not editable)

    def build_prompt(self) -> str:
        """Assemble a full system prompt from structured fields or advanced text."""
        if self.is_advanced and self.advanced_prompt:
            return self.advanced_prompt
        parts = []
        if self.role:
            parts.append(f"你是一位{self.role}。")
        else:
            parts.append("你是一位专业的直播对话分析助手。")
        parts.append("根据对话摘要和最新对话内容，给出实时分析和建议。")
        if self.focus_tags:
            parts.append(f"\n重点关注：{', '.join(self.focus_tags)}。")
        if self.output_tags:
            parts.append(f"\n输出需包含：{', '.join(self.output_tags)}。")
        if self.extra_instructions:
            parts.append(f"\n{self.extra_instructions}")
        parts.append("\n请用简洁的结构化格式输出（使用 ## 标题分段），便于快速阅读。")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "focus_tags": self.focus_tags,
            "output_tags": self.output_tags,
            "extra_instructions": self.extra_instructions,
            "is_advanced": self.is_advanced,
            "advanced_prompt": self.advanced_prompt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisPreset":
        return cls(
            name=d.get("name", ""),
            role=d.get("role", ""),
            focus_tags=d.get("focus_tags", []),
            output_tags=d.get("output_tags", []),
            extra_instructions=d.get("extra_instructions", ""),
            is_advanced=d.get("is_advanced", False),
            advanced_prompt=d.get("advanced_prompt", ""),
        )


# Built-in presets
ANALYSIS_PRESETS: dict[str, AnalysisPreset] = {
    "带货直播": AnalysisPreset(
        name="带货直播",
        role="直播带货分析师",
        focus_tags=["报价", "承诺", "让步信号", "关键诉求"],
        output_tags=["价格对比", "建议话术", "风险提醒"],
        extra_instructions="注意识别对方的定价策略和限时话术，提醒砍价机会。",
        builtin=True,
    ),
    "商务谈判": AnalysisPreset(
        name="商务谈判",
        role="商务谈判顾问",
        focus_tags=["关键诉求", "矛盾点", "让步信号", "承诺"],
        output_tags=["局势判断", "建议话术", "风险提醒"],
        extra_instructions="分析双方立场差异，识别对方的底线信号和让步空间。",
        builtin=True,
    ),
    "情感连线": AnalysisPreset(
        name="情感连线",
        role="情感分析师",
        focus_tags=["情绪变化", "关键诉求", "矛盾点"],
        output_tags=["情绪分析", "建议话术", "风险提醒"],
        extra_instructions="注意识别对方的情绪转折点和语气变化，提供共情话术建议。",
        builtin=True,
    ),
    "采访访谈": AnalysisPreset(
        name="采访访谈",
        role="访谈助理",
        focus_tags=["关键信息", "未回答问题", "关键诉求"],
        output_tags=["信息提取", "话题建议", "建议话术"],
        extra_instructions="提取对方回答中的关键事实，标记被回避或未完整回答的问题。",
        builtin=True,
    ),
    "娱乐连麦": AnalysisPreset(
        name="娱乐连麦",
        role="直播互动策划",
        focus_tags=["互动节奏", "情绪变化"],
        output_tags=["话题建议", "建议话术"],
        extra_instructions="关注对话节奏和气氛变化，在冷场时及时建议新话题。",
        builtin=True,
    ),
    "客服售后": AnalysisPreset(
        name="客服售后",
        role="客服督导",
        focus_tags=["关键诉求", "情绪变化", "矛盾点"],
        output_tags=["问题归类", "建议话术", "风险提醒"],
        extra_instructions="识别客户核心问题和情绪状态，判断是否需要升级处理。",
        builtin=True,
    ),
}

SUMMARY_COMPRESS_PROMPT = (
    "将以下对话摘要和新增对话合并，生成简洁的结构化摘要。\n"
    "保留：关键事实、双方立场、已达成共识、待解决问题、情绪变化。\n"
    "删除：重复信息、无实质内容的寒暄。\n"
    "输出纯文本，不超过500字。"
)
```

- [ ] **Step 2: Verify with ruff**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 analysis_presets.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add analysis_presets.py
git commit -m "feat: 新增场景分析预设系统 (6个内置场景 + 结构化模板)"
```

---

### Task 3: Create SummaryCompressor

Async component that compresses dialogue history into rolling summaries.

**Files:**
- Create: `summary_compressor.py`

- [ ] **Step 1: Create SummaryCompressor class**

```python
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
```

- [ ] **Step 2: Verify with ruff**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 summary_compressor.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add summary_compressor.py
git commit -m "feat: 新增滚动摘要压缩器 (异步非阻塞)"
```

---

### Task 4: Create AnalysisScheduler

Core analysis component: debounce, latest-wins, manual trigger, streaming output.

**Files:**
- Create: `analyzer.py`

- [ ] **Step 1: Create AnalysisScheduler class**

```python
# analyzer.py
"""Analysis scheduler: debounce + latest-wins + manual trigger with streaming output."""

import logging
import threading
import time

from openai import OpenAI

from analysis_presets import AnalysisPreset
from dialogue_buffer import DialogueBuffer, Utterance
from summary_compressor import format_utterances

log = logging.getLogger(__name__)


class AnalysisScheduler:
    """Manages AI analysis timing and API calls.

    - Auto-trigger: debounce 800ms or batch ≥ 3 sentences
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
```

- [ ] **Step 2: Verify with ruff**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 analyzer.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add analyzer.py
git commit -m "feat: 新增分析调度器 (debounce + latest-wins + 手动触发 + 流式输出)"
```

---

### Task 5: Modify audio_capture.py for Dual Channel

Split mic from additive mix-in into independent output channel.

**Files:**
- Modify: `audio_capture.py`

- [ ] **Step 1: Add independent mic output queue**

In `__init__`, after `self.audio_queue = queue.Queue(maxsize=100)`, add a second queue:

```python
self.mic_queue = queue.Queue(maxsize=100)
```

- [ ] **Step 2: Add `get_mic_audio()` method**

After the existing `get_audio()` method, add:

```python
def get_mic_audio(self, timeout=1.0):
    """Return mic-only audio chunk, or None if timeout."""
    try:
        return self.mic_queue.get(timeout=timeout)
    except queue.Empty:
        return None
```

- [ ] **Step 3: Modify `_read_loop()` to output mic separately**

In `_read_loop()`, find the mixing section where `mic_chunk` is added to `loopback_audio`. Change the logic so that instead of mixing (`audio = loopback_audio + mic_chunk`), the mic chunk is put into `mic_queue` independently:

Replace the mixing block (the section that does `audio = loopback_audio + mic_chunk`) with:

```python
# Put mic audio to separate queue (independent channel)
if len(mic_chunk) > 0 and mic_rms is not None:
    try:
        self.mic_queue.put_nowait(mic_chunk)
    except queue.Full:
        try:
            self.mic_queue.get_nowait()
        except queue.Empty:
            pass
        self.mic_queue.put_nowait(mic_chunk)

# System audio goes to main queue (no mixing)
audio = loopback_audio
```

Keep `mic_rms` calculation before this block so monitor still works. The `get_audio()` return value remains `(audio, mic_rms)` — the `mic_rms` is still useful for the monitor bar display.

- [ ] **Step 4: Handle mic-only mode**

When `_loopback_disabled` is True and mic is active, the existing code generates silence chunks. Keep this behavior for the system audio queue. The mic queue will still get real mic data from `_mic_buf` draining.

- [ ] **Step 5: Verify with ruff and commit**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 audio_capture.py`

```bash
git add audio_capture.py
git commit -m "feat: 音频捕获双通道分离 (系统音频 + 麦克风独立输出)"
```

---

### Task 6: Modify vad_processor.py for Hard Max Duration

Add forced 3-second split to prevent buffer bloat during fast speech.

**Files:**
- Modify: `vad_processor.py`

- [ ] **Step 1: Add `hard_max_duration` parameter**

In `__init__`, after the existing `max_speech_duration` parameter, add:

```python
self._hard_max_samples = int(3.0 * sample_rate)  # 3s hard limit
```

- [ ] **Step 2: Add hard max check in `process_chunk()`**

In `process_chunk()`, find the existing check `if self._speech_samples >= self._max_speech_samples`. Add a check **before** it:

```python
# Hard max: force split at 3s regardless, using best pause point
if self._speech_samples >= self._hard_max_samples:
    seg = self._split_at_best_pause()
    if seg is not None:
        return seg
```

This uses the existing `_split_at_best_pause()` backtrack logic so the split happens at a natural pause point.

- [ ] **Step 3: Make hard_max configurable via `update_settings()`**

In `update_settings()`, add:

```python
if "hard_max_duration" in settings:
    self._hard_max_samples = int(float(settings["hard_max_duration"]) * self._sample_rate)
```

- [ ] **Step 4: Verify with ruff and commit**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 vad_processor.py`

```bash
git add vad_processor.py
git commit -m "feat: VAD 增加3秒硬性切分上限 (快速语音不积压)"
```

---

### Task 7: Update config.yaml and i18n

Add analysis-related defaults and UI strings.

**Files:**
- Modify: `config.yaml`
- Modify: `i18n/zh.yaml`
- Modify: `i18n/en.yaml`

- [ ] **Step 1: Add analysis section to config.yaml**

After the `translation:` section, add:

```yaml
analysis:
  debounce_ms: 800
  batch_threshold: 3
  summary_threshold: 15
  hard_max_vad_duration: 3.0
  default_preset: "带货直播"
```

- [ ] **Step 2: Add Chinese UI strings to i18n/zh.yaml**

Add the following entries (find the appropriate location in the existing file structure):

```yaml
# Analysis
analysis_panel: "AI 分析"
analysis_trigger: "分析"
scene_label: "场景"
speaker_other: "对方"
speaker_self: "我方"
analysis_count: "分析"
summary_tokens: "摘要"
speech_skipped: "...语速过快，部分跳过..."
preset_tab: "场景预设"
preset_name: "场景名称"
preset_role: "AI角色"
preset_focus: "关注重点"
preset_output: "输出包含"
preset_extra: "额外指令"
preset_advanced: "高级模式"
preset_add: "新增"
preset_edit: "编辑"
preset_delete: "删除"
preset_duplicate: "复制"
preset_builtin: "(内置)"
mic_device: "麦克风设备"
system_device: "系统音频设备"
```

- [ ] **Step 3: Add English UI strings to i18n/en.yaml**

```yaml
# Analysis
analysis_panel: "AI Analysis"
analysis_trigger: "Analyze"
scene_label: "Scene"
speaker_other: "Other"
speaker_self: "Self"
analysis_count: "Analysis"
summary_tokens: "Summary"
speech_skipped: "...speech too fast, partially skipped..."
preset_tab: "Scene Presets"
preset_name: "Scene Name"
preset_role: "AI Role"
preset_focus: "Focus On"
preset_output: "Output Includes"
preset_extra: "Extra Instructions"
preset_advanced: "Advanced Mode"
preset_add: "Add"
preset_edit: "Edit"
preset_delete: "Delete"
preset_duplicate: "Duplicate"
preset_builtin: "(Built-in)"
mic_device: "Microphone"
system_device: "System Audio"
```

- [ ] **Step 4: Commit**

```bash
git add config.yaml i18n/zh.yaml i18n/en.yaml
git commit -m "feat: 新增分析相关配置和 i18n 文案"
```

---

### Task 8: Modify subtitle_overlay.py — UI Layout

Replace translation UI with analysis UI: speaker labels, scene combo, analysis panel.

**Files:**
- Modify: `subtitle_overlay.py`

- [ ] **Step 1: Modify ChatMessage to show speaker labels instead of translations**

In `ChatMessage.__init__`, the current layout shows original text + translation label. Change it to show speaker-tagged original text only.

Replace the `_build_header_html` method to include speaker tag:

```python
def _build_header_html(self, s=None):
    if s is None:
        s = ChatMessage._current_style
    # self._speaker is set via a new parameter
    speaker = getattr(self, '_speaker', '')
    speaker_color = "#5B9BD5" if speaker == "对方" else "#70AD47"  # blue vs green
    ts_color = s.get("timestamp_color", "#888888")
    return (
        f'<span style="color:{speaker_color};font-weight:bold;">[{speaker}]</span> '
        f'<span style="color:{ts_color};font-size:{max(8, s.get("font_size", 14) - 2)}pt;">'
        f'{self._timestamp}</span>'
    )
```

Add `speaker` parameter to `ChatMessage.__init__`:

```python
def __init__(self, msg_id, timestamp, original, source_lang, asr_ms, speaker="", parent=None):
    # ... existing init ...
    self._speaker = speaker
```

Remove the translation label (`self._tl_label`) and related methods (`set_translation`, `update_streaming`, `_flush_streaming`). These will be replaced by the analysis panel.

- [ ] **Step 2: Replace target language combo with scene preset combo in DragHandle**

In `DragHandle.__init__`, find the row2b section where `_target_lang` combo is created (around line 775-791). Replace it with a scene preset combo:

```python
# Scene preset combo (replaces target language)
scene_label = QLabel(t("scene_label"))
scene_label.setStyleSheet("color:white; font-size:10px;")
row2b.addWidget(scene_label)

self._scene_combo = QComboBox()
self._scene_combo.setFixedHeight(18)
self._scene_combo.setStyleSheet("QComboBox { font-size: 10px; }")
self._scene_combo.currentIndexChanged.connect(self._on_scene_changed)
row2b.addWidget(self._scene_combo)
```

Add signal to DragHandle:

```python
scene_changed = pyqtSignal(str)  # preset name
```

Add methods:

```python
def set_scenes(self, preset_names: list[str], active: str = ""):
    self._scene_combo.blockSignals(True)
    self._scene_combo.clear()
    for name in preset_names:
        self._scene_combo.addItem(name, name)
    if active:
        idx = self._scene_combo.findData(active)
        if idx >= 0:
            self._scene_combo.setCurrentIndex(idx)
    self._scene_combo.blockSignals(False)

def _on_scene_changed(self, index):
    name = self._scene_combo.itemData(index)
    if name:
        self.scene_changed.emit(name)
```

Also remove `source_language_changed` signal and the source language combo (no longer needed — ASR language is auto-detected from both channels).

- [ ] **Step 3: Add analysis panel to SubtitleOverlay**

In `SubtitleOverlay.__init__`, after the scroll area (`_scroll`), add the analysis panel:

```python
# -- Analysis panel (lower half) --
analysis_bar = QHBoxLayout()
analysis_title = QLabel(t("analysis_panel"))
analysis_title.setStyleSheet("color:white; font-weight:bold; font-size:12px;")
analysis_bar.addWidget(analysis_title)
analysis_bar.addStretch()

self._analyze_btn = QPushButton(t("analysis_trigger"))
self._analyze_btn.setFixedHeight(22)
self._analyze_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 8px; }")
self._analyze_btn.clicked.connect(self._on_analyze_clicked)
analysis_bar.addWidget(self._analyze_btn)

container_layout.addLayout(analysis_bar)

self._analysis_text = QTextEdit()
self._analysis_text.setReadOnly(True)
self._analysis_text.setMinimumHeight(120)
self._analysis_text.setStyleSheet(
    "QTextEdit { background: rgba(0,0,0,180); color: white; "
    "border: 1px solid #333; font-size: 12px; padding: 4px; }"
)
container_layout.addWidget(self._analysis_text)
```

Add signals:

```python
analyze_requested = pyqtSignal()           # manual trigger
scene_changed = pyqtSignal(str)            # preset name
update_analysis_signal = pyqtSignal(str)   # streaming analysis text
finish_analysis_signal = pyqtSignal(str)   # final analysis text
```

Connect signals:

```python
self.update_analysis_signal.connect(self._on_update_analysis)
self.finish_analysis_signal.connect(self._on_finish_analysis)
# Forward DragHandle scene signal
self._handle.scene_changed.connect(self.scene_changed.emit)
```

Add slots:

```python
def _on_analyze_clicked(self):
    self.analyze_requested.emit()

def _on_update_analysis(self, text):
    self._analysis_text.setMarkdown(text)
    # Auto-scroll to bottom
    sb = self._analysis_text.verticalScrollBar()
    sb.setValue(sb.maximum())

def _on_finish_analysis(self, text):
    self._analysis_text.setMarkdown(text)

def update_analysis(self, text):
    """Thread-safe streaming analysis update."""
    self.update_analysis_signal.emit(text)

def finish_analysis(self, text):
    """Thread-safe final analysis update."""
    self.finish_analysis_signal.emit(text)
```

- [ ] **Step 4: Update add_message_signal to include speaker**

Change signal signature:

```python
add_message_signal = pyqtSignal(int, str, str, str, float, str)  # + speaker
```

Update `_on_add_message` to pass speaker to ChatMessage:

```python
def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms, speaker):
    msg = ChatMessage(msg_id, timestamp, original, source_lang, asr_ms, speaker=speaker)
    # ... rest same
```

Update `add_message` public method:

```python
def add_message(self, msg_id, timestamp, original, source_lang, asr_ms, speaker=""):
    self.add_message_signal.emit(msg_id, timestamp, original, source_lang, asr_ms, speaker)
```

- [ ] **Step 5: Verify with ruff and commit**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 subtitle_overlay.py`

```bash
git add subtitle_overlay.py
git commit -m "feat: overlay 改造 (说话人标签 + 场景预设下拉 + AI分析面板)"
```

---

### Task 9: Modify control_panel.py — Settings Tabs

Replace Translation tab with Analysis tab, add Scene Preset editor tab.

**Files:**
- Modify: `control_panel.py`

- [ ] **Step 1: Replace Translation tab with Analysis tab**

Replace `_create_translation_tab()` method. Keep the models group (QListWidget + Add/Edit/Duplicate/Remove buttons) since model management is still needed. Replace the prompt group with analysis settings:

```python
def _create_analysis_tab(self):
    """Tab for AI analysis model and settings."""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Models group (reuse existing model management code from _create_translation_tab)
    models_group = self._create_models_group()  # extract existing code into helper
    layout.addWidget(models_group)

    # Analysis settings group
    analysis_group = QGroupBox(t("analysis_panel"))
    ag_layout = QFormLayout(analysis_group)

    # Debounce
    self._debounce_spin = QSpinBox()
    self._debounce_spin.setRange(200, 3000)
    self._debounce_spin.setSuffix(" ms")
    self._debounce_spin.setValue(800)
    self._debounce_spin.valueChanged.connect(self._auto_save)
    ag_layout.addRow("Debounce:", self._debounce_spin)

    # Batch threshold
    self._batch_spin = QSpinBox()
    self._batch_spin.setRange(1, 10)
    self._batch_spin.setValue(3)
    self._batch_spin.valueChanged.connect(self._auto_save)
    ag_layout.addRow(t("analysis_count") + ":", self._batch_spin)

    # Summary threshold
    self._summary_spin = QSpinBox()
    self._summary_spin.setRange(5, 50)
    self._summary_spin.setValue(15)
    self._summary_spin.valueChanged.connect(self._auto_save)
    ag_layout.addRow(t("summary_tokens") + ":", self._summary_spin)

    # Timeout
    self._timeout_spin = QSpinBox()
    self._timeout_spin.setRange(1, 60)
    self._timeout_spin.setSuffix(" s")
    self._timeout_spin.setValue(10)
    self._timeout_spin.valueChanged.connect(self._auto_save)
    ag_layout.addRow("Timeout:", self._timeout_spin)

    layout.addWidget(analysis_group)
    layout.addStretch()
    return tab
```

- [ ] **Step 2: Add Scene Preset editor tab**

```python
def _create_preset_tab(self):
    """Tab for scene preset structured editor."""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Preset list
    list_row = QHBoxLayout()
    self._preset_list = QListWidget()
    self._preset_list.currentRowChanged.connect(self._on_preset_selected)
    list_row.addWidget(self._preset_list)

    btn_col = QVBoxLayout()
    self._preset_add_btn = QPushButton(t("preset_add"))
    self._preset_add_btn.clicked.connect(self._on_preset_add)
    btn_col.addWidget(self._preset_add_btn)

    self._preset_dup_btn = QPushButton(t("preset_duplicate"))
    self._preset_dup_btn.clicked.connect(self._on_preset_duplicate)
    btn_col.addWidget(self._preset_dup_btn)

    self._preset_del_btn = QPushButton(t("preset_delete"))
    self._preset_del_btn.clicked.connect(self._on_preset_delete)
    btn_col.addWidget(self._preset_del_btn)

    btn_col.addStretch()
    list_row.addLayout(btn_col)
    layout.addLayout(list_row)

    # Editor area
    editor_group = QGroupBox(t("preset_edit"))
    eg_layout = QFormLayout(editor_group)

    self._pe_name = QLineEdit()
    eg_layout.addRow(t("preset_name") + ":", self._pe_name)

    self._pe_role = QLineEdit()
    eg_layout.addRow(t("preset_role") + ":", self._pe_role)

    # Focus tags as checkboxes
    from analysis_presets import FOCUS_TAGS, OUTPUT_TAGS
    self._pe_focus_checks = {}
    focus_widget = QWidget()
    focus_flow = QHBoxLayout(focus_widget)
    focus_flow.setContentsMargins(0, 0, 0, 0)
    for tag in FOCUS_TAGS:
        cb = QCheckBox(tag)
        self._pe_focus_checks[tag] = cb
        focus_flow.addWidget(cb)
    eg_layout.addRow(t("preset_focus") + ":", focus_widget)

    # Output tags as checkboxes
    self._pe_output_checks = {}
    output_widget = QWidget()
    output_flow = QHBoxLayout(output_widget)
    output_flow.setContentsMargins(0, 0, 0, 0)
    for tag in OUTPUT_TAGS:
        cb = QCheckBox(tag)
        self._pe_output_checks[tag] = cb
        output_flow.addWidget(cb)
    eg_layout.addRow(t("preset_output") + ":", output_widget)

    self._pe_extra = QTextEdit()
    self._pe_extra.setMaximumHeight(60)
    eg_layout.addRow(t("preset_extra") + ":", self._pe_extra)

    # Advanced mode toggle
    self._pe_advanced_check = QCheckBox(t("preset_advanced"))
    self._pe_advanced_check.toggled.connect(self._on_preset_advanced_toggle)
    eg_layout.addRow(self._pe_advanced_check)

    self._pe_advanced_edit = QTextEdit()
    self._pe_advanced_edit.setMaximumHeight(120)
    self._pe_advanced_edit.setVisible(False)
    eg_layout.addRow(self._pe_advanced_edit)

    layout.addWidget(editor_group)

    # Save button for preset edits
    save_btn = QPushButton(t("preset_edit"))
    save_btn.clicked.connect(self._on_preset_save)
    layout.addWidget(save_btn)

    return tab
```

- [ ] **Step 3: Add preset management methods**

```python
def _load_presets(self):
    """Load built-in + user presets into list."""
    from analysis_presets import ANALYSIS_PRESETS, AnalysisPreset
    self._presets = {}
    # Built-in
    for name, preset in ANALYSIS_PRESETS.items():
        self._presets[name] = preset
    # User custom from settings
    saved = self._current_settings.get("analysis_presets", [])
    for d in saved:
        p = AnalysisPreset.from_dict(d)
        self._presets[p.name] = p
    self._refresh_preset_list()

def _refresh_preset_list(self):
    self._preset_list.clear()
    for name, preset in self._presets.items():
        suffix = " " + t("preset_builtin") if preset.builtin else ""
        self._preset_list.addItem(name + suffix)

def _on_preset_selected(self, row):
    if row < 0:
        return
    name = list(self._presets.keys())[row]
    preset = self._presets[name]
    self._pe_name.setText(preset.name)
    self._pe_name.setReadOnly(preset.builtin)
    self._pe_role.setText(preset.role)
    for tag, cb in self._pe_focus_checks.items():
        cb.setChecked(tag in preset.focus_tags)
    for tag, cb in self._pe_output_checks.items():
        cb.setChecked(tag in preset.output_tags)
    self._pe_extra.setPlainText(preset.extra_instructions)
    self._pe_advanced_check.setChecked(preset.is_advanced)
    self._pe_advanced_edit.setPlainText(preset.advanced_prompt)
    self._pe_advanced_edit.setVisible(preset.is_advanced)

def _on_preset_advanced_toggle(self, checked):
    self._pe_advanced_edit.setVisible(checked)

def _on_preset_add(self):
    from analysis_presets import AnalysisPreset
    name = f"自定义{len(self._presets) + 1}"
    self._presets[name] = AnalysisPreset(name=name)
    self._refresh_preset_list()
    self._preset_list.setCurrentRow(len(self._presets) - 1)

def _on_preset_duplicate(self):
    row = self._preset_list.currentRow()
    if row < 0:
        return
    from analysis_presets import AnalysisPreset
    src = list(self._presets.values())[row]
    d = src.to_dict()
    d["name"] = src.name + " (副本)"
    new_preset = AnalysisPreset.from_dict(d)
    self._presets[new_preset.name] = new_preset
    self._refresh_preset_list()

def _on_preset_delete(self):
    row = self._preset_list.currentRow()
    if row < 0:
        return
    name = list(self._presets.keys())[row]
    if self._presets[name].builtin:
        return  # can't delete built-in
    del self._presets[name]
    self._refresh_preset_list()
    self._save_presets()

def _on_preset_save(self):
    row = self._preset_list.currentRow()
    if row < 0:
        return
    old_name = list(self._presets.keys())[row]
    preset = self._presets[old_name]
    if preset.builtin:
        return

    new_name = self._pe_name.text().strip()
    preset.name = new_name or old_name
    preset.role = self._pe_role.text().strip()
    preset.focus_tags = [t for t, cb in self._pe_focus_checks.items() if cb.isChecked()]
    preset.output_tags = [t for t, cb in self._pe_output_checks.items() if cb.isChecked()]
    preset.extra_instructions = self._pe_extra.toPlainText().strip()
    preset.is_advanced = self._pe_advanced_check.isChecked()
    preset.advanced_prompt = self._pe_advanced_edit.toPlainText().strip()

    if new_name != old_name:
        del self._presets[old_name]
        self._presets[new_name] = preset

    self._refresh_preset_list()
    self._save_presets()

def _save_presets(self):
    """Save user presets to settings (excluding built-in)."""
    user_presets = [p.to_dict() for p in self._presets.values() if not p.builtin]
    self._current_settings["analysis_presets"] = user_presets
    self._auto_save()

def get_preset(self, name: str):
    return self._presets.get(name)

def get_all_preset_names(self) -> list[str]:
    return list(self._presets.keys())
```

- [ ] **Step 4: Update tab creation in `__init__`**

Replace the tab creation section:

```python
# Replace:
#   tabs.addTab(self._create_translation_tab(), t("translation"))
# With:
tabs.addTab(self._create_analysis_tab(), t("analysis_panel"))
tabs.addTab(self._create_preset_tab(), t("preset_tab"))
```

- [ ] **Step 5: Update VAD/ASR tab for dual device selection**

In `_create_vad_tab()`, add a mic device combo after the existing system audio device combo:

```python
# Mic device combo
mic_label = QLabel(t("mic_device"))
self._mic_device_combo = QComboBox()
self._mic_device_combo.currentTextChanged.connect(self._auto_save)
# Populate with available input devices
layout.addRow(mic_label, self._mic_device_combo)
```

- [ ] **Step 6: Update `_apply_settings()` to include new fields**

Add to the settings dict construction:

```python
s["debounce_ms"] = self._debounce_spin.value()
s["batch_threshold"] = self._batch_spin.value()
s["summary_threshold"] = self._summary_spin.value()
s["mic_device"] = self._mic_device_combo.currentText()
```

- [ ] **Step 7: Verify with ruff and commit**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 control_panel.py`

```bash
git add control_panel.py
git commit -m "feat: 设置面板改造 (AI分析tab + 场景预设编辑器 + 双设备选择)"
```

---

### Task 10: Rewrite main.py Pipeline

The largest change: replace single pipeline thread with multi-threaded async architecture.

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add new imports and instance variables**

At the top of `LiveTranslateApp.__init__`, add:

```python
from dialogue_buffer import DialogueBuffer
from analyzer import AnalysisScheduler
from summary_compressor import SummaryCompressor
from analysis_presets import ANALYSIS_PRESETS

# In __init__:
self._dialogue_buffer = DialogueBuffer()
self._analyzer = AnalysisScheduler(self._dialogue_buffer)
self._compressor = SummaryCompressor(self._dialogue_buffer)
```

- [ ] **Step 2: Create dual VAD + ASR queue architecture**

Add new instance variables:

```python
# Dual channel VAD
self._vad_system = None   # VAD for system audio
self._vad_mic = None      # VAD for mic audio

# ASR queues
self._asr_queue_system = queue.Queue(maxsize=10)
self._asr_queue_mic = queue.Queue(maxsize=10)

# Thread references
self._audio_thread_system = None
self._audio_thread_mic = None
self._asr_thread_system = None
self._asr_thread_mic = None
```

- [ ] **Step 3: Create audio pipeline thread functions**

```python
def _audio_pipeline_system(self):
    """Thread: read system audio → VAD → push segments to ASR queue."""
    silence_chunk = np.zeros(int(0.032 * 16000), dtype=np.float32)
    while self._running:
        item = self._audio.get_audio(timeout=1.0)
        if item is None:
            # Inject silence for speech end detection
            if self._vad_system._is_speaking and not self._paused:
                n = int(self._vad_system._silence_limit / 0.032) + 2
                for _ in range(n):
                    seg = self._vad_system.process_chunk(silence_chunk)
                    if seg is not None:
                        self._enqueue_asr(self._asr_queue_system, seg, "对方")
                        break
            continue
        chunk, mic_rms = item
        if self._paused:
            continue
        rms = float(np.sqrt(np.mean(chunk**2)))
        if self._overlay:
            self._overlay.update_monitor(rms, self._vad_system.last_confidence, mic_rms)
        seg = self._vad_system.process_chunk(chunk)
        if seg is not None:
            self._enqueue_asr(self._asr_queue_system, seg, "对方")

def _audio_pipeline_mic(self):
    """Thread: read mic audio → VAD → push segments to ASR queue."""
    silence_chunk = np.zeros(int(0.032 * 16000), dtype=np.float32)
    while self._running:
        item = self._audio.get_mic_audio(timeout=1.0)
        if item is None:
            if self._vad_mic._is_speaking and not self._paused:
                n = int(self._vad_mic._silence_limit / 0.032) + 2
                for _ in range(n):
                    seg = self._vad_mic.process_chunk(silence_chunk)
                    if seg is not None:
                        self._enqueue_asr(self._asr_queue_mic, seg, "我方")
                        break
            continue
        if self._paused:
            continue
        seg = self._vad_mic.process_chunk(item)
        if seg is not None:
            self._enqueue_asr(self._asr_queue_mic, seg, "我方")

def _enqueue_asr(self, q: queue.Queue, segment, speaker: str):
    """Push segment to ASR queue with backpressure handling."""
    try:
        q.put_nowait((segment, speaker))
    except queue.Full:
        # Merge: drop oldest, keep newest
        try:
            old_seg, old_speaker = q.get_nowait()
            merged = np.concatenate([old_seg, segment])
            q.put_nowait((merged, speaker))
            log.warning("ASR queue full for %s, merged segments", speaker)
        except queue.Empty:
            q.put_nowait((segment, speaker))
```

- [ ] **Step 4: Create ASR worker thread functions**

```python
def _asr_worker(self, asr_queue: queue.Queue, speaker_label: str):
    """Thread: consume ASR queue → transcribe → push to dialogue buffer + UI."""
    while self._running:
        try:
            item = asr_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        segment, speaker = item
        seg_len = len(segment) / 16000

        if not self._asr_ready or self._asr is None:
            log.debug("ASR not ready, dropping %s segment", speaker)
            continue

        asr_start = time.perf_counter()
        with self._asr_lock:
            try:
                result = self._asr.transcribe(segment)
            except Exception as e:
                log.error("ASR error (%s): %s", speaker, e)
                continue
        asr_ms = (time.perf_counter() - asr_start) * 1000

        text = (result.get("text") or "").strip()
        if not text:
            continue

        # Text density filter
        alnum_count = sum(1 for c in text if c.isalnum())
        if seg_len >= 2.0 and alnum_count <= 3:
            log.debug("Density filter: discarded '%s' (%.1fs)", text, seg_len)
            continue

        # Add to dialogue buffer (notifies analyzer automatically)
        self._dialogue_buffer.add(speaker, text)

        # Update UI — show in scrolling area
        msg_id = self._next_msg_id()
        ts = time.strftime("%H:%M:%S")
        source_lang = result.get("language", "")
        if self._overlay:
            self._overlay.add_message(msg_id, ts, text, source_lang, asr_ms, speaker)
```

- [ ] **Step 5: Wire analyzer callbacks to overlay signals**

```python
def _setup_analyzer(self):
    """Connect analyzer streaming output to overlay panel."""
    from translator import make_openai_client

    def on_stream_chunk(text):
        if self._overlay:
            self._overlay.update_analysis(text)

    def on_stream_done(text):
        if self._overlay:
            self._overlay.finish_analysis(text)

    self._analyzer.on_stream_chunk = on_stream_chunk
    self._analyzer.on_stream_done = on_stream_done

def _update_analyzer_client(self, model_cfg: dict):
    """Called when active model changes. Updates analyzer + compressor clients."""
    from translator import make_openai_client
    client = make_openai_client(
        model_cfg["api_base"], model_cfg["api_key"],
        model_cfg.get("proxy", "none"),
        timeout=model_cfg.get("timeout", 10),
    )
    model_name = model_cfg["model"]
    self._analyzer.set_client(client, model_name)
    self._compressor.set_client(client, model_name)
```

- [ ] **Step 6: Rewrite `start()` and `stop()`**

```python
def start(self):
    if self._running:
        return
    self._running = True
    self._paused = False

    # Initialize dual VAD
    vad_cfg = {
        "threshold": self._vad_threshold,
        "min_speech_duration": self._min_speech,
        "max_speech_duration": self._max_speech,
    }
    self._vad_system = VADProcessor(**self._vad_init_args())
    self._vad_mic = VADProcessor(**self._vad_init_args())

    # Start audio capture
    self._audio.start()

    # Start audio pipeline threads
    self._audio_thread_system = threading.Thread(
        target=self._audio_pipeline_system, daemon=True)
    self._audio_thread_mic = threading.Thread(
        target=self._audio_pipeline_mic, daemon=True)
    self._audio_thread_system.start()
    self._audio_thread_mic.start()

    # Start ASR worker threads
    self._asr_thread_system = threading.Thread(
        target=self._asr_worker,
        args=(self._asr_queue_system, "对方"), daemon=True)
    self._asr_thread_mic = threading.Thread(
        target=self._asr_worker,
        args=(self._asr_queue_mic, "我方"), daemon=True)
    self._asr_thread_system.start()
    self._asr_thread_mic.start()

    # Start analyzer + compressor
    self._setup_analyzer()
    self._analyzer.start()
    self._compressor.start()

    log.info("Dual-channel pipeline started (6 threads)")

def stop(self):
    self._running = False
    self._audio.stop()

    # Join all threads
    for t in [self._audio_thread_system, self._audio_thread_mic,
              self._asr_thread_system, self._asr_thread_mic]:
        if t:
            t.join(timeout=3)

    self._analyzer.stop()
    self._compressor.stop()

    log.info("Pipeline stopped")
```

- [ ] **Step 7: Connect overlay signals**

In the section where overlay signals are connected (around line 1243+), add:

```python
# Scene preset change
overlay.scene_changed.connect(live_trans._on_scene_changed)
# Manual analysis trigger
overlay.analyze_requested.connect(live_trans._analyzer.trigger_manual)
```

Add handler:

```python
def _on_scene_changed(self, preset_name: str):
    preset = self._panel.get_preset(preset_name)
    if preset:
        self._analyzer.set_preset(preset)
        log.info("Scene preset changed to: %s", preset_name)
```

- [ ] **Step 8: Remove old translation pipeline code**

Remove:
- `_translate_async()` method
- `_process_segment()` (replaced by `_asr_worker`)
- `_do_interim_asr()` and `_process_interim_final()` (incremental ASR — replaced by async ASR queue)
- `_pipeline_loop()` (replaced by 4 separate thread functions)
- References to `self._translator` (replaced by `self._analyzer`)
- `ThreadPoolExecutor` for translation (no longer needed)

Keep:
- ASR engine loading/switching (`_switch_asr_engine`, `_asr_lock`)
- Model management signals
- Pause/resume logic (set `_paused` flag, checked by audio threads)
- `_next_msg_id()` helper

- [ ] **Step 9: Update `_on_model_changed` to use analyzer**

Replace the translator initialization with analyzer client update:

```python
def _on_model_changed(self, model_cfg: dict):
    self._update_analyzer_client(model_cfg)
    # ... keep other model-related logic
```

- [ ] **Step 10: Verify with ruff and commit**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 main.py`

```bash
git add main.py
git commit -m "feat: 管道重写为双通道异步架构 (6工作线程 + AI分析调度)"
```

---

### Task 11: Integration and Deferred Init

Wire everything together in the startup flow.

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update `_deferred_init` to load presets and set scene**

In the `_deferred_init` function (called via `QTimer.singleShot(100)`):

```python
def _deferred_init():
    panel._apply_settings()
    panel._load_presets()

    # Set models
    models = panel.get_settings().get("models", [])
    active_idx = panel.get_settings().get("active_model", 0)
    overlay.set_models(models, active_idx)

    # Set scene presets on overlay
    preset_names = panel.get_all_preset_names()
    default_preset = panel.get_settings().get("default_preset", "带货直播")
    overlay._handle.set_scenes(preset_names, default_preset)

    # Set initial preset on analyzer
    preset = panel.get_preset(default_preset)
    if preset:
        live_trans._analyzer.set_preset(preset)

    # Apply style
    style = panel.get_settings().get("style")
    if style:
        overlay.apply_style(style)

    # Start active model
    active_model = panel.get_active_model()
    if active_model:
        live_trans._on_model_changed(active_model)
```

- [ ] **Step 2: Connect clear signal to dialogue buffer**

```python
overlay.clear_signal.connect(lambda: live_trans._dialogue_buffer.clear())
```

- [ ] **Step 3: Update settings_changed handler to propagate analysis settings**

In `_on_settings_changed`:

```python
def _on_settings_changed(self, settings: dict):
    # ... existing settings handling ...

    # Update analyzer settings
    if "debounce_ms" in settings:
        self._analyzer.DEBOUNCE_MS = settings["debounce_ms"]
    if "batch_threshold" in settings:
        self._analyzer.BATCH_THRESHOLD = settings["batch_threshold"]
    if "summary_threshold" in settings:
        self._compressor._threshold = settings["summary_threshold"]

    # Update mic device
    if "mic_device" in settings:
        self._audio.set_mic_device(settings["mic_device"])
```

- [ ] **Step 4: Verify full app with ruff**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 *.py`

```bash
git add main.py
git commit -m "feat: 集成所有组件, 完成启动流程和信号连接"
```

---

### Task 12: Final Cleanup and Smoke Test

Remove unused code and verify the app launches.

**Files:**
- Modify: `main.py`
- Possibly: `subtitle_overlay.py`

- [ ] **Step 1: Remove `translator.py` import references in main.py**

Replace any `from translator import Translator` with the new imports. Keep `make_openai_client` import since it's shared.

- [ ] **Step 2: Clean up unused signals**

In `subtitle_overlay.py`, remove:
- `update_translation_signal` (no longer used)
- `update_streaming_signal` (replaced by `update_analysis_signal`)
- `target_language_changed` signal (replaced by `scene_changed`)
- `source_language_changed` signal (no longer needed)

Keep `translator.py` file in the repo (not deleted) — it may still be useful for the benchmark tab or future use.

- [ ] **Step 3: Run ruff on all files**

Run: `python -m ruff check --select F,E,W --ignore E501,E402 *.py`
Fix any remaining issues.

- [ ] **Step 4: Smoke test**

Run: `.venv/Scripts/python.exe main.py`

Verify:
- App launches without crash
- Overlay shows: header with scene combo + model combo
- Upper area ready for scrolling dialogue
- Lower area shows analysis panel with "分析" button
- Settings panel opens with new tabs (AI Analysis, Scene Preset)
- Scene presets show 6 built-in presets

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: 直播对话实时分析系统完成 (双通道ASR + AI分析 + 场景预设)"
```

---

## Execution Order & Dependencies

```
Task 1 (DialogueBuffer) ──────┐
Task 2 (AnalysisPresets) ──────┼──→ Task 4 (Analyzer) ─────┐
Task 3 (SummaryCompressor) ────┘                            │
                                                            │
Task 5 (audio_capture dual) ───┐                            │
Task 6 (vad hard max) ─────────┤                            │
Task 7 (config + i18n) ────────┤                            │
Task 8 (overlay UI) ───────────┼──→ Task 10 (main.py) ──→ Task 11 (integration) ──→ Task 12 (cleanup)
Task 9 (control_panel) ────────┘
```

Tasks 1-3 are independent and can be done in parallel.
Tasks 5-9 are independent and can be done in parallel.
Task 4 depends on Tasks 1-2.
Task 10 depends on all prior tasks.
Tasks 11-12 are sequential after Task 10.
