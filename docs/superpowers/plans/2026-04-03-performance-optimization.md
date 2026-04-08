# 性能优化：解决卡死与识别延迟 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the app freezing and ASR recognition getting stuck for long periods by eliminating thread contention, adding timeouts, and improving pipeline throughput.

**Architecture:** The root causes are: (1) a single `_asr_lock` serializing two ASR worker threads during inference, (2) blocking `speaker_id.identify()` inside the ASR worker thread, (3) an uninterruptible 8-second `time.sleep()` in the analyzer, (4) no timeouts on ASR inference or summary compressor API calls. Fix each independently.

**Tech Stack:** Python threading, queue, numpy, PyQt6 signals

---

## File Structure

| File | Changes |
|------|---------|
| `main.py` | Remove `_asr_lock` from transcribe path; move speaker ID to async; increase queue size |
| `analyzer.py` | Replace `time.sleep()` with interruptible `Event.wait()` |
| `summary_compressor.py` | Add timeout to API call |
| `dialogue_buffer.py` | Add max size cap to utterance list |

---

### Task 1: Remove `_asr_lock` from transcribe() — eliminate ASR thread serialization

The `_asr_lock` is held during the entire `transcribe()` call (5-30s). Two ASR threads share this lock, so they serialize. The lock only needs to protect model switching, not inference. Each ASR worker thread calls `transcribe()` on the **same** `self._asr` object, but faster-whisper/SenseVoice/FunASR inference is internally thread-safe for read-only inference (model weights are frozen). The lock is only needed when `_asr` is being reassigned during engine switch.

**Files:**
- Modify: `main.py:555-560` (transcribe call)
- Modify: `main.py:373-376` (engine switch lock)
- Modify: `main.py:462` (engine assignment)

- [ ] **Step 1: Replace `_asr_lock` usage in `_asr_worker` with a local reference snapshot**

In `main.py`, find the `_asr_worker` method. Replace the lock-protected transcribe block:

```python
# BEFORE (main.py:554-560):
            asr_start = time.perf_counter()
            with self._asr_lock:
                try:
                    result = self._asr.transcribe(segment)
                except Exception as e:
                    log.error("ASR error (%s): %s", channel_name, e)
                    continue
            asr_ms = (time.perf_counter() - asr_start) * 1000

# AFTER:
            asr_start = time.perf_counter()
            asr_ref = self._asr  # snapshot; may become None during switch
            if asr_ref is None:
                continue
            try:
                result = asr_ref.transcribe(segment)
            except Exception as e:
                log.error("ASR error (%s): %s", channel_name, e)
                continue
            asr_ms = (time.perf_counter() - asr_start) * 1000
```

- [ ] **Step 2: Keep `_asr_lock` only for engine switch in `_switch_asr_engine`**

The lock in `_switch_asr_engine` (line 373) is already correct — it protects the assignment. No change needed there. But verify `_asr_lock` is not used elsewhere unnecessarily:

Run: `grep -n '_asr_lock' main.py`

Remove any other `_asr_lock` usage around transcribe calls. The lock should only appear at the engine switch assignment point.

- [ ] **Step 3: Verify and commit**

Run the app, confirm both ASR channels can process simultaneously (check logs for interleaved `ASR [对方]` and `ASR [我方]` messages without long gaps).

```bash
git add main.py
git commit -m "perf: remove _asr_lock from transcribe path, allow parallel ASR inference"
```

---

### Task 2: Move speaker identification off the ASR worker thread

`speaker_id.identify()` does neural network inference (1-10s) **inside** the ASR worker thread, blocking the next segment from being processed. Move it to fire-and-forget: emit the message to UI immediately with a default speaker, then update asynchronously.

**Files:**
- Modify: `main.py:585-604` (speaker ID + UI emit in `_asr_worker`)
- Modify: `main.py:198-199` (add speaker ID queue)

- [ ] **Step 1: Add a speaker identification worker thread and queue**

In `main.py` `__init__`, add:

```python
        self._speaker_queue = queue.Queue(maxsize=50)
```

In `start()`, after the ASR threads are started, add:

```python
        self._speaker_thread = threading.Thread(target=self._speaker_worker, daemon=True)
        self._speaker_thread.start()
```

In `stop()`, add `self._speaker_thread` to the join list.

- [ ] **Step 2: Create the `_speaker_worker` method**

```python
    def _speaker_worker(self):
        """Thread: consume speaker ID queue -> identify -> update UI label."""
        while self._running:
            try:
                msg_id, segment = self._speaker_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self._speaker_id.ready:
                continue
            try:
                speaker = self._speaker_id.identify(segment)
                if self._overlay:
                    self._overlay.update_speaker_signal.emit(msg_id, speaker)
            except Exception as e:
                log.error("Speaker ID error: %s", e)
```

- [ ] **Step 3: Add `update_speaker_signal` to SubtitleOverlay**

In `subtitle_overlay.py`, add a new signal and handler to update a message's speaker label after identification:

```python
    update_speaker_signal = pyqtSignal(int, str)  # msg_id, speaker
```

Connect it in `__init__`:
```python
        self.update_speaker_signal.connect(self._on_update_speaker)
```

Add handler:
```python
    def _on_update_speaker(self, msg_id, speaker):
        msg = self._messages.get(msg_id)
        if msg:
            msg.update_speaker(speaker)
```

Add `update_speaker` to `ChatMessage` — update the speaker label text.

- [ ] **Step 4: Modify `_asr_worker` to use default speaker immediately, enqueue identification**

```python
            # Speaker identification: mic = fixed "我方", system = async diarization
            if is_mic_channel:
                speaker = "我方"
            else:
                speaker = default_speaker  # "对方" initially

            self._asr_count += 1
            log.info("ASR [%s] [%s] (%.0fms): %s", speaker, source_lang, asr_ms, text)
            self._dialogue_buffer.add(speaker, text)

            self._msg_counter += 1
            msg_id = self._msg_counter
            ts = time.strftime("%H:%M:%S")
            if self._overlay:
                self._overlay.add_message(msg_id, ts, text, source_lang, asr_ms, speaker)

            # Async speaker identification for system channel
            if not is_mic_channel and self._speaker_id.ready:
                try:
                    self._speaker_queue.put_nowait((msg_id, segment))
                except queue.Full:
                    pass  # skip speaker ID if backlogged
```

- [ ] **Step 5: Commit**

```bash
git add main.py subtitle_overlay.py
git commit -m "perf: move speaker identification to async worker thread"
```

---

### Task 3: Make analyzer sleep interruptible

`analyzer.py:140` has `time.sleep(remaining)` (up to 8 seconds) that blocks the entire analyzer thread. Manual triggers can't interrupt it. Replace with `Event.wait()`.

**Files:**
- Modify: `analyzer.py:119-141`

- [ ] **Step 1: Add a cancel event and replace sleep with interruptible wait**

```python
# BEFORE (analyzer.py:127-136):
            # Enforce minimum display time (unless manual trigger)
            if not self._manual_trigger and self._last_analysis_done_time > 0:
                elapsed = time.time() - self._last_analysis_done_time
                remaining = self.MIN_DISPLAY_S - elapsed
                if remaining > 0:
                    # Wait for display time, then re-check
                    time.sleep(remaining)
                    if not self._running:
                        break

# AFTER:
            # Enforce minimum display time (unless manual trigger)
            if not self._manual_trigger and self._last_analysis_done_time > 0:
                elapsed = time.time() - self._last_analysis_done_time
                remaining = self.MIN_DISPLAY_S - elapsed
                if remaining > 0:
                    # Interruptible wait — wakes on manual trigger or stop
                    self._request_event.wait(timeout=remaining)
                    self._request_event.clear()
                    if not self._running:
                        break
```

This works because `trigger_manual()` already calls `self._request_event.set()`, which will wake the wait early.

- [ ] **Step 2: Commit**

```bash
git add analyzer.py
git commit -m "perf: replace blocking sleep with interruptible Event.wait in analyzer"
```

---

### Task 4: Add timeout to summary compressor API call

`summary_compressor.py:88` does a synchronous `chat.completions.create()` with no timeout. If the API stalls, the thread hangs forever.

**Files:**
- Modify: `summary_compressor.py:87-96`

- [ ] **Step 1: Add timeout to the API call**

```python
# BEFORE (summary_compressor.py:88-96):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_COMPRESS_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=600,
                temperature=0.3,
            )

# AFTER:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_COMPRESS_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=600,
                temperature=0.3,
                timeout=30,
            )
```

- [ ] **Step 2: Commit**

```bash
git add summary_compressor.py
git commit -m "perf: add 30s timeout to summary compressor API call"
```

---

### Task 5: Cap dialogue buffer size to prevent unbounded memory growth

`dialogue_buffer.py` `_utterances` list grows indefinitely. Over hours of streaming, this becomes an unbounded memory leak.

**Files:**
- Modify: `dialogue_buffer.py:16-39`

- [ ] **Step 1: Add max size and trimming to `add()`**

```python
# BEFORE (dialogue_buffer.py:16-19):
class DialogueBuffer:
    """Collects utterances from dual ASR channels, manages pending analysis queue and rolling summary."""

    def __init__(self):

# AFTER:
class DialogueBuffer:
    """Collects utterances from dual ASR channels, manages pending analysis queue and rolling summary."""

    MAX_UTTERANCES = 2000  # keep last N utterances; older ones covered by summary

    def __init__(self):
```

In `add()`, after appending, trim if over limit:

```python
    def add(self, speaker: str, text: str) -> Utterance:
        """Add a new utterance. Thread-safe. Notifies listeners."""
        u = Utterance(speaker=speaker, text=text, timestamp=time.time())
        with self._lock:
            self._utterances.append(u)
            self._pending.append(u)
            # Trim old utterances (already covered by summary)
            if len(self._utterances) > self.MAX_UTTERANCES:
                trim = len(self._utterances) - self.MAX_UTTERANCES
                # Only trim what's already summarized
                safe_trim = min(trim, self._summary_cursor)
                if safe_trim > 0:
                    self._utterances = self._utterances[safe_trim:]
                    self._summary_cursor -= safe_trim
        for fn in self._listeners:
            try:
                fn(u)
            except Exception:
                pass
        return u
```

- [ ] **Step 2: Commit**

```bash
git add dialogue_buffer.py
git commit -m "perf: cap dialogue buffer at 2000 utterances to prevent memory growth"
```

---

### Task 6: Increase ASR queue size and improve backpressure

`maxsize=10` is too small. When ASR is slow, queue fills up and segments get merged, creating even longer segments that make ASR even slower (vicious cycle).

**Files:**
- Modify: `main.py:198-199` (queue size)
- Modify: `main.py:526-537` (`_enqueue_asr` backpressure)

- [ ] **Step 1: Increase queue size and drop oldest instead of merging**

```python
# Queue size (main.py:198-199):
        self._asr_queue_system = queue.Queue(maxsize=30)
        self._asr_queue_mic = queue.Queue(maxsize=30)
```

Fix `_enqueue_asr` to drop oldest instead of merging (merging creates mega-segments):

```python
    def _enqueue_asr(self, q, segment, speaker):
        """Push segment to ASR queue with backpressure handling."""
        try:
            q.put_nowait((segment, speaker))
        except queue.Full:
            # Drop oldest segment to make room (avoid merging which creates slow mega-segments)
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((segment, speaker))
            except queue.Full:
                pass
            log.warning("ASR queue full for %s, dropped oldest segment", speaker)
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "perf: increase ASR queue to 30, drop oldest instead of merging on overflow"
```

---

### Task 7: Add timeout guard to ASR transcribe calls

ASR inference has no timeout. If the model hangs, the worker thread blocks forever.

**Files:**
- Modify: `main.py:555-560` (transcribe call in `_asr_worker`)

- [ ] **Step 1: Wrap transcribe with a timeout log warning**

We can't easily kill a native inference call, but we can log a warning when inference takes too long, helping diagnose stuck states. Add a duration check:

```python
            asr_start = time.perf_counter()
            asr_ref = self._asr
            if asr_ref is None:
                continue
            try:
                result = asr_ref.transcribe(segment)
            except Exception as e:
                log.error("ASR error (%s): %s", channel_name, e)
                continue
            asr_ms = (time.perf_counter() - asr_start) * 1000
            if asr_ms > 10000:
                log.warning("ASR slow (%s): %.0fms for %.1fs segment", channel_name, asr_ms, seg_len)
```

- [ ] **Step 2: Add max segment length cap before enqueueing**

Prevent excessively long segments from being sent to ASR (they cause multi-second inference). In `_enqueue_asr`, cap segment to 30 seconds:

```python
    def _enqueue_asr(self, q, segment, speaker):
        """Push segment to ASR queue with backpressure handling."""
        # Cap segment length to avoid very slow ASR inference
        max_samples = 30 * 16000  # 30 seconds
        if len(segment) > max_samples:
            segment = segment[-max_samples:]
            log.warning("Segment too long for %s, trimmed to 30s", speaker)
        try:
            q.put_nowait((segment, speaker))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((segment, speaker))
            except queue.Full:
                pass
            log.warning("ASR queue full for %s, dropped oldest segment", speaker)
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "perf: add ASR slow-inference warning and 30s segment length cap"
```
