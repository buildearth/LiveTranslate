# 多窗口并行 AI 分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support multiple independent AI analysis windows running different scene presets in parallel, each with scrolling history.

**Architecture:** Each popup window gets its own `AnalysisScheduler` instance. The `AnalysisScheduler` is refactored to maintain its own per-instance pending queue (instead of draining the shared `DialogueBuffer._pending`), so multiple analyzers can coexist without competing for utterances. A new `AnalysisWindow` widget provides the standalone UI. Right-clicking the scene combo in the overlay opens a preset in a new window.

**Tech Stack:** PyQt6 (QWidget, signals), Python threading, OpenAI API

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `analyzer.py` | Analysis scheduler | Modify: per-instance pending queue instead of shared buffer drain |
| `analysis_window.py` | Standalone analysis popup window | Create |
| `subtitle_overlay.py` | Main overlay with scene combo | Modify: add right-click context menu on scene combo |
| `main.py` | App controller | Modify: manage multiple analysis windows |
| `i18n/zh.yaml` | Chinese translations | Modify: add new keys |
| `i18n/en.yaml` | English translations | Modify: add new keys |

---

### Task 1: Refactor AnalysisScheduler to use per-instance pending queue

Currently `AnalysisScheduler._loop()` calls `self._buffer.take_pending()` which drains a shared list. When multiple analyzers exist, the first to drain gets all utterances, others get nothing. Each analyzer must maintain its own pending queue.

**Files:**
- Modify: `analyzer.py`

- [ ] **Step 1: Add per-instance pending list and modify `_on_new_utterance` to collect utterances locally**

In `analyzer.py`, in `__init__`, add:

```python
        self._own_pending: list[Utterance] = []
        self._pending_lock = threading.Lock()
```

Modify `_on_new_utterance` to append to the local list:

```python
    def _on_new_utterance(self, u: Utterance):
        with self._pending_lock:
            self._own_pending.append(u)
            self._pending_count = len(self._own_pending)
        if self._api_busy:
            return
        if self._pending_count >= self.BATCH_THRESHOLD:
            self._cancel_debounce()
            self._request_event.set()
        else:
            self._restart_debounce()
```

- [ ] **Step 2: Add `_take_own_pending` method and update `_loop` to use it**

Add method:

```python
    def _take_own_pending(self) -> list[Utterance]:
        with self._pending_lock:
            batch = self._own_pending[:]
            self._own_pending.clear()
            return batch
```

In `_loop`, replace:

```python
            pending = self._buffer.take_pending()
            if not pending:
                continue
            self._pending_count = 0
```

With:

```python
            pending = self._take_own_pending()
            if not pending:
                continue
```

- [ ] **Step 3: Lint and commit**

```bash
python -m ruff check --select F,E,W --ignore E501,E402 analyzer.py
git add analyzer.py
git commit -m "refactor: AnalysisScheduler uses per-instance pending queue for multi-analyzer support"
```

---

### Task 2: Create AnalysisWindow standalone popup

**Files:**
- Create: `analysis_window.py`

- [ ] **Step 1: Create the AnalysisWindow class**

Create file `analysis_window.py`:

```python
"""Standalone analysis window for parallel AI analysis."""

import time

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
)
from PyQt6.QtGui import QFont

from i18n import t


class AnalysisWindow(QWidget):
    """Independent floating window displaying AI analysis results for a specific preset.

    Results are appended chronologically with timestamps; streaming updates
    the latest entry in-place until finalized.
    """

    closed = pyqtSignal(object)  # emits self when window is closed
    manual_trigger = pyqtSignal()  # user clicked analyze button
    update_stream_signal = pyqtSignal(str)   # thread-safe streaming update
    finish_stream_signal = pyqtSignal(str)   # thread-safe final result

    def __init__(self, preset_name: str, parent=None):
        super().__init__(parent)
        self._preset_name = preset_name
        self._streaming_active = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(320, 200)
        self.resize(400, 300)

        self._drag_pos = None
        self._setup_ui()
        self.update_stream_signal.connect(self._on_update_stream)
        self.finish_stream_signal.connect(self._on_finish_stream)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        container = QWidget()
        container.setStyleSheet(
            "background: rgba(20, 20, 20, 220); border-radius: 6px;"
        )
        clayout = QVBoxLayout(container)
        clayout.setContentsMargins(6, 4, 6, 6)
        clayout.setSpacing(4)

        # Title bar
        title_bar = QHBoxLayout()
        self._title_label = QLabel(self._preset_name)
        self._title_label.setStyleSheet(
            "color: #ddd; font-weight: bold; font-size: 12px; padding: 2px;"
        )
        title_bar.addWidget(self._title_label)
        title_bar.addStretch()

        analyze_btn = QPushButton(t("analysis_trigger"))
        analyze_btn.setFixedHeight(20)
        analyze_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 1px 6px; color: #aaa; "
            "background: rgba(255,255,255,20); border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { background: rgba(255,255,255,40); }"
        )
        analyze_btn.clicked.connect(self.manual_trigger.emit)
        title_bar.addWidget(analyze_btn)

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { font-size: 12px; color: #aaa; background: transparent; border: none; }"
            "QPushButton:hover { color: #f55; }"
        )
        close_btn.clicked.connect(self.close)
        title_bar.addWidget(close_btn)

        clayout.addLayout(title_bar)

        # Analysis text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "QTextEdit { background: rgba(0,0,0,120); color: white; "
            "border: 1px solid #333; font-size: 12px; padding: 4px; }"
        )
        clayout.addWidget(self._text)

        layout.addWidget(container)

    # -- Drag support (middle-click) --

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.MiddleButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # -- Analysis output --

    def on_stream_chunk(self, text: str):
        """Called from analyzer thread with partial streaming text."""
        if not self._streaming_active:
            self._streaming_active = True
            ts = time.strftime("%H:%M:%S")
            self._text.append(
                f'<div style="color:#666; margin:4px 0 2px 0;">\u2500\u2500\u2500\u2500 {ts} \u2500\u2500\u2500\u2500</div>'
            )
        # Replace last block with updated streaming content
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        # Find and remove previous streaming content after last separator
        doc = self._text.document()
        block = doc.lastBlock()
        # Remove all blocks after the last separator line
        while block.isValid() and "\u2500\u2500\u2500\u2500" not in block.text():
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.movePosition(cursor.MoveOperation.PreviousBlock, cursor.MoveMode.KeepAnchor)
            block = block.previous()
        # Move past the separator line
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)

        # Simple approach: clear everything after separator and re-append
        self._text.setUpdatesEnabled(False)
        self._remove_content_after_last_separator()
        self._text.append(f'<div style="color:#eee;">{text}</div>')
        self._text.setUpdatesEnabled(True)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_stream_done(self, text: str, _prev_text: str = ""):
        """Called from analyzer thread when analysis completes."""
        self._streaming_active = False
        self._text.setUpdatesEnabled(False)
        self._remove_content_after_last_separator()
        self._text.append(f'<div style="color:#eee;">{text}</div>')
        self._text.setUpdatesEnabled(True)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _remove_content_after_last_separator(self):
        """Remove text content after the last timestamp separator line."""
        html = self._text.toHtml()
        # Find last separator
        sep = "\u2500\u2500\u2500\u2500"
        idx = html.rfind(sep)
        if idx < 0:
            return
        # Find the end of the separator's closing tag
        close_div = html.find("</div>", idx)
        if close_div >= 0:
            # Keep everything up to and including the separator div, remove the rest
            # But we need to preserve the HTML structure, so just clear and re-set
            pass
        # Simpler approach: rebuild from plain text segments
        # Not ideal but reliable - just clear the last content block
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        doc = self._text.document()
        block = doc.lastBlock()
        while block.isValid() and sep not in block.text():
            block = block.previous()
        if block.isValid():
            # Select from after separator block to end
            next_block = block.next()
            if next_block.isValid():
                cursor.setPosition(next_block.position())
                cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()

    def _on_update_stream(self, text: str):
        self.on_stream_chunk(text)

    def _on_finish_stream(self, text: str):
        self.on_stream_done(text)

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)
```

- [ ] **Step 2: Lint and commit**

```bash
python -m ruff check --select F,E,W --ignore E501,E402 analysis_window.py
git add analysis_window.py
git commit -m "feat: add AnalysisWindow standalone popup for parallel analysis"
```

---

### Task 3: Add right-click context menu on scene combo

**Files:**
- Modify: `subtitle_overlay.py` (DragHandle class, around line 788-796)
- Modify: `i18n/zh.yaml`
- Modify: `i18n/en.yaml`

- [ ] **Step 1: Add signal and context menu to DragHandle's scene combo**

In `subtitle_overlay.py`, find the `DragHandle` class. In the signals section (around line 860), add:

```python
    open_analysis_window = pyqtSignal(str)  # preset_name
```

After `self._scene_combo.currentIndexChanged.connect(self._on_scene_changed)` (around line 795), add:

```python
        self._scene_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._scene_combo.customContextMenuRequested.connect(self._on_scene_context_menu)
```

Add the context menu handler method in DragHandle:

```python
    def _on_scene_context_menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        name = self._scene_combo.currentData()
        if not name:
            return
        menu = QMenu(self)
        action = menu.addAction(t("open_in_new_window").format(name=name))
        result = menu.exec(self._scene_combo.mapToGlobal(pos))
        if result == action:
            self.open_analysis_window.emit(name)
```

- [ ] **Step 2: Forward signal from SubtitleOverlay**

In `SubtitleOverlay` class signals section (around line 900), add:

```python
    open_analysis_window_requested = pyqtSignal(str)  # preset_name
```

In `SubtitleOverlay.__init__`, find where `self._handle` signals are connected. Add:

```python
        self._handle.open_analysis_window.connect(self.open_analysis_window_requested.emit)
```

- [ ] **Step 3: Add i18n keys**

In `i18n/zh.yaml`, in the Analysis section, add:

```yaml
open_in_new_window: "在新窗口中打开「{name}」"
```

In `i18n/en.yaml`, in the Analysis section, add:

```yaml
open_in_new_window: "Open \"{name}\" in new window"
```

- [ ] **Step 4: Lint and commit**

```bash
python -m ruff check --select F,E,W --ignore E501,E402 subtitle_overlay.py
git add subtitle_overlay.py i18n/zh.yaml i18n/en.yaml
git commit -m "feat: add right-click context menu to open analysis preset in new window"
```

---

### Task 4: Wire up multi-window management in main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add analysis window tracking list in `LiveTranslateApp.__init__`**

Find `self._analyzer = AnalysisScheduler(self._dialogue_buffer)` (around line 189). After it, add:

```python
        self._analysis_windows = []  # list of (AnalysisScheduler, AnalysisWindow)
```

- [ ] **Step 2: Add `_open_analysis_window` method**

After `_on_scene_changed` method, add:

```python
    def _open_analysis_window(self, preset_name):
        """Open a new independent analysis window for the given preset."""
        if not self._panel:
            return
        preset = self._panel.get_preset(preset_name)
        if not preset:
            log.warning("Preset not found: %s", preset_name)
            return

        # Create independent analyzer
        analyzer = AnalysisScheduler(self._dialogue_buffer)
        analyzer.set_preset(preset)

        # Share the same API client/model
        active_model = self._panel.get_active_model()
        if active_model:
            client = make_openai_client(
                active_model["api_base"], active_model["api_key"],
                active_model.get("proxy", "none"),
                timeout=active_model.get("timeout", 10),
            )
            analyzer.set_client(client, active_model["model"])

        # Create window
        from analysis_window import AnalysisWindow
        win = AnalysisWindow(preset_name)

        # Connect analyzer output to window (thread-safe via Qt signals)
        analyzer.on_stream_chunk = lambda text: win.update_stream_signal.emit(text)
        analyzer.on_stream_done = lambda text, prev: win.finish_stream_signal.emit(text)

        # Connect window controls
        win.manual_trigger.connect(analyzer.trigger_manual)
        win.closed.connect(lambda w: self._close_analysis_window(w))

        self._analysis_windows.append((analyzer, win))
        analyzer.start()
        win.show()
        log.info("Opened analysis window: %s (total: %d)", preset_name, len(self._analysis_windows))

    def _close_analysis_window(self, win):
        """Clean up when an analysis window is closed."""
        for i, (analyzer, w) in enumerate(self._analysis_windows):
            if w is win:
                analyzer.stop()
                self._analysis_windows.pop(i)
                log.info("Closed analysis window (remaining: %d)", len(self._analysis_windows))
                break
```

- [ ] **Step 3: Update `_update_analyzer_client` to sync all windows**

Find `_update_analyzer_client` method. After the existing `self._analyzer.set_client(...)` and `self._compressor.set_client(...)` lines, add:

```python
        for analyzer, _win in self._analysis_windows:
            analyzer.set_client(client, model_name)
```

- [ ] **Step 4: Update `stop()` to clean up all analysis windows**

In the `stop()` method, after `self._analyzer.stop()`, add:

```python
        for analyzer, win in self._analysis_windows:
            analyzer.stop()
            win.close()
        self._analysis_windows.clear()
```

- [ ] **Step 5: Connect the overlay signal**

Find the signal connections section at the bottom of `main()` (around line 1280). After `overlay.retain_history_changed.connect(...)`, add:

```python
    overlay.open_analysis_window_requested.connect(live_trans._open_analysis_window)
```

- [ ] **Step 6: Lint and commit**

```bash
python -m ruff check --select F,E,W --ignore E501,E402 main.py
git add main.py
git commit -m "feat: wire up multi-window analysis management in LiveTranslateApp"
```

