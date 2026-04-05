"""Standalone analysis window for parallel AI analysis."""

import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QSizeGrip,
)

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
        self._stream_start_pos = -1  # cursor position where streaming content starts

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

        # Resize grip
        grip_row = QHBoxLayout()
        grip_row.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip.setStyleSheet("background: transparent;")
        grip_row.addWidget(grip)
        clayout.addLayout(grip_row)

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

    # -- Thread-safe slots --

    def _on_update_stream(self, text: str):
        """Qt slot: update streaming content on main thread."""
        if not self._streaming_active:
            self._streaming_active = True
            ts = time.strftime("%H:%M:%S")
            sep_html = (
                '<div style="color:#666; margin:6px 0 2px 0;">'
                f'\u2500\u2500\u2500\u2500 {ts} \u2500\u2500\u2500\u2500</div>'
            )
            self._text.append(sep_html)
            # Record cursor position after separator for later replacement
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._stream_start_pos = cursor.position()

        # Replace streaming content: select from start_pos to end, then replace
        cursor = self._text.textCursor()
        cursor.setPosition(self._stream_start_pos)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        cursor.insertHtml(f'<div style="color:#eee;">{text}</div>')
        # Scroll to bottom
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finish_stream(self, text: str):
        """Qt slot: finalize analysis result on main thread."""
        if self._streaming_active:
            # Replace streaming content with final version
            cursor = self._text.textCursor()
            cursor.setPosition(self._stream_start_pos)
            cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
            cursor.insertHtml(f'<div style="color:#eee;">{text}</div>')
        else:
            # No streaming happened, just append with separator
            ts = time.strftime("%H:%M:%S")
            self._text.append(
                f'<div style="color:#666; margin:6px 0 2px 0;">'
                f'\u2500\u2500\u2500\u2500 {ts} \u2500\u2500\u2500\u2500</div>'
            )
            self._text.append(f'<div style="color:#eee;">{text}</div>')
        self._streaming_active = False
        self._stream_start_pos = -1
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)
