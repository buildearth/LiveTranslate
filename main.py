"""
LiveTranslate - Phase 0 Prototype
Real-time audio translation using WASAPI loopback + faster-whisper + LLM.
"""

import sys
import signal
import logging
import threading
import queue
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from model_manager import (
    apply_cache_env,
    get_missing_models,
    is_asr_cached,
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
    get_qwen3_asr_model_dir,
)

# Set cache env BEFORE importing torch so TORCH_HOME is respected
apply_cache_env()

# Qwen3-ASR uses onnxruntime-directml (libomp140.dll) which conflicts with
# PyTorch's libiomp5md.dll. Allow coexistence since they don't run concurrently.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch must be imported before PyQt6 to avoid DLL conflicts on Windows
import torch  # noqa: F401

from audio_capture import AudioCapture
from vad_processor import VADProcessor
from asr_engine import ASREngine
from speaker_identifier import SpeakerIdentifier
from translator import Translator, make_openai_client
from dialogue_buffer import DialogueBuffer
from analyzer import AnalysisScheduler
from summary_compressor import SummaryCompressor

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QMessageBox
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import QTimer, Qt

from subtitle_overlay import SubtitleOverlay
from subtitle_window import SubtitleWindow
from log_window import LogWindow
from control_panel import (
    ControlPanel,
    SETTINGS_FILE,
    _load_saved_settings,
    _save_settings,
)
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from i18n import t, set_lang, LANGUAGES, COMMON_LANG_CODES


def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"livetrans_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    for noisy in (
        "httpcore",
        "httpx",
        "openai",
        "filelock",
        "huggingface_hub",
        "funasr",
        "modelscope",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Log file: {log_file}")

    # FunASR/ModelScope spam the root logger; suppress after our own init log
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

    _logger = logging.getLogger("LiveTranslate")

    def _excepthook(exc_type, exc_value, exc_tb):
        _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        _logger.critical(
            f"Uncaught exception in thread {args.thread}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    return _logger


log = logging.getLogger("LiveTranslate")


def create_app_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return QIcon(pix)


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiveTranslateApp:
    def __init__(self, config):
        self._config = config
        self._running = False
        self._paused = False
        self._asr_ready = False  # True when ASR model is loaded

        self._audio = AudioCapture(
            device=config["audio"].get("device"),
            sample_rate=config["audio"]["sample_rate"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        self._asr_type = None
        self._asr = None
        self._asr_device = config["asr"]["device"]
        self._whisper_model_size = config["asr"]["model_size"]
        self._asr_lock = threading.Lock()
        self._target_language = config["translation"]["target_language"]

        # Keep a Translator instance for backward compat (subtitle window extra langs, etc.)
        self._translator = Translator(
            api_base=config["translation"]["api_base"],
            api_key=config["translation"]["api_key"],
            model=config["translation"]["model"],
            target_language=self._target_language,
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
            streaming=config["translation"]["streaming"],
            system_prompt=config["translation"].get("system_prompt"),
        )
        self._overlay = None
        self._subwin = None
        self._panel = None

        self._asr_count = 0
        self._translate_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._input_price = 0.0
        self._output_price = 0.0
        self._msg_counter = 0

        # Dual-channel async architecture
        self._dialogue_buffer = DialogueBuffer()
        self._analyzer = AnalysisScheduler(self._dialogue_buffer)
        self._compressor = SummaryCompressor(self._dialogue_buffer)

        # Speaker identification for system audio
        self._speaker_id = SpeakerIdentifier(device="cuda" if torch.cuda.is_available() else "cpu")

        # Dual VAD instances (created in start())
        self._vad_system = None
        self._vad_mic = None
        self._asr_queue_system = queue.Queue(maxsize=10)
        self._asr_queue_mic = queue.Queue(maxsize=10)

        # Thread references
        self._audio_thread_system = None
        self._audio_thread_mic = None
        self._asr_thread_system = None
        self._asr_thread_mic = None

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay

    def set_subtitle_window(self, subwin: SubtitleWindow):
        self._subwin = subwin

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
        panel.settings_changed.connect(self._on_settings_changed)
        panel.model_changed.connect(self._on_model_changed)
        panel.models_list_changed.connect(self._on_models_list_changed)

    def _on_models_list_changed(self, models: list, active_idx: int):
        if self._overlay:
            self._overlay.set_models(models, active_idx)

    def _on_settings_changed(self, settings):
        # Propagate VAD settings to both channels
        if self._vad_system:
            self._vad_system.update_settings(settings)
        if self._vad_mic:
            self._vad_mic.update_settings(settings)
        if "style" in settings and self._overlay:
            self._overlay.apply_style(settings["style"])
        if "asr_language" in settings and self._asr:
            self._asr.set_language(settings["asr_language"])
        # ASR compute device change: try in-place migration first
        new_device = settings.get("asr_device")
        if new_device and new_device != self._asr_device:
            old_device = self._asr_device
            self._asr_device = new_device
            if self._asr is not None and hasattr(self._asr, "to_device"):
                result = self._asr.to_device(new_device)
                if result is not False:
                    log.info(f"ASR device migrated: {old_device} -> {new_device}")
                    if self._overlay:
                        display_name = ASR_DISPLAY_NAMES.get(
                            self._asr_type, self._asr_type
                        )
                        self._overlay.update_asr_device(
                            f"{display_name} [{new_device}]"
                        )
                    import gc

                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    self._asr_type = None  # ctranslate2: force reload
            else:
                self._asr_type = None  # no engine loaded: force reload
        new_whisper_size = settings.get("whisper_model_size")
        if new_whisper_size and new_whisper_size != self._whisper_model_size:
            self._whisper_model_size = new_whisper_size
            if self._asr_type == "whisper":
                self._asr_type = None
        if "asr_engine" in settings:
            self._switch_asr_engine(settings["asr_engine"])
        if "audio_device" in settings:
            old_device = self._audio._device_name
            self._audio.set_device(settings["audio_device"])
            if old_device != settings.get("audio_device"):
                if self._vad_system:
                    self._vad_system.flush()
                    self._vad_system._reset()
                if self._overlay:
                    self._overlay.update_monitor(0.0, 0.0)
        if "mic_device" in settings:
            self._audio.set_mic_device(settings["mic_device"])
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)
        if "timeout" in settings and self._translator:
            self._translator.set_timeout(settings["timeout"])

    def _on_target_language_changed(self, lang: str):
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._translator:
            self._translator.set_target_language(lang)
        if self._panel:
            settings = self._panel.get_settings()
            settings["target_language"] = lang
            from control_panel import _save_settings

            _save_settings(settings)

    def _on_model_changed(self, model_config: dict):
        log.info(
            f"Switching translator: {model_config['name']} ({model_config['model']})"
        )
        prompt = None
        if self._panel:
            prompt = self._panel.get_settings().get("system_prompt")
        if not prompt:
            prompt = self._config["translation"].get("system_prompt")
        timeout = 10
        if self._panel:
            timeout = self._panel.get_settings().get("timeout", 10)
        self._translator = Translator(
            api_base=model_config["api_base"],
            api_key=model_config["api_key"],
            model=model_config["model"],
            target_language=self._target_language,
            max_tokens=self._config["translation"]["max_tokens"],
            temperature=self._config["translation"]["temperature"],
            streaming=model_config.get("streaming", True),
            system_prompt=prompt,
            proxy=model_config.get("proxy", "none"),
            no_system_role=model_config.get("no_system_role", False),
            no_think=model_config.get("no_think", True),
            json_response=model_config.get("json_response", False),
            timeout=timeout,
        )
        self._translator.set_context_turns(model_config.get("context_turns", 0))
        self._input_price = model_config.get("input_price", 0)
        self._output_price = model_config.get("output_price", 0)
        # Update analyzer + compressor clients
        self._update_analyzer_client(model_config)

    def _switch_asr_engine(self, engine_type: str):
        if engine_type == self._asr_type:
            return
        log.info(f"Switching ASR engine: {self._asr_type} -> {engine_type}")
        self._asr_ready = False
        # Flush and reset both VADs to stop accumulating audio during engine switch
        if self._vad_system:
            self._vad_system.flush()
            self._vad_system._reset()
        if self._vad_mic:
            self._vad_mic.flush()
            self._vad_mic._reset()
        device = self._asr_device
        hub = "ms"
        if self._panel:
            hub = self._panel.get_settings().get("hub", "ms")

        model_size = self._config["asr"]["model_size"]
        if self._panel:
            model_size = self._panel.get_settings().get(
                "whisper_model_size", model_size
            )
        cached = is_asr_cached(engine_type, model_size, hub)
        display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
        if engine_type == "whisper":
            display_name = f"Whisper {model_size}"

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        if not cached:
            missing = get_missing_models(engine_type, model_size, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing:
                dlg = ModelDownloadDialog(missing, hub=hub, parent=parent)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {engine_type}")
                    # Restore readiness if old engine is still available
                    if self._asr is not None:
                        self._asr_ready = True
                    return

        with self._asr_lock:
            old_engine = self._asr
            self._asr = None

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=display_name), parent=parent
        )

        new_asr = [None]
        load_error = [None]

        def _load():
            nonlocal old_engine
            try:
                if old_engine is not None:
                    log.info(
                        f"Releasing old ASR engine: {old_engine.__class__.__name__}"
                    )
                    if hasattr(old_engine, "unload"):
                        old_engine.unload()
                    old_engine = None
                dev = device
                dev_index = 0
                if dev.startswith("cuda:"):
                    part = dev.split("(")[0].strip()  # "cuda:0"
                    dev_index = int(part.split(":")[1])
                    dev = "cuda"

                if engine_type == "qwen3-asr":
                    from asr_qwen3 import Qwen3ASREngine

                    new_asr[0] = Qwen3ASREngine(
                        model_dir=get_qwen3_asr_model_dir(),
                        use_dml=(dev != "cpu"),
                    )
                elif engine_type == "sensevoice":
                    from asr_sensevoice import SenseVoiceEngine

                    new_asr[0] = SenseVoiceEngine(device=device, hub=hub)
                elif engine_type in ("funasr-nano", "funasr-mlt-nano"):
                    from asr_funasr_nano import FunASRNanoEngine

                    new_asr[0] = FunASRNanoEngine(
                        device=device, hub=hub, engine_type=engine_type
                    )
                else:
                    download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
                    compute = self._config["asr"]["compute_type"]
                    if dev == "cpu" and compute == "float16":
                        compute = "int8"
                    new_asr[0] = ASREngine(
                        model_size=model_size,
                        device=dev,
                        device_index=dev_index,
                        compute_type=compute,
                        language=self._config["asr"]["language"],
                        download_root=download_root,
                    )
            except Exception as e:
                load_error[0] = str(e)
                log.error(f"Failed to load ASR engine: {e}", exc_info=True)

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

        poll_timer = QTimer()

        def _check():
            if not thread.is_alive():
                poll_timer.stop()
                dlg.accept()

        poll_timer.setInterval(100)
        poll_timer.timeout.connect(_check)
        poll_timer.start()

        dlg.exec()
        poll_timer.stop()

        if load_error[0]:
            QMessageBox.warning(
                parent,
                t("error_title"),
                t("error_load_asr").format(error=load_error[0]),
            )
            # Old engine was already released; mark ASR as unavailable
            self._asr_type = None
            return

        self._asr = new_asr[0]
        self._asr_type = engine_type
        if self._panel:
            asr_lang = self._panel.get_settings().get("asr_language", "auto")
            self._asr.set_language(asr_lang)
        self._asr_ready = True
        if self._overlay:
            self._overlay.update_asr_device(f"{display_name} [{device}]")
        log.info(f"ASR engine ready: {engine_type} on {device}")

    def _compute_cost(self):
        if self._input_price > 0 or self._output_price > 0:
            return (self._total_prompt_tokens * self._input_price +
                    self._total_completion_tokens * self._output_price) / 1_000_000
        return 0.0

    # ── Dual-channel pipeline threads ──

    def _audio_pipeline_system(self):
        """Thread: read system audio -> VAD -> push segments to ASR queue."""
        silence_chunk = np.zeros(int(0.032 * 16000), dtype=np.float32)
        while self._running:
            item = self._audio.get_audio(timeout=1.0)
            if item is None:
                if self._vad_system and self._vad_system._is_speaking and not self._paused:
                    n = int(self._vad_system._silence_limit / 0.032) + 2
                    for _ in range(n):
                        seg = self._vad_system.process_chunk(silence_chunk)
                        if seg is not None:
                            self._enqueue_asr(self._asr_queue_system, seg, "\u5bf9\u65b9")
                            break
                continue
            chunk, mic_rms = item
            if self._paused:
                continue
            rms = float(np.sqrt(np.mean(chunk**2)))
            if self._overlay:
                self._overlay.update_monitor(rms, self._vad_system.last_confidence if self._vad_system else 0, mic_rms)
            if self._vad_system:
                seg = self._vad_system.process_chunk(chunk)
                if seg is not None:
                    self._enqueue_asr(self._asr_queue_system, seg, "\u5bf9\u65b9")

    def _audio_pipeline_mic(self):
        """Thread: read mic audio -> VAD -> push segments to ASR queue."""
        silence_chunk = np.zeros(int(0.032 * 16000), dtype=np.float32)
        while self._running:
            item = self._audio.get_mic_audio(timeout=1.0)
            if item is None:
                if self._vad_mic and self._vad_mic._is_speaking and not self._paused:
                    n = int(self._vad_mic._silence_limit / 0.032) + 2
                    for _ in range(n):
                        seg = self._vad_mic.process_chunk(silence_chunk)
                        if seg is not None:
                            self._enqueue_asr(self._asr_queue_mic, seg, "\u6211\u65b9")
                            break
                continue
            if self._paused:
                continue
            if self._vad_mic:
                seg = self._vad_mic.process_chunk(item)
                if seg is not None:
                    self._enqueue_asr(self._asr_queue_mic, seg, "\u6211\u65b9")

    def _enqueue_asr(self, q, segment, speaker):
        """Push segment to ASR queue with backpressure handling."""
        try:
            q.put_nowait((segment, speaker))
        except queue.Full:
            try:
                old_seg, _old_speaker = q.get_nowait()
                merged = np.concatenate([old_seg, segment])
                q.put_nowait((merged, speaker))
                log.warning("ASR queue full for %s, merged segments", speaker)
            except queue.Empty:
                q.put_nowait((segment, speaker))

    def _asr_worker(self, asr_queue, is_mic_channel: bool):
        """Thread: consume ASR queue -> transcribe -> identify speaker -> push to buffer + UI."""
        channel_name = "mic" if is_mic_channel else "system"
        while self._running:
            try:
                item = asr_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            segment, default_speaker = item
            seg_len = len(segment) / 16000

            if not self._asr_ready or self._asr is None:
                log.debug("ASR not ready, dropping %s segment", channel_name)
                continue

            asr_start = time.perf_counter()
            with self._asr_lock:
                try:
                    result = self._asr.transcribe(segment)
                except Exception as e:
                    log.error("ASR error (%s): %s", channel_name, e)
                    continue
            asr_ms = (time.perf_counter() - asr_start) * 1000

            if result is None:
                continue

            text = (result.get("text") or "").strip()
            if not text or not any(c.isalnum() for c in text):
                continue

            # Text density filter
            alnum_count = sum(1 for c in text if c.isalnum())
            if seg_len >= 2.0 and alnum_count <= 3:
                log.debug("Density filter: discarded '%s' (%.1fs)", text, seg_len)
                continue

            source_lang = result.get("language", "")

            # Language filter
            asr_lang_setting = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
            if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
                log.info("Language filter (%s): expected '%s' but got '%s', discarding: %s",
                         channel_name, asr_lang_setting, source_lang, text[:60])
                continue

            # Speaker identification: mic channel = fixed "我方", system channel = diarization
            if is_mic_channel:
                speaker = "我方"
            elif self._speaker_id.ready:
                speaker = self._speaker_id.identify(segment)
            else:
                speaker = default_speaker

            self._asr_count += 1
            log.info("ASR [%s] [%s] (%.0fms): %s", speaker, source_lang, asr_ms, text)

            # Add to dialogue buffer (notifies analyzer automatically)
            self._dialogue_buffer.add(speaker, text)

            # Update UI — show in scrolling area
            self._msg_counter += 1
            msg_id = self._msg_counter
            ts = time.strftime("%H:%M:%S")
            if self._overlay:
                self._overlay.add_message(msg_id, ts, text, source_lang, asr_ms, speaker)

    # ── Analyzer / Compressor setup ──

    def _setup_analyzer(self):
        """Connect analyzer streaming output to overlay panel."""
        def on_stream_chunk(text):
            if self._overlay:
                self._overlay.update_analysis(text)

        def on_stream_done(text, prev_text=""):
            if self._overlay:
                self._overlay.finish_analysis(text, prev_text)

        self._analyzer.on_stream_chunk = on_stream_chunk
        self._analyzer.on_stream_done = on_stream_done

    def _update_analyzer_client(self, model_cfg):
        """Called when active model changes. Updates analyzer + compressor clients."""
        client = make_openai_client(
            model_cfg["api_base"], model_cfg["api_key"],
            model_cfg.get("proxy", "none"),
            timeout=model_cfg.get("timeout", 10),
        )
        model_name = model_cfg["model"]
        self._analyzer.set_client(client, model_name)
        self._compressor.set_client(client, model_name)

    def _on_scene_changed(self, preset_name):
        """Handle scene preset change from overlay combo."""
        if self._panel:
            preset = self._panel.get_preset(preset_name)
            if preset:
                self._analyzer.set_preset(preset)
                log.info("Scene preset changed to: %s", preset_name)

    # ── Start / Stop / Pause ──

    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._msg_counter = 0

        # Initialize dual VAD (reuse existing VAD config)
        vad_kwargs = dict(
            sample_rate=self._config["audio"]["sample_rate"],
            threshold=self._config["asr"]["vad_threshold"],
            min_speech_duration=self._config["asr"]["min_speech_duration"],
            max_speech_duration=self._config["asr"]["max_speech_duration"],
            chunk_duration=self._config["audio"]["chunk_duration"],
        )
        self._vad_system = VADProcessor(**vad_kwargs)
        self._vad_mic = VADProcessor(**vad_kwargs)

        # Load speaker identification model
        if not self._speaker_id.ready:
            self._speaker_id.load()

        self._audio.start()

        # Audio pipeline threads
        self._audio_thread_system = threading.Thread(target=self._audio_pipeline_system, daemon=True)
        self._audio_thread_mic = threading.Thread(target=self._audio_pipeline_mic, daemon=True)
        self._audio_thread_system.start()
        self._audio_thread_mic.start()

        # ASR worker threads (is_mic_channel: False=system with diarization, True=mic fixed "我方")
        self._asr_thread_system = threading.Thread(
            target=self._asr_worker, args=(self._asr_queue_system, False), daemon=True)
        self._asr_thread_mic = threading.Thread(
            target=self._asr_worker, args=(self._asr_queue_mic, True), daemon=True)
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

        for t_ref in [self._audio_thread_system, self._audio_thread_mic,
                      self._asr_thread_system, self._asr_thread_mic]:
            if t_ref:
                t_ref.join(timeout=3)

        self._analyzer.stop()
        self._compressor.stop()

        self._audio_thread_system = None
        self._audio_thread_mic = None
        self._asr_thread_system = None
        self._asr_thread_mic = None

        log.info("Pipeline stopped")

    def pause(self):
        self._paused = True
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        log.info("Pipeline resumed")


def main():
    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    saved = _load_saved_settings()

    # Log actual effective config
    _asr_eng = (saved or {}).get("asr_engine", "whisper")
    _active_idx = (saved or {}).get("active_model", 0)
    _models = (saved or {}).get("models", [])
    if 0 <= _active_idx < len(_models):
        _m = _models[_active_idx]
        _model_info = f"{_m.get('name', '?')} ({_m.get('model', '?')})"
    else:
        _model_info = f"{config['translation']['model']} (default)"
    log.info(f"Config loaded: ASR={_asr_eng}, Translator={_model_info}")

    # Apply UI language before creating any widgets
    if saved and saved.get("ui_lang"):
        set_lang(saved["ui_lang"])

    os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # First launch → setup wizard (hub + download) → configure translation API
    if not SETTINGS_FILE.exists():
        wizard = SetupWizardDialog()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        saved = _load_saved_settings()
        log.info("Setup wizard completed")

        # Prompt user to configure translation API
        from dialogs import ModelEditDialog

        info = QMessageBox(
            QMessageBox.Icon.Information,
            t("window_setup"),
            t("setup_api_hint"),
        )
        info.exec()

        dlg = ModelEditDialog(None, {
            "name": "hunyuan-mt-chimera-7b",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "sk-lm-tHzDfNGm:dgxlip7eebn3HIMxivqN",
            "model": "hunyuan-mt-chimera-7b",
        })
        dlg.setWindowTitle(t("setup_api_title"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if data.get("api_key"):
                saved["models"] = [data]
                saved["active_model"] = 0
                _save_settings(saved)
                log.info(f"Translation API configured: {data['name']}")
        # If user skips, ControlPanel will create default placeholder from config.yaml

    # Non-first launch but models missing → download dialog
    else:
        missing = get_missing_models(
            saved.get("asr_engine", "sensevoice"),
            config["asr"]["model_size"],
            saved.get("hub", "ms"),
        )
        if missing:
            log.info(f"Missing models: {[m['name'] for m in missing]}")
            dlg = ModelDownloadDialog(missing, hub=saved.get("hub", "ms"))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)

    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    panel = ControlPanel(config, saved_settings=saved)

    overlay = SubtitleOverlay(config["subtitle"])
    if saved:
        ox = saved.get("overlay_x")
        oy = saved.get("overlay_y")
        ow = saved.get("overlay_w")
        oh = saved.get("overlay_h")
        if ox is not None and oy is not None:
            if SubtitleWindow._is_pos_visible(ox, oy):
                overlay.move(ox, oy)
            else:
                screen = QApplication.primaryScreen()
                geo = screen.availableGeometry()
                overlay.move(geo.right() - overlay.width() - 20, geo.bottom() - overlay.height() - 60)
        if ow and oh:
            overlay.resize(ow, oh)
    overlay.show()

    # Subtitle window
    subwin_cfg = (saved or {}).get("subtitle_mode")
    subwin = SubtitleWindow(subwin_cfg)
    subwin_was_enabled = (subwin_cfg or {}).get("enabled", False)

    live_trans = LiveTranslateApp(config)
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)

    def _deferred_init():
        panel._apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        target = panel.get_settings().get("target_language", "zh")
        overlay.set_target_language(target)
        asr_lang = panel.get_settings().get("asr_language", "auto")
        overlay.set_source_language(asr_lang)
        style = panel.get_settings().get("style")
        if style:
            overlay.apply_style(style)
        active_model = panel.get_active_model()
        if active_model:
            live_trans._on_model_changed(active_model)

        # Load scene presets into overlay
        preset_names = panel.get_all_preset_names()
        default_preset = (saved or {}).get("default_preset", "\u5e26\u8d27\u76f4\u64ad")
        overlay.set_scenes(preset_names, default_preset)
        # Set initial preset on analyzer
        initial_preset = panel.get_preset(default_preset)
        if initial_preset:
            live_trans._analyzer.set_preset(initial_preset)

    QTimer.singleShot(100, _deferred_init)

    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    menu = QMenu()

    # --- Pause / Resume toggle ---
    pause_action = QAction(t("tray_pause"))
    _is_running = [True]  # mutable for closure

    def on_start():
        try:
            live_trans.start()
            overlay.set_running(True)
            _is_running[0] = True
            pause_action.setText(t("tray_pause"))
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)

    def on_pause():
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))

    def on_resume():
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

    def on_toggle_pause():
        if _is_running[0]:
            on_pause()
        else:
            on_resume()

    pause_action.triggered.connect(on_toggle_pause)
    menu.addAction(pause_action)
    menu.addSeparator()

    # --- Show/hide overlay ---
    overlay_toggle_action = QAction(t("tray_hide_overlay"))

    _hide_notified = [False]

    def on_toggle_overlay():
        if overlay.isVisible():
            overlay.hide()
            overlay_toggle_action.setText(t("tray_show_overlay"))
            if not _hide_notified[0]:
                _hide_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("hide_tray_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            overlay.show()
            overlay.raise_()
            overlay_toggle_action.setText(t("tray_hide_overlay"))

    overlay_toggle_action.triggered.connect(on_toggle_overlay)
    menu.addAction(overlay_toggle_action)

    # --- Subtitle window toggle ---
    def _save_overlay_pos():
        settings = panel.get_settings()
        pos = overlay.pos()
        size = overlay.size()
        settings["overlay_x"] = pos.x()
        settings["overlay_y"] = pos.y()
        settings["overlay_w"] = size.width()
        settings["overlay_h"] = size.height()
        panel._current_settings.update({
            "overlay_x": pos.x(), "overlay_y": pos.y(),
            "overlay_w": size.width(), "overlay_h": size.height(),
        })
        _save_settings(settings)

    overlay.position_changed.connect(_save_overlay_pos)

    subwin_toggle_action = QAction(t("subwin_show"), checkable=True)

    def _save_subwin_state():
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["enabled"] = subwin.isVisible()
        pos = subwin.pos()
        sm["window_x"] = pos.x()
        sm["window_y"] = pos.y()
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)

    _subwin_notified = [False]

    def on_toggle_subwin(checked):
        if checked:
            subwin.show()
            subwin.raise_()
            if not _subwin_notified[0]:
                _subwin_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("subwin_drag_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            subwin.hide()
        overlay.set_subtitle_checked(checked)
        _save_subwin_state()

    subwin_toggle_action.toggled.connect(on_toggle_subwin)
    subwin.position_changed.connect(_save_subwin_state)

    # Sync when subtitle window is manually closed (e.g. Alt+F4)
    def _on_subwin_closed():
        subwin_toggle_action.blockSignals(True)
        subwin_toggle_action.setChecked(False)
        subwin_toggle_action.blockSignals(False)
        overlay.set_subtitle_checked(False)
        _save_subwin_state()

    subwin.window_closed.connect(_on_subwin_closed)

    # Restore subtitle window visibility from saved state
    if subwin_was_enabled:
        subwin_toggle_action.setChecked(True)

    menu.addAction(subwin_toggle_action)

    # Connect overlay subtitle button
    def _on_overlay_subtitle_toggle():
        subwin_toggle_action.setChecked(not subwin_toggle_action.isChecked())

    overlay.subtitle_toggled.connect(_on_overlay_subtitle_toggle)

    # Connect panel subtitle settings changes
    def _on_panel_subtitle_changed(s):
        subwin.apply_settings(s)

    panel.subtitle_settings_changed.connect(_on_panel_subtitle_changed)

    def _on_reset_positions():
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        subwin.move(100, 100)
        _save_subwin_state()
        ow, oh = overlay.width(), overlay.height()
        overlay.move(geo.right() - ow - 50, geo.bottom() - oh - 100)
        _save_overlay_pos()

    panel.reset_positions.connect(_on_reset_positions)

    menu.addSeparator()

    # --- Show log / panel ---
    log_action = QAction(t("tray_show_log"))
    panel_action = QAction(t("tray_show_panel"))

    def on_toggle_log():
        if log_window.isVisible():
            log_window.hide()
        else:
            log_window.show()
            log_window.raise_()

    def on_toggle_panel():
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    log_action.triggered.connect(on_toggle_log)
    panel_action.triggered.connect(on_toggle_panel)
    menu.addAction(panel_action)
    menu.addAction(log_action)
    menu.addSeparator()

    # --- Overlay submenu (click-through, topmost, auto-scroll, taskbar) ---
    overlay_menu = QMenu(t("tray_menu_overlay"))

    ct_action = QAction(t("click_through"), checkable=True)
    topmost_action = QAction(t("top_most"), checkable=True)
    topmost_action.setChecked(True)
    autoscroll_action = QAction(t("auto_scroll"), checkable=True)
    autoscroll_action.setChecked(True)
    taskbar_action = QAction(t("taskbar"), checkable=True)

    # Tray → overlay sync
    ct_action.toggled.connect(lambda v: overlay._handle._ct_check.setChecked(v))
    topmost_action.toggled.connect(
        lambda v: overlay._handle._topmost_check.setChecked(v)
    )
    autoscroll_action.toggled.connect(
        lambda v: overlay._handle._auto_scroll.setChecked(v)
    )
    taskbar_action.toggled.connect(
        lambda v: overlay._handle._taskbar_check.setChecked(v)
    )

    # Overlay → tray sync
    overlay._handle.click_through_toggled.connect(lambda v: ct_action.setChecked(v))
    overlay._handle.topmost_toggled.connect(lambda v: topmost_action.setChecked(v))
    overlay._handle.auto_scroll_toggled.connect(
        lambda v: autoscroll_action.setChecked(v)
    )
    overlay._handle.taskbar_toggled.connect(lambda v: taskbar_action.setChecked(v))

    overlay_menu.addAction(ct_action)
    overlay_menu.addAction(topmost_action)
    overlay_menu.addAction(autoscroll_action)
    overlay_menu.addAction(taskbar_action)
    menu.addMenu(overlay_menu)

    # --- Model submenu ---
    model_menu = QMenu(t("tray_menu_model"))
    model_action_group = QActionGroup(model_menu)
    model_action_group.setExclusive(True)

    def _rebuild_model_menu():
        for a in model_action_group.actions():
            model_action_group.removeAction(a)
        model_menu.clear()
        settings = panel.get_settings()
        models = settings.get("models", [])
        active = settings.get("active_model", 0)
        for i, m in enumerate(models):
            name = m.get("name", m.get("model", "?"))
            action = QAction(name, checkable=True)
            if i == active:
                action.setChecked(True)
            model_action_group.addAction(action)
            action.triggered.connect(lambda checked, idx=i: _on_tray_model_switch(idx))
            model_menu.addAction(action)

    def _on_tray_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
            overlay.set_models(models, index)

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
        _rebuild_model_menu()

    model_menu.aboutToShow.connect(_rebuild_model_menu)
    menu.addMenu(model_menu)

    # --- Target language submenu ---
    lang_menu = QMenu(t("tray_menu_target_lang"))
    lang_action_group = QActionGroup(lang_menu)
    lang_action_group.setExclusive(True)
    _lang_actions = {}
    lang_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        if code == "auto":
            continue
        action = QAction(f"{code} - {native}", checkable=True)
        lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, lc=code: _on_tray_lang_switch(lc))
        if code in COMMON_LANG_CODES:
            lang_menu.addAction(action)
        else:
            lang_more_menu.addAction(action)
        _lang_actions[code] = action

    lang_menu.addMenu(lang_more_menu)

    current_target = panel.get_settings().get("target_language", "zh")
    if current_target in _lang_actions:
        _lang_actions[current_target].setChecked(True)

    def _on_tray_lang_switch(lang_code):
        overlay.set_target_language(lang_code)
        live_trans._on_target_language_changed(lang_code)
        from control_panel import _save_settings

        settings = panel.get_settings()
        settings["target_language"] = lang_code
        panel._current_settings["target_language"] = lang_code
        _save_settings(settings)

    # Overlay → tray lang sync
    def _on_overlay_lang_changed(lang_code):
        if lang_code in _lang_actions:
            _lang_actions[lang_code].setChecked(True)

    overlay.target_language_changed.connect(_on_overlay_lang_changed)

    menu.addMenu(lang_menu)

    # --- ASR language hint submenu ---
    asr_lang_menu = QMenu(t("tray_menu_asr_lang"))
    asr_lang_action_group = QActionGroup(asr_lang_menu)
    asr_lang_action_group.setExclusive(True)
    _asr_lang_actions = {}
    asr_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        label = t("asr_lang_auto") if code == "auto" else native
        action = QAction(f"{code} - {label}", checkable=True)
        asr_lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, c=code: _on_tray_asr_lang(c))
        if code in COMMON_LANG_CODES:
            asr_lang_menu.addAction(action)
        else:
            asr_more_menu.addAction(action)
        _asr_lang_actions[code] = action

    asr_lang_menu.addMenu(asr_more_menu)

    current_asr_lang = panel.get_settings().get("asr_language", "auto")
    if current_asr_lang in _asr_lang_actions:
        _asr_lang_actions[current_asr_lang].setChecked(True)

    def _on_tray_asr_lang(code):
        from control_panel import _save_settings

        if live_trans._asr:
            live_trans._asr.set_language(code)
        settings = panel.get_settings()
        settings["asr_language"] = code
        panel._current_settings["asr_language"] = code
        _save_settings(settings)
        # Sync control panel combo
        idx = panel._asr_lang.findData(code)
        if idx >= 0:
            panel._asr_lang.blockSignals(True)
            panel._asr_lang.setCurrentIndex(idx)
            panel._asr_lang.blockSignals(False)

    menu.addMenu(asr_lang_menu)
    menu.addSeparator()

    # --- Quit ---
    quit_action = QAction(t("quit"))

    def on_quit():
        live_trans.stop()
        app.quit()

    quit_action.triggered.connect(on_quit)
    menu.addAction(quit_action)

    # --- Connect overlay signals ---
    overlay.settings_requested.connect(on_toggle_panel)
    overlay.target_language_changed.connect(live_trans._on_target_language_changed)

    def _on_overlay_source_lang(code):
        """Overlay source language combo → sync to panel + ASR engine + tray."""
        _on_tray_asr_lang(code)
        overlay.set_source_language(code)

    def _on_panel_asr_lang_changed(_index):
        """Panel ASR language combo → sync to overlay."""
        code = panel._asr_lang.currentData() or "auto"
        overlay.set_source_language(code)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel._asr_lang.currentIndexChanged.connect(_on_panel_asr_lang_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)
    overlay.start_requested.connect(on_resume)
    overlay.stop_requested.connect(on_pause)
    overlay.hide_requested.connect(on_toggle_overlay)
    overlay.quit_requested.connect(on_quit)
    overlay.scene_changed.connect(live_trans._on_scene_changed)
    overlay.analyze_requested.connect(live_trans._analyzer.trigger_manual)
    overlay.clear_analysis_history.connect(live_trans._analyzer.clear_history)
    overlay.retain_history_changed.connect(live_trans._analyzer.set_retain_history)
    overlay.clear_signal.connect(lambda: live_trans._dialogue_buffer.clear())

    tray.setContextMenu(menu)
    tray.show()

    QTimer.singleShot(500, on_start)

    signal.signal(signal.SIGINT, lambda *_: on_quit())
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
