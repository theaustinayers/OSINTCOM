"""
OSINTCOM - Cross-platform Voice Activity Detection and Recording Tool
PyQt5 GUI with real-time audio monitoring, VAD, recording, and Discord integration.
"""

import sys
import os
import time
import threading
import datetime
import json
import collections
import traceback

# FIX: Set Qt plugin path for cx_Freeze bundled executable
_base_path = os.path.dirname(sys.executable if hasattr(sys, 'frozen') else __file__)
_plugin_path = os.path.join(_base_path, 'PyQt5', 'plugins')
if os.path.exists(_plugin_path):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _plugin_path

import numpy as np
import sounddevice as sd
import requests
from scipy.signal import butter, sosfiltfilt, welch, find_peaks
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QDialog, QLineEdit,
    QGroupBox, QStatusBar, QSlider, QTextEdit, QDialogButtonBox, QMessageBox, QCheckBox,
    QListWidget, QListWidgetItem, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPainter, QColor, QLinearGradient, QFontDatabase

# ============================================================================
# Constants
# ============================================================================
CONFIG_FILE = "osintcom_config.json"

CHANNELS = 1
BLOCK_SIZE = 2048
PRE_ROLL_SECONDS = 5.0  # 5 seconds of audio before VAD triggers
MIN_RECORDING_DURATION = 1.5

# Formant-Primary VAD v1.36 — Sensitivity presets
# Scores voice 0-100: Formants(40) + VoiceBand(20) + SNR(15) + Pitch(15) + Modulation(10)
# Thresholds calibrated from real HF captures (14405 kHz S8/S9 voice, 14000 kHz noise floor).
# Key finding: S8/S9 voice SNR is only 0-4 dB above noise floor — all thresholds must be
# reachable in that range. word_peak_threshold must be <= observed max confidence (~57%).
#
# v1.27 NB-noise baseline (11175 kHz, NB on, no voice): confidence EMA ≈ 41%.
# Voice baseline (NB on, 14321 kHz): confidence EMA ≈ 43–44%.
# confidence_stop is the EMA stop threshold: recording ends when EMA drops below this
# after hangover expires. L3 stop=42 sits just above noise EMA (41%) and below voice EMA (43%).
# max_recording_duration caps runaway false-positive recordings (seconds).
# NB NOTE: Noise Blanker on your radio is recommended for best voice detection.
#          With NB on a quiet/inactive frequency, use L3+ to avoid false triggers.
# confirm_window_seconds:    how many seconds of audio chunks to observe (converted to chunk count).
# confirm_min_ratio:          fraction of EXACTLY confirm_window_chunks that must be >= start_threshold.
# confirm_min_run_chunks:     longest CONSECUTIVE run of frames >= threshold required to START recording.
#   Using chunk count (not wall clock) makes the gate immune to audio callback batching.
#   Data: 11175 noise max 7/64 hits, max run=3 — well below ratio=20% (13 hits) and run=5.
# hangover_repin_run_chunks:  SEPARATE run gate used DURING recording to hold the hangover open.
#   Decoupled from confirm_min_run_chunks so voice can sustain recording without needing
#   the same strict run as confirm. Voice easily hits 3 consecutive frames; noise crashes
#   (max run=4 at threshold) may occasionally tie this but EMA-stop still governs.
SENSITIVITY_PRESETS = {
    # Level 1: Maximum sensitivity — catches very weak/faint stations, more false positives.
    1: {"confidence_start": 35, "confidence_continue": 18, "confidence_stop": 36,
        "word_peak_threshold": 42, "post_roll_seconds": 10, "max_recording_duration": 600,
        "confirm_window_seconds": 3.0, "confirm_min_ratio": 0.15, "confirm_min_run_chunks": 3,
        "hangover_repin_run_chunks": 3,
        "formant_threshold_db": 3, "formant_prominence_db": 3.5,
        "flat_penalty_factor": 0.65, "min_formants": 1, "noise_floor_db": -68},
    # Level 2: Very sensitive — good for weak/faint SSB/USB.
    # Confirm gate: run=6 (noise max run at >=46% = 4 → blocked by 2 chunks).
    # Hangover repin: run=3 — voice holds open with just 0.14s sustained above word_peak.
    # This prevents fragmented recordings where confirm is strict but sustain is lenient.
    2: {"confidence_start": 46, "confidence_continue": 35, "confidence_stop": 48,
        "word_peak_threshold": 52, "post_roll_seconds": 10, "max_recording_duration": 480,
        "confirm_window_seconds": 3.0, "confirm_min_ratio": 0.18, "confirm_min_run_chunks": 6,
        "hangover_repin_run_chunks": 3,
        "formant_threshold_db": 4, "formant_prominence_db": 4.5,
        "flat_penalty_factor": 0.55, "min_formants": 1, "noise_floor_db": -65},
    # Level 3: Balanced (default) — data-validated for HFGCS frequencies.
    # 11175 noise: max run=4 @ >=50%, ratio=10.9%. 8992 noise: max run=6, ratio=21.5%.
    # Gates raised to cover worst observed HF band: run=7 (noise max=6), ratio=25% (noise max=21.5%).
    # Voice min word run at 21.5ch/s = 8+ chunks → passes both gates comfortably.
    3: {"confidence_start": 50, "confidence_continue": 35, "confidence_stop": 42,
        "word_peak_threshold": 55, "post_roll_seconds": 10, "max_recording_duration": 300,
        "confirm_window_seconds": 3.0, "confirm_min_ratio": 0.25, "confirm_min_run_chunks": 7,
        "hangover_repin_run_chunks": 3,
        "formant_threshold_db": 5, "formant_prominence_db": 6.0,
        "flat_penalty_factor": 0.40, "min_formants": 1, "noise_floor_db": -60},
    # Level 4: Strict — requires strong formant + voiceband validation.
    4: {"confidence_start": 60, "confidence_continue": 42, "confidence_stop": 47,
        "word_peak_threshold": 65, "post_roll_seconds": 10, "max_recording_duration": 240,
        "confirm_window_seconds": 3.0, "confirm_min_ratio": 0.25, "confirm_min_run_chunks": 6,
        "hangover_repin_run_chunks": 4,
        "formant_threshold_db": 7, "formant_prominence_db": 7.5,
        "flat_penalty_factor": 0.25, "min_formants": 2, "noise_floor_db": -55},
    # Level 5: Voice only — strong signal required, minimum false positives.
    5: {"confidence_start": 68, "confidence_continue": 50, "confidence_stop": 52,
        "word_peak_threshold": 72, "post_roll_seconds": 10, "max_recording_duration": 180,
        "confirm_window_seconds": 3.0, "confirm_min_ratio": 0.30, "confirm_min_run_chunks": 8,
        "hangover_repin_run_chunks": 4,
        "formant_threshold_db": 8, "formant_prominence_db": 9.0,
        "flat_penalty_factor": 0.15, "min_formants": 2, "noise_floor_db": -50},
}

SENSITIVITY_LABELS = {
    1: "1 – Most Sensitive", 2: "2 – Relaxed", 3: "3 – Balanced (Default)",
    4: "4 – Strict", 5: "5 – Voice Only",
}

# ============================================================================
# Signals
# ============================================================================
class WorkerSignals(QObject):
    level = pyqtSignal(float)
    voice = pyqtSignal(bool)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    recording = pyqtSignal(bool)  # True when recording starts, False when stops

# ============================================================================
# Audio Meter Widget
# ============================================================================
class AudioMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setMinimumHeight(32)
        self.setMaximumHeight(48)

    def set_level(self, db: float):
        clamped = max(-60.0, min(0.0, db))
        self._level = (clamped + 60.0) / 60.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        bar_w = int(w * self._level)
        if bar_w > 0:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(0, 200, 0))
            grad.setColorAt(0.6, QColor(255, 255, 0))
            grad.setColorAt(1.0, QColor(255, 0, 0))
            painter.fillRect(0, 0, bar_w, h, grad)
        painter.setPen(QColor(80, 80, 80))
        painter.drawRect(0, 0, w - 1, h - 1)
        db_val = (self._level * 60.0) - 60.0
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(4, h - 6, f"{db_val:+.1f} dB")
        painter.end()

# ============================================================================
# Animated Ticker Widget
# ============================================================================
class AnimatedTicker(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setMaximumHeight(60)
        self.setStyleSheet("background-color: #000000; border: 1px solid #333333;")
        
        # Load ShareTechMono font
        self.ticker_font = QFont("Monospace", 20)  # Fallback font
        try:
            font_path = os.path.join(os.path.dirname(__file__), "ShareTechMono-Regular.ttf")
            if os.path.exists(font_path):
                font_id = QFontDatabase.addApplicationFont(font_path)
                if font_id >= 0:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        self.ticker_font = QFont(families[0], 20)
        except Exception as e:
            print(f"Warning: Could not load ShareTechMono font: {e}")
        
        self._text = "***EAM INCOMING***"
        self._offset = 0
        self._is_active = False
        self._flash_opacity = 1.0
        self._flash_increasing = False
        
        # Timer for animation
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.setInterval(30)  # ~33 FPS
    
    def start_animation(self):
        """Start the scrolling animation."""
        self._is_active = True
        self._offset = 0
        self._flash_opacity = 1.0
        self.anim_timer.start()
    
    def stop_animation(self):
        """Stop the scrolling animation."""
        self._is_active = False
        self.anim_timer.stop()
        self._offset = 0
        self.update()
    
    def _update_animation(self):
        """Update animation state."""
        if not self._is_active:
            return
        
        # Scroll text
        self._offset += 3
        text_width = len(self._text) * 14
        if self._offset > text_width + self.width():
            self._offset = -self.width()
        
        # Flash effect
        if self._flash_increasing:
            self._flash_opacity += 0.05
            if self._flash_opacity >= 1.0:
                self._flash_opacity = 1.0
                self._flash_increasing = False
        else:
            self._flash_opacity -= 0.05
            if self._flash_opacity <= 0.5:
                self._flash_opacity = 0.5
                self._flash_increasing = True
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the animated ticker."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        if self._is_active:
            # Draw scrolling text with flash effect
            painter.setFont(self.ticker_font)
            
            # Set color with flash opacity (red with flashing)
            color = QColor(255, 0, 0)  # Red
            color.setAlpha(int(255 * self._flash_opacity))
            painter.setPen(color)
            
            # Draw text at offset position (scrolling from right to left)
            painter.drawText(self._offset, self.height() // 2 + 8, self._text)
        else:
            # Idle state - dim text
            painter.setFont(self.ticker_font)
            painter.setPen(QColor(64, 64, 64))
            painter.drawText(10, self.height() // 2 + 8, "READY")
        
        painter.end()

# ============================================================================
# Dialogs
# ============================================================================
class WebhookManagerDialog(QDialog):
    def __init__(self, webhooks: list = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Discord Webhooks")
        self.setMinimumSize(700, 500)
        
        self.webhooks = webhooks or []  # List of {"nickname": str, "url": str, "enabled": bool, "role_id": str}
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Manage Discord Webhooks:", font=QFont("Segoe UI", 11, QFont.Bold)))
        layout.addWidget(QLabel("Enable/disable webhooks and manage destinations for recording uploads"))
        layout.addSpacing(8)
        
        # Webhook list
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(QLabel("Configured Webhooks:"))
        layout.addWidget(self.list_widget)
        
        # Edit panel
        edit_group = QGroupBox("Edit Webhook")
        edit_layout = QVBoxLayout(edit_group)
        
        nickname_layout = QHBoxLayout()
        nickname_layout.addWidget(QLabel("Nickname:"))
        self.nickname_edit = QLineEdit()
        self.nickname_edit.setPlaceholderText("e.g., Main Server, Backup, Archive...")
        nickname_layout.addWidget(self.nickname_edit)
        edit_layout.addLayout(nickname_layout)
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://discord.com/api/webhooks/...")
        url_layout.addWidget(self.url_edit)
        edit_layout.addLayout(url_layout)
        
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role ID (optional):"))
        self.role_edit = QLineEdit()
        self.role_edit.setPlaceholderText("e.g., 1474631083042541730 (leave blank to not ping)")
        role_layout.addWidget(self.role_edit)
        edit_layout.addLayout(role_layout)
        
        self.enabled_chk = QCheckBox("Enabled (will receive uploads)")
        self.enabled_chk.setChecked(True)
        edit_layout.addWidget(self.enabled_chk)
        
        layout.addWidget(edit_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Webhook")
        add_btn.clicked.connect(self._add_webhook)
        button_layout.addWidget(add_btn)
        
        update_btn = QPushButton("Update Selected")
        update_btn.clicked.connect(self._update_webhook)
        button_layout.addWidget(update_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_webhook)
        button_layout.addWidget(remove_btn)
        layout.addLayout(button_layout)
        
        layout.addSpacing(12)
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(dialog_buttons)
        
        self._refresh_list()
    
    def _refresh_list(self):
        """Refresh the webhook list display."""
        self.list_widget.clear()
        for i, webhook in enumerate(self.webhooks):
            status = "✓" if webhook.get("enabled", True) else "✗"
            nickname = webhook.get("nickname", "Unnamed")
            url_short = webhook.get("url", "")[-20:] if webhook.get("url") else ""
            text = f"{status} {nickname} ({url_short})"
            self.list_widget.addItem(text)
    
    def _on_selection_changed(self):
        """Load selected webhook into edit fields."""
        current_row = self.list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.webhooks):
            webhook = self.webhooks[current_row]
            self.nickname_edit.setText(webhook.get("nickname", ""))
            self.url_edit.setText(webhook.get("url", ""))
            self.role_edit.setText(webhook.get("role_id", ""))
            self.enabled_chk.setChecked(webhook.get("enabled", True))
    
    def _add_webhook(self):
        """Add a new webhook."""
        nickname = self.nickname_edit.text().strip()
        url = self.url_edit.text().strip()
        role_id = self.role_edit.text().strip()
        
        if not nickname:
            QMessageBox.warning(self, "Missing Nickname", "Please enter a nickname for the webhook")
            return
        if not url:
            QMessageBox.warning(self, "Missing URL", "Please enter a webhook URL")
            return
        
        # Check for duplicates
        if any(w.get("nickname") == nickname for w in self.webhooks):
            QMessageBox.warning(self, "Duplicate Nickname", f"A webhook with nickname '{nickname}' already exists")
            return
        
        self.webhooks.append({
            "nickname": nickname,
            "url": url,
            "role_id": role_id,
            "enabled": self.enabled_chk.isChecked()
        })
        
        self.nickname_edit.clear()
        self.url_edit.clear()
        self.role_edit.clear()
        self.enabled_chk.setChecked(True)
        self._refresh_list()
        QMessageBox.information(self, "Success", f"Webhook '{nickname}' added")
    
    def _update_webhook(self):
        """Update selected webhook."""
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a webhook to update")
            return
        
        nickname = self.nickname_edit.text().strip()
        url = self.url_edit.text().strip()
        role_id = self.role_edit.text().strip()
        
        if not nickname:
            QMessageBox.warning(self, "Missing Nickname", "Please enter a nickname for the webhook")
            return
        if not url:
            QMessageBox.warning(self, "Missing URL", "Please enter a webhook URL")
            return
        
        # Check for duplicate nicknames (excluding current)
        for i, w in enumerate(self.webhooks):
            if i != current_row and w.get("nickname") == nickname:
                QMessageBox.warning(self, "Duplicate Nickname", f"A webhook with nickname '{nickname}' already exists")
                return
        
        self.webhooks[current_row] = {
            "nickname": nickname,
            "url": url,
            "role_id": role_id,
            "enabled": self.enabled_chk.isChecked()
        }
        
        self._refresh_list()
        QMessageBox.information(self, "Success", f"Webhook '{nickname}' updated")
    
    def _remove_webhook(self):
        """Remove selected webhook."""
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a webhook to remove")
            return
        
        nickname = self.webhooks[current_row].get("nickname", "Unnamed")
        confirm = QMessageBox.question(self, "Confirm Removal", 
                                      f"Remove webhook '{nickname}'?",
                                      QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.webhooks.pop(current_row)
            self.nickname_edit.clear()
            self.url_edit.clear()
            self._refresh_list()
    
    def get_webhooks(self) -> list:
        return self.webhooks

class WebhookCustomizeDialog(QDialog):
    def __init__(self, message_template: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Webhook Message")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Message text:"))
        self.msg_edit = QTextEdit()
        self.msg_edit.setPlainText(message_template or "EMERGENCY ACTION MESSAGE INCOMING")
        self.msg_edit.setFixedHeight(60)
        layout.addWidget(self.msg_edit)
        layout.addWidget(QLabel("Preview:"))
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #a6adc8; padding: 6px; background: #313244; border-radius: 4px;")
        layout.addWidget(self.preview_label)
        self.msg_edit.textChanged.connect(self._update_preview)
        self._update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self, *_):
        msg = self.msg_edit.toPlainText().strip() or "EMERGENCY ACTION MESSAGE INCOMING"
        freq = "8992 kHz"
        self.preview_label.setText(f"{msg}\n─ {freq} | {datetime.datetime.utcnow().isoformat()}Z")

    def get_message_template(self) -> str:
        return self.msg_edit.toPlainText().strip()

class AudioSettingsDialog(QDialog):
    """Dialog for configuring audio processing options."""
    def __init__(self, current_settings: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Processing Settings")
        self.setMinimumWidth(500)
        
        self.settings = current_settings or {}
        
        layout = QVBoxLayout(self)
        
        # Bandpass Filter
        layout.addWidget(QLabel("Audio Processing Options:", font=QFont("Segoe UI", 11, QFont.Bold)))
        layout.addSpacing(8)
        
        self.use_bandpass_chk = QCheckBox("Use Bandpass Filter (300-3000 Hz for SSB)")
        self.use_bandpass_chk.setChecked(self.settings.get("use_bandpass", False))
        self.use_bandpass_chk.setToolTip("Removes static outside speech frequency range")
        layout.addWidget(self.use_bandpass_chk)
        
        # Enhanced Denoise
        self.use_denoise_chk = QCheckBox("Enhanced Noise Reduction")
        self.use_denoise_chk.setChecked(self.settings.get("use_denoise", True))
        self.use_denoise_chk.setToolTip("Apply aggressive noise reduction (slower)")
        layout.addWidget(self.use_denoise_chk)
        
        # Denoise strength slider
        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(QLabel("Denoise Strength:"))
        self.denoise_strength = QSlider(Qt.Horizontal)
        self.denoise_strength.setMinimum(1)
        self.denoise_strength.setMaximum(10)
        self.denoise_strength.setValue(self.settings.get("denoise_strength", 6))
        self.denoise_strength.setTickPosition(QSlider.TicksBelow)
        self.denoise_strength.setTickInterval(1)
        denoise_layout.addWidget(self.denoise_strength)
        self.denoise_label = QLabel(str(self.settings.get("denoise_strength", 6)))
        denoise_layout.addWidget(self.denoise_label)
        self.denoise_strength.valueChanged.connect(lambda v: self.denoise_label.setText(str(v)))
        layout.addLayout(denoise_layout)
        layout.addSpacing(8)
        
        # Silence Removal
        self.remove_silence_chk = QCheckBox("Remove Silent Gaps")
        self.remove_silence_chk.setChecked(self.settings.get("remove_silence", False))
        self.remove_silence_chk.setToolTip("Remove quiet periods between speech bursts")
        layout.addWidget(self.remove_silence_chk)
        
        # Voice Extraction
        self.voice_extract_chk = QCheckBox("Voice-Only Extraction (Speech Bursts Only)")
        self.voice_extract_chk.setChecked(self.settings.get("voice_extract", False))
        self.voice_extract_chk.setToolTip("Send only clear voice segments, remove all background")
        layout.addWidget(self.voice_extract_chk)
        
        # Voice extraction confidence threshold slider
        extract_threshold_layout = QHBoxLayout()
        extract_threshold_layout.addWidget(QLabel("Voice Extraction Confidence Threshold:"))
        self.voice_extract_threshold = QSlider(Qt.Horizontal)
        self.voice_extract_threshold.setMinimum(45)
        self.voice_extract_threshold.setMaximum(70)
        self.voice_extract_threshold.setValue(self.settings.get("voice_extract_threshold", 58))
        self.voice_extract_threshold.setTickPosition(QSlider.TicksBelow)
        self.voice_extract_threshold.setTickInterval(1)
        extract_threshold_layout.addWidget(self.voice_extract_threshold)
        self.extract_threshold_label = QLabel(str(self.settings.get("voice_extract_threshold", 58)) + "%")
        extract_threshold_layout.addWidget(self.extract_threshold_label)
        self.voice_extract_threshold.valueChanged.connect(lambda v: self.extract_threshold_label.setText(str(v) + "%"))
        layout.addLayout(extract_threshold_layout)
        
        layout.addSpacing(12)
        layout.addWidget(QLabel("ℹ️ Voice-Only mode sends cleanest audio but may cut very quiet voices. Lower threshold = include more borderline segments.", 
                               font=QFont("Segoe UI", 9)))
        
        layout.addStretch()
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> dict:
        return {
            "use_bandpass": self.use_bandpass_chk.isChecked(),
            "use_denoise": self.use_denoise_chk.isChecked(),
            "denoise_strength": self.denoise_strength.value(),
            "remove_silence": self.remove_silence_chk.isChecked(),
            "voice_extract": self.voice_extract_chk.isChecked(),
            "voice_extract_threshold": self.voice_extract_threshold.value(),
        }

# ============================================================================
# Main Window
# ============================================================================
class OSINTCOMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSINTCOM")
        self.setMinimumSize(350, 150)  # Allow resizing down to compact size
        self.setFont(QFont("Segoe UI", 10))
        
        # Compact mode tracking
        self._compact_mode = False
        self._normal_size = (900, 800)
        self._last_compact_size = (350, 200)
        self._normal_widget = None
        self._compact_widget = None
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
        
        # State
        self._running = False
        self._recording = False
        self._stream = None
        self._sample_rate = 48000  # Default matches FlexRadio DAX IQ; overwritten from device at runtime
        self._lock = threading.Lock()
        self._ring_buffer = collections.deque(maxlen=int(PRE_ROLL_SECONDS * self._sample_rate / BLOCK_SIZE))
        self._audio_buffer_lock = threading.Lock()
        self._audio_buffer = []
        self._peak_db = -60.0
        self._voice_detected = False
        self._voice_started_at = None
        self._voice_silence_at = None
        self._silence_timer_remaining = 0
        self._sensitivity_level = 3
        self._webhooks = []  # List of {"nickname": str, "url": str, "enabled": bool, "role_id": str}
        self._webhook_message = "EMERGENCY ACTION MESSAGE INCOMING"
        self._frequency = ""
        self._save_dir = os.path.join(os.path.expanduser("~"), "Documents", "OSINTCOM")
        self._signals = WorkerSignals()
        
        # DEBUG MODE: Enable for detailed VAD diagnostics
        # Set to True to see confidence scores, noise floor, thresholds in console
        self._meter_debug = True  # ENABLE DEBUG: Shows all VAD calculations
        
        # Adaptive noise floor learning
        self._noise_floor_db = -25.0  # Initial estimate
        self._noise_samples = collections.deque(maxlen=100)  # Last 100 samples
        self._noise_learning_time = 3.0  # Learn for 3 seconds on startup
        self._noise_relearn_interval = 240.0  # Re-learn every 4 minutes (240 seconds)
        self._last_learning_time = time.time()
        self._last_relearn_time = time.time()
        self._learning_enabled = True
        self._learning_phase = "startup"  # "startup" or "periodic"
        
        # Noise calibration
        self._calibration_active = False
        self._calibration_samples = []
        self._calibration_start = None
        
        # Automatic periodic calibration (every 5 minutes when no voice detected)
        self._last_confirmed_voice_time = None
        self._periodic_calibration_enabled = True
        self._calibration_interval_seconds = 300
        
        # Voice confidence tracking
        self._voice_confidence_duration = 0.0  # Cumulative seconds of high confidence
        self._last_high_confidence_time = None  # Timestamp of last high confidence frame
        self._last_pitch_score = 0.0  # Store pitch score for voice-only detection
        self._hangover_remaining = 0.0  # Post-roll hangover countdown
        self._hangover_voice_run = 0       # Consecutive frames >= word_peak_threshold (run gate)
        self._low_confidence_frames = 0  # Track consecutive low-confidence frames
        self._post_roll_silence_frames = 0  # Track silence during post-roll

        # Confirmation-window state (v1.36 — chunk-count based, not wall-clock)
        # A single frame above start_threshold enters CONFIRMING; recording only begins
        # after BOTH confirm_min_ratio AND confirm_min_run_chunks are satisfied over
        # exactly confirm_window_chunks chunks. Wall-clock is NOT used for the gate
        # decision — this prevents audio-callback burst batching from spoofing the window.
        self._confirming = False              # True while in CONFIRMING state
        self._confirm_hit_count = 0           # Frames >= start_threshold seen so far
        self._confirm_total_count = 0         # Total frames counted in window
        self._confirm_current_run = 0         # Current consecutive-hit streak
        self._confirm_max_run = 0             # Best consecutive-hit streak this window
        self._confirm_start_time = None       # Wall-clock only for status-bar display
        
        # Audio processing settings
        self._audio_settings = {
            "use_bandpass": False,
            "use_denoise": True,
            "denoise_strength": 6,
            "remove_silence": False,
            "voice_extract": False,
            "voice_extract_threshold": 58,
        }
        
        os.makedirs(self._save_dir, exist_ok=True)
        
        self._init_ui()
        self._connect_signals()
        self._init_timer()
        self._load_config()

    def _init_ui(self):
        # Apply modern dark theme
        dark_stylesheet = """
        QMainWindow, QWidget {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QGroupBox {
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 4px;
            padding-top: 8px;
            margin-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QLabel {
            color: #e0e0e0;
        }
        QPushButton {
            background-color: #2d5016;
            color: #e0e0e0;
            border: 1px solid #3d6b1f;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3d6b1f;
        }
        QPushButton:pressed {
            background-color: #1d3d0f;
        }
        QPushButton:disabled {
            background-color: #444;
            color: #888;
        }
        QComboBox {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a3e;
            color: #e0e0e0;
            selection-background-color: #3d6b1f;
        }
        QLineEdit {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
        }
        QSlider::groove:horizontal {
            background-color: #444;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background-color: #3d6b1f;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QStatusBar {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border-top: 1px solid #444;
        }
        """
        self.setStyleSheet(dark_stylesheet)
        
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel("OSINTCOM")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Version Label
        version_label = QLabel("v1.36")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        version_label.setToolTip("OSINTCOM v1.36 — Chunk-count confirm gate (wall-clock exploit closed)\nDual gate: ratio + run over exactly N chunks. Enable NB — NEVER use NR")
        layout.addWidget(version_label)
        
        # Animated Ticker Display
        self.ticker = AnimatedTicker()
        layout.addWidget(self.ticker)

        # HFGCS Frequencies
        freq_group = QGroupBox("HFGCS Frequencies")
        freq_layout = QVBoxLayout(freq_group)
        self.freq_buttons = []
        
        # Main HFGCS frequencies (big 4)
        main_frequencies = [4724.0, 8992.0, 11175.0, 15016.0]
        main_freq_layout = QHBoxLayout()
        for freq in main_frequencies:
            btn = QPushButton(f"{freq:.0f} kHz")
            btn.setCheckable(True)
            btn.setMinimumWidth(110)
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, f=freq: self._on_freq_selected(f))
            main_freq_layout.addWidget(btn)
            self.freq_buttons.append(btn)
        freq_layout.addLayout(main_freq_layout)
        
        # Charlie frequencies (smaller buttons)
        charlie_freqs = {
            "Charlie Alpha": 6691.0,
            "Charlie Bravo": 11187.0,
            "Charlie Charlie": 17892.0,
            "Charlie Delta": 3038.0,
            "Charlie Echo": 9031.0,
            "Charlie Foxtrot": 4703.0,
            "Charlie Golf": 8974.0,
            "Charlie Hotel": 11264.5,
        }
        charlie_freq_layout = QHBoxLayout()
        for label, freq in charlie_freqs.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumWidth(95)
            btn.clicked.connect(lambda checked, f=freq: self._on_freq_selected(f))
            charlie_freq_layout.addWidget(btn)
            self.freq_buttons.append(btn)
        freq_layout.addLayout(charlie_freq_layout)
        
        # Keep reference to all frequencies for consistency
        self.hfgcs_frequencies = main_frequencies + list(charlie_freqs.values())
        layout.addWidget(freq_group)

        # Audio Interface
        audio_group = QGroupBox("Audio Interface")
        audio_layout = QVBoxLayout(audio_group)
        self.device_combo = QComboBox()
        self._populate_devices()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        audio_layout.addWidget(self.device_combo)
        self.audio_meter = AudioMeter()
        audio_layout.addWidget(self.audio_meter)
        self.meter_label = QLabel("-60.0 dB")
        audio_layout.addWidget(self.meter_label)
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity:"))
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setMinimum(1)
        self.sens_slider.setMaximum(5)
        self.sens_slider.setValue(3)
        self.sens_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_row.addWidget(self.sens_slider)
        self.sens_label = QLabel(SENSITIVITY_LABELS[3])
        sens_row.addWidget(self.sens_label)
        self.sens_slider.setToolTip(
            "Sensitivity level for voice detection.\n"
            "Tip: Enable Noise Blanker (NB) on your radio for best detection results.\n"
            "Warning: Noise Reduction (NR) will suppress voice — keep NR OFF."
        )
        audio_layout.addLayout(sens_row)

        # NB/NR recommendation tip
        nb_tip = QLabel("⚡ NB on  |  NR off  — for best voice detection")
        nb_tip.setStyleSheet("color: #f0c040; font-size: 9px; padding: 1px 2px;")
        nb_tip.setWordWrap(True)
        nb_tip.setToolTip(
            "Noise Blanker (NB): Enable on your radio — improves pitch detection on SSB.\n"
            "Noise Reduction (NR): KEEP OFF — NR drops SNR by ~6dB and kills voice detection.\n"
            "Validated on 3841, 3907, 3993, 11175, 11253, 14200 kHz captures."
        )
        audio_layout.addWidget(nb_tip)

        self.voice_label = QLabel("static")
        self.voice_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
        audio_layout.addWidget(self.voice_label)
        
        # Denoiser checkbox
        denoise_row = QHBoxLayout()
        self.denoise_check = QCheckBox("Apply Denoiser")
        self.denoise_check.setChecked(HAS_NOISEREDUCE)  # Only check if available
        if not HAS_NOISEREDUCE:
            self.denoise_check.setEnabled(False)
        denoise_row.addWidget(self.denoise_check)
        denoise_row.addStretch()
        audio_layout.addLayout(denoise_row)
        layout.addWidget(audio_group)

        # Discord Webhook
        discord_group = QGroupBox("Discord Webhook")
        discord_layout = QVBoxLayout(discord_group)
        self.freq_display = QLineEdit()
        self.freq_display.setReadOnly(True)
        self.freq_display.setPlaceholderText("Frequency")
        discord_layout.addWidget(self.freq_display)
        webhook_row = QHBoxLayout()
        self.webhook_edit = QLineEdit()
        self.webhook_edit.setPlaceholderText("Webhook URL")
        webhook_row.addWidget(self.webhook_edit)
        webhook_btn = QPushButton("Webhooks")
        webhook_btn.clicked.connect(self._open_webhook_dialog)
        webhook_row.addWidget(webhook_btn)
        discord_layout.addLayout(webhook_row)
        msg_row = QHBoxLayout()
        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText("Customize Message")
        msg_row.addWidget(self.message_edit)
        msg_btn = QPushButton("Customize")
        msg_btn.clicked.connect(self._open_customize_dialog)
        msg_row.addWidget(msg_btn)
        discord_layout.addLayout(msg_row)
        layout.addWidget(discord_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.file_btn = QPushButton("File Location")
        self.file_btn.clicked.connect(self._on_file_location)
        self.audio_settings_btn = QPushButton("Audio Settings")
        self.audio_settings_btn.clicked.connect(self._open_audio_settings)
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._save_config)
        self.calibrate_btn = QPushButton("Calibrate Noise")
        self.calibrate_btn.clicked.connect(self._on_calibrate_noise)
        self.shrink_btn = QPushButton("↓ Shrink")
        self.shrink_btn.clicked.connect(lambda: self._switch_to_compact(remember_size=True))
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.file_btn)
        controls_layout.addWidget(self.audio_settings_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.calibrate_btn)
        controls_layout.addWidget(self.shrink_btn)
        layout.addWidget(controls_group)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | VAD: Idle | Recording: Stopped | Pre-roll: 5s")

        # Store normal widget
        self._normal_widget = central
        
        # Build compact widget upfront
        if self._compact_widget is None:
            self._compact_widget = self._build_compact_widget()
        
        # Create a stacked widget to manage both views safely
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self._normal_widget)      # Index 0 = normal
        self.stacked_widget.addWidget(self._compact_widget)      # Index 1 = compact
        self.stacked_widget.setCurrentIndex(0)                   # Start with normal
        
        # Set stacked widget as central widget
        self.setCentralWidget(self.stacked_widget)

    def _connect_signals(self):
        self._signals.level.connect(self._on_level)
        self._signals.voice.connect(self._on_voice)
        self._signals.status.connect(self._on_status)
        self._signals.error.connect(self._on_error)
        self._signals.recording.connect(self._on_recording)

    def _build_compact_widget(self) -> QWidget:
        """Build minimal compact monitoring widget"""
        compact = QWidget()
        layout = QVBoxLayout(compact)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Frequency display
        freq_label = QLabel("Frequency:")
        freq_label.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        self.compact_freq_display = QLabel(self._frequency or "None")
        self.compact_freq_display.setStyleSheet("font-size: 16px; font-weight: bold; color: #3d6b1f;")
        freq_row = QHBoxLayout()
        freq_row.addWidget(freq_label)
        freq_row.addWidget(self.compact_freq_display)
        freq_row.addStretch()
        layout.addLayout(freq_row)
        
        # VAD Status with color indicator
        vad_row = QHBoxLayout()
        vad_label = QLabel("Voice:")
        vad_label.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        self.compact_vad_indicator = QLabel("● STATIC")
        self.compact_vad_indicator.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff4444;")
        vad_row.addWidget(vad_label)
        vad_row.addWidget(self.compact_vad_indicator)
        vad_row.addStretch()
        layout.addLayout(vad_row)
        
        # Status text
        self.compact_status = QLabel("Idle...")
        self.compact_status.setStyleSheet("color: #aaa; font-size: 9px;")
        layout.addWidget(self.compact_status)
        
        # Control buttons - only Expand in compact mode
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.compact_expand_btn = QPushButton("↑ Expand")
        self.compact_expand_btn.setMaximumWidth(80)
        self.compact_expand_btn.clicked.connect(self._switch_to_normal)
        button_layout.addWidget(self.compact_expand_btn)
        
        layout.addLayout(button_layout)
        
        return compact

    def _switch_to_compact(self, remember_size=False):
        """Switch to compact monitoring mode"""
        if self._compact_mode:
            return
        
        try:
            # Store normal size before switching
            if remember_size:
                self._last_compact_size = (self.width(), self.height())
            
            self._compact_mode = True
            
            # Switch using stacked widget
            self.stacked_widget.setCurrentIndex(1)
            self.update()
            
            # Set compact window size with bounds check
            w = max(350, min(600, self._last_compact_size[0]))
            h = max(150, min(400, self._last_compact_size[1]))
            self.resize(w, h)
        except Exception as e:
            print(f"[ERROR] _switch_to_compact: {e}")

    def _switch_to_normal(self):
        """Switch back to normal full UI mode"""
        if not self._compact_mode:
            return
        
        try:
            self._compact_mode = False
            
            # Switch using stacked widget
            self.stacked_widget.setCurrentIndex(0)
            self.update()
            
            # Restore normal window size with safe bounds
            w = max(700, min(1400, self._normal_size[0]))
            h = max(620, min(1000, self._normal_size[1]))
            self.resize(w, h)
        except Exception as e:
            print(f"[ERROR] _switch_to_normal: {e}")

    def _update_compact_display(self):
        """Update compact widget with current status"""
        if self._compact_widget is None:
            return
        
        # Always update labels (they'll be visible when in compact mode)
        # Convert frequency to proper format if it's a string representation of a float
        try:
            freq_val = float(self._frequency) if self._frequency else None
            freq_display = f"{freq_val:.0f}" if freq_val else "None"
        except (ValueError, TypeError):
            freq_display = str(self._frequency) if self._frequency else "None"
        
        self.compact_freq_display.setText(freq_display)
        
        # Update VAD indicator based on voice_label
        if hasattr(self, 'voice_label'):
            is_voice = "voice" in self.voice_label.text().lower()
            if is_voice:
                self.compact_vad_indicator.setText("● VOICE")
                self.compact_vad_indicator.setStyleSheet("font-size: 14px; font-weight: bold; color: #00dd00;")
            else:
                self.compact_vad_indicator.setText("● STATIC")
                self.compact_vad_indicator.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff4444;")
        
        # Update status from status bar
        if self.status_bar:
            status_text = self.status_bar.currentMessage()
            # Truncate long status
            if len(status_text) > 45:
                status_text = status_text[:42] + "..."
            self.compact_status.setText(status_text)

    def resizeEvent(self, event):
        """Detect window resize and switch to compact mode if needed"""
        super().resizeEvent(event)
        
        # Only process if we've initialized
        if not hasattr(self, '_compact_mode'):
            return
        
        current_width = self.width()
        
        # Threshold: switch to compact if width < 600
        if not self._compact_mode and current_width < 600:
            self._switch_to_compact()
        # Switch back to normal if width >= 680 (hysteresis to avoid flickering)
        elif self._compact_mode and current_width >= 680:
            self._switch_to_normal()

    def _init_timer(self):
        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(50)
        self._meter_timer.timeout.connect(self._update_meter)
        
        # Periodic calibration timer (every 5 minutes)
        self._calibration_timer = QTimer(self)
        self._calibration_timer.setInterval(self._calibration_interval_seconds * 1000)
        self._calibration_timer.timeout.connect(self._on_periodic_calibration)

    def _populate_devices(self):
        try:
            devices = sd.query_devices()
            self.device_combo.clear()
            seen_names = set()  # Track unique device names
            
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    device_name = dev['name']
                    
                    # Skip duplicates
                    if device_name in seen_names:
                        continue
                    seen_names.add(device_name)
                    
                    # Add icon based on device type
                    if 'loopback' in device_name.lower() or 'stereo mix' in device_name.lower():
                        icon = "🔊"  # Speaker for loopback/stereo mix
                    elif 'speaker' in device_name.lower() or 'output' in device_name.lower():
                        icon = "🔊"
                    else:
                        icon = "🎤"  # Microphone for input devices
                    
                    display_name = f"{icon} {device_name} (ID {idx})"
                    self.device_combo.addItem(display_name, idx)
        except Exception as e:
            self._signals.error.emit(f"Device query failed: {e}")

    def _on_device_changed(self, index):
        if self._running:
            self._on_stop()

    def _on_freq_selected(self, freq):
        self._frequency = str(freq)
        self.freq_display.setText(f"{freq:.0f} kHz")
        self._update_compact_display()
        self._save_config()  # Auto-save frequency

    def _on_sensitivity_changed(self, value):
        self._sensitivity_level = value
        self.sens_label.setText(SENSITIVITY_LABELS[value])

    def _open_webhook_dialog(self):
        dlg = WebhookManagerDialog(self._webhooks, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhooks = dlg.get_webhooks()
            # Update display
            enabled_count = sum(1 for w in self._webhooks if w.get("enabled", True))
            total_count = len(self._webhooks)
            if total_count > 0:
                self.webhook_edit.setText(f"✓ {enabled_count}/{total_count} webhooks active")
            else:
                self.webhook_edit.setText("No webhooks configured")
            self.webhook_edit.setReadOnly(True)
            self._save_config()  # Auto-save webhooks

    def _open_customize_dialog(self):
        dlg = WebhookCustomizeDialog(self._webhook_message, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhook_message = dlg.get_message_template()
            self.message_edit.setText(self._webhook_message[:40] + "..." if len(self._webhook_message) > 40 else self._webhook_message)
            self._save_config()  # Auto-save customize

    def _open_audio_settings(self):
        """Open audio processing settings dialog."""
        dlg = AudioSettingsDialog(self._audio_settings, self)
        if dlg.exec_() == QDialog.Accepted:
            self._audio_settings = dlg.get_settings()

    def _on_file_location(self):
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory", self._save_dir)
        if path:
            self._save_dir = path

    def _on_start(self):
        if not self._frequency:
            QMessageBox.warning(self, "Warning", "Please select a frequency first.")
            return
        if not self._webhooks:
            QMessageBox.warning(self, "Warning", "Please configure Discord webhooks first.")
            return
        self._running = True
        self._learning_phase = "startup"  # Start with initial learning
        self._last_learning_time = time.time()
        self._last_relearn_time = time.time()
        self._noise_samples.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._meter_timer.start()
        self._calibration_timer.start()  # Start periodic calibration
        self._start_audio_stream()

    def _on_stop(self):
        self._running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._meter_timer.stop()
        self._calibration_timer.stop()  # Stop periodic calibration
        self._stop_audio_stream()

    def _on_calibrate_noise(self):
        """Record 10 seconds of ambient noise, analyze it, and auto-adjust VAD thresholds."""
        # CRITICAL: Stop any active recording and reset VAD state
        if self._recording:
            self._finalize_recording()  # Force stop any ongoing recording
        
        # Reset all VAD state to clean state
        self._hangover_remaining = 0.0
        self._voice_confidence_duration = 0.0
        self._last_high_confidence_time = None
        self._voice_detected = False
        self._low_confidence_frames = 0
        self._post_roll_silence_frames = 0
        
        # Clear hangover tracking flags
        if hasattr(self, '_hangover_started'):
            delattr(self, '_hangover_started')
        
        # Clear ring buffer to ensure clean 10-second calibration sample
        with self._lock:
            self._ring_buffer.clear()
        
        msg = QMessageBox()
        msg.setWindowTitle("Calibrate Noise Floor")
        msg.setText("Recording 10 seconds of ambient noise...\n\nMake sure the radio is NOT transmitting voice.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIcon(QMessageBox.Information)
        
        # Show dialog but don't block
        self.calibrate_btn.setEnabled(False)
        self._calibration_samples = []
        self._calibration_start = time.time()
        self._calibration_active = True
        self.status_bar.showMessage("🎤 CALIBRATING... Do not speak (10s)")
        
        # Turn button red to indicate active calibration
        self.calibrate_btn.setStyleSheet(
            "QPushButton { background-color: #FF4444; color: white; font-weight: bold; }"
        )
        
        # After 10 seconds, analyze and adjust
        QTimer.singleShot(10000, self._analyze_calibration)

    def _analyze_calibration(self):
        """Analyze recorded noise and adjust VAD thresholds."""
        self._calibration_active = False
        
        if len(self._calibration_samples) < 100:
            self.status_bar.showMessage("Calibration failed: insufficient audio samples")
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setStyleSheet("")  # Reset button style
            return
        
        try:
            # Concatenate all samples
            audio_data = np.concatenate(self._calibration_samples)
            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            
            # Analyze noise properties — update the ACTUAL noise floor used by VAD v1.19
            rms_noise = np.sqrt(np.mean(audio_data ** 2))
            old_rms = getattr(self, '_noise_floor_rms', 0.001)
            old_db = self._noise_floor_db
            
            # Update noise floor (the values _detect_voice() actually uses)
            self._noise_floor_rms = rms_noise * 0.9  # 90% of measured noise
            self._noise_floor_db = 20 * np.log10(self._noise_floor_rms + 1e-10)
            
            # Check modulation (CV) of noise for diagnostics
            chunk_size = len(audio_data) // 10
            rms_values = []
            for i in range(10):
                start = i * chunk_size
                end = start + chunk_size if i < 9 else len(audio_data)
                chunk_rms = np.sqrt(np.mean(audio_data[start:end] ** 2))
                rms_values.append(chunk_rms)
            
            rms_mean = np.mean(rms_values)
            if rms_mean > 1e-10:
                cv_noise = np.std(rms_values) / rms_mean
            else:
                cv_noise = 0.0
            
            # Log the changes
            msg = f"📊 Calibration Complete:\n\n"
            msg += f"Noise RMS: {rms_noise:.6f}\n"
            msg += f"Noise Floor: {old_db:.1f} → {self._noise_floor_db:.1f} dB\n"
            msg += f"Noise CV: {cv_noise:.2f}\n"
            
            self.status_bar.showMessage(f"✓ Calibrated: Noise floor = {self._noise_floor_db:.1f} dB")
            
            QMessageBox.information(self, "Calibration Complete", msg)
            
        except Exception as e:
            self.status_bar.showMessage(f"Calibration error: {str(e)[:50]}")
            QMessageBox.warning(self, "Calibration Error", f"Failed to analyze noise: {str(e)[:100]}")
        finally:
            # Reset VAD state after calibration completes
            self._hangover_remaining = 0.0
            self._voice_confidence_duration = 0.0
            self._last_high_confidence_time = None
            self._voice_detected = False
            self._low_confidence_frames = 0
            self._post_roll_silence_frames = 0
            
            # Clear hangover tracking flags
            if hasattr(self, '_hangover_started'):
                delattr(self, '_hangover_started')
            
            # Clear ring buffer
            with self._lock:
                self._ring_buffer.clear()
            
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setStyleSheet("")  # Reset button style to normal
            self._calibration_samples = []

    def _on_periodic_calibration(self):
        """Automatic calibration every 5 minutes if no voice detected recently."""
        if not self._periodic_calibration_enabled or self._running is False:
            return
        
        # Check if voice was detected in the past 5 minutes
        now = time.time()
        if self._last_confirmed_voice_time is None:
            # No voice ever detected, safe to calibrate
            time_since_voice = float('inf')
        else:
            time_since_voice = now - self._last_confirmed_voice_time
        
        # Only auto-calibrate if no voice in past 5 minutes AND not currently recording
        if time_since_voice >= self._calibration_interval_seconds and not self._recording:
            self._run_auto_calibration()
    
    def _run_auto_calibration(self):
        """Run automatic calibration silently without user interaction."""
        # Clear ring buffer for clean sample
        with self._lock:
            self._ring_buffer.clear()
        
        self._calibration_samples = []
        self._calibration_start = time.time()
        self._calibration_active = True
        
        self.status_bar.showMessage("🎤 Auto-calibrating noise floor (10s)...")
        
        # After 10 seconds, analyze
        QTimer.singleShot(10000, self._analyze_auto_calibration)
    
    def _analyze_auto_calibration(self):
        """Analyze auto-calibration samples and adjust thresholds silently."""
        self._calibration_active = False
        
        if len(self._calibration_samples) < 100:
            # Not enough samples, skip
            self.status_bar.showMessage("Auto-calibration: insufficient samples, skipped")
            return
        
        try:
            # Concatenate all samples
            audio_data = np.concatenate(self._calibration_samples)
            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            
            # Update noise floor — the values _detect_voice() actually uses
            rms_noise = np.sqrt(np.mean(audio_data ** 2))
            old_db = self._noise_floor_db
            
            self._noise_floor_rms = rms_noise * 0.9
            self._noise_floor_db = 20 * np.log10(self._noise_floor_rms + 1e-10)
            
            # Log silently to status bar
            self.status_bar.showMessage(
                f"Auto-calibrated: Noise floor {old_db:.1f} → {self._noise_floor_db:.1f} dB"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Auto-calibration error: {str(e)[:40]}")
        finally:
            # Reset VAD state
            self._hangover_remaining = 0.0
            self._voice_confidence_duration = 0.0
            self._last_high_confidence_time = None
            self._voice_detected = False
            if hasattr(self, '_low_confidence_frames'):
                self._low_confidence_frames = 0
            if hasattr(self, '_post_roll_silence_frames'):
                self._post_roll_silence_frames = 0
            if hasattr(self, '_hangover_started'):
                delattr(self, '_hangover_started')
            
            # Clear ring buffer
            with self._lock:
                self._ring_buffer.clear()

    def _start_audio_stream(self):
        try:
            device_data = self.device_combo.currentData()
            device_idx = device_data if device_data is not None else None
            device_info = sd.query_devices(device_idx)
            self._sample_rate = int(device_info['default_samplerate'])
            # Recreate ring buffer with correct size for this device's sample rate
            with self._lock:
                self._ring_buffer = collections.deque(
                    maxlen=int(PRE_ROLL_SECONDS * self._sample_rate / BLOCK_SIZE)
                )
            self._stream = sd.InputStream(
                device=device_idx, channels=CHANNELS, samplerate=self._sample_rate,
                blocksize=BLOCK_SIZE, callback=self._audio_callback
            )
            self._stream.start()
            self._signals.status.emit(f"Listening on {device_info['name']} @ {self._sample_rate} Hz")
        except Exception as e:
            self._signals.error.emit(f"Stream start failed: {e}")
            self._running = False

    def _stop_audio_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._signals.error.emit(f"Stream stop failed: {e}")
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            return
        # Minimal lock time: only update ring buffer and peak dB
        audio_chunk = indata[:, 0].copy()
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        self._peak_db = 20 * np.log10(rms + 1e-10)
        
        with self._lock:
            self._ring_buffer.append(audio_chunk)
        
        # Capture calibration samples if calibration active
        if self._calibration_active:
            self._calibration_samples.append(audio_chunk)
        
        # Buffer recording chunk outside main lock
        if self._recording:
            with self._audio_buffer_lock:
                self._audio_buffer.append(audio_chunk)

    def _detect_voice(self, audio: np.ndarray) -> float:
        """v1.25 Speech Formant Primary Detection.
        
        Validated on real FlexRadio IQ data - speech formants are the discriminator:
        1. Speech Formants (40pts) - PRIMARY: 300-4000 Hz spectral peaks
        2. SNR Gate (15pts) - Foundation threshold
        3. Voice Band Organization (20pts) - Spectral structure 300-3000 Hz
        4. Pitch Detection (15pts) - Fundamental frequency 85-250 Hz
        5. Modulation (10pts) - Amplitude dynamics
        
        Key insight: Noise can have high dynamic range (11.68 dB), but speech has
        characteristic formant peaks (F1:200-900Hz, F2:700-2200Hz, F3:1500-3500Hz)
        that are absent in noise. Voice2 capture showed 3.5kHz formants.
        
        Returns: 0-100 confidence where 55+ = voice (see sensitivity preset thresholds)
        """
        if len(audio) < 512:
            return 0.0
        
        try:
            debug = self._meter_debug
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # ===== INITIALIZATION =====
            if not hasattr(self, '_noise_floor_rms'):
                self._noise_floor_rms = 0.001
                self._snr_history = collections.deque(maxlen=300)
                self._hangover_remaining = 0.0
            if not hasattr(self, '_formant_buffer'):
                # Accumulate last 4 blocks (~185ms @ 2048/44100Hz) for formant analysis.
                # Longer window averages out noise variance (std 3.9 dB) while
                # preserving stable voice formants that persist across frames.
                self._formant_buffer = collections.deque(maxlen=4)
            
            # ===== LEARNING PHASE =====
            if self._learning_phase == "startup":
                now = time.time()
                if now - self._last_learning_time < self._noise_learning_time:
                    rms = np.sqrt(np.mean(audio ** 2))
                    # Accumulate samples for averaging (not just overwrite)
                    if not hasattr(self, '_learning_rms_samples'):
                        self._learning_rms_samples = []
                    if rms < 0.1:  # Accept any reasonable signal (radio noise floor can be > 0.01)
                        self._learning_rms_samples.append(rms)
                    return 0.0
                else:
                    # Average all collected samples for a robust noise floor
                    if hasattr(self, '_learning_rms_samples') and self._learning_rms_samples:
                        n_samples = len(self._learning_rms_samples)
                        self._noise_floor_rms = np.median(self._learning_rms_samples) * 0.9
                        del self._learning_rms_samples
                    else:
                        n_samples = 0
                    self._learning_phase = "periodic"
                    self._noise_floor_db = 20 * np.log10(self._noise_floor_rms + 1e-10)
                    print(f"✓ LEARNING COMPLETE: Noise floor = {self._noise_floor_db:.1f} dB (from {n_samples} samples)")
                    self._last_relearn_time = now
                    return 0.0
            
            # ===== PERIODIC NOISE FLOOR UPDATE =====
            elif self._learning_phase == "periodic" and not self._recording:
                now = time.time()
                rms = np.sqrt(np.mean(audio ** 2))
                if now - self._last_relearn_time > self._noise_relearn_interval and rms < 0.005:
                    self._noise_floor_rms = self._noise_floor_rms * 0.99 + rms * 0.01
                    self._noise_floor_db = 20 * np.log10(self._noise_floor_rms + 1e-10)
                    self._last_relearn_time = now
            
            # ===== 5-COMPONENT FORMANT-PRIMARY DETECTION =====
            confidence = 0.0
            rms = np.sqrt(np.mean(audio ** 2))
            snr_db = 20 * np.log10((rms + 1e-10) / (self._noise_floor_rms + 1e-10))
            self._snr_history.append(snr_db)
            self._last_snr_db = snr_db
            
            # Use recent SNR (last 5 frames ~250ms) not 15-second floor
            recent_frames = 5
            if len(self._snr_history) >= recent_frames:
                recent_snr = list(self._snr_history)[-recent_frames:]
                snr_percentile = np.median(recent_snr)
            else:
                snr_percentile = snr_db
            
            # ===== COMPONENT 1: SNR GATE (15 points) =====
            if self._recording:
                snr_threshold = -2.0  # Very lenient during recording
            elif self._hangover_remaining > 0:
                snr_threshold = 0.0  # Lenient during hangover
            else:
                snr_threshold = 0.0  # Weak radio: don't gate on SNR, just score it (0dB threshold)
            
            # SNR scoring: 0pts at threshold, 15pts at threshold+10dB
            if snr_percentile < snr_threshold:
                snr_score = 0.0
            elif snr_percentile > snr_threshold + 10.0:
                snr_score = 15.0
            else:
                # Linear: 0-15pts between threshold and threshold+10dB
                snr_score = ((snr_percentile - snr_threshold) / 10.0) * 15.0
            
            confidence += snr_score
            
            if debug:
                print(f"  SNR Gate: +{snr_score:.0f}pts (SNR {snr_percentile:.1f}dB)")
            
            # ===== COMPONENT 2: SPEECH FORMANTS (40 points) - PRIMARY =====
            # Accumulate into rolling buffer (~185ms at 48kHz) before scoring.
            self._formant_buffer.append(audio)
            formant_audio = np.concatenate(list(self._formant_buffer))
            formant_score, formant_count = self._score_formants(formant_audio)
            
            if debug:
                print(f"  Formants: +{formant_score:.0f}pts ({formant_count} clusters)")
            
            # ===== COMPONENT 3: VOICE BAND ORGANIZATION (20 points) =====
            voiceband_score = self._score_voice_band(audio)
            
            # --- FLAT-SPECTRUM FORMANT PENALTY ---
            if voiceband_score == 0.0:
                preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
                flat_penalty = preset.get("flat_penalty_factor", 0.40)
                formant_score *= flat_penalty
            
            # ===== v1.26 SNR GATE ON SPECTRAL SCORES =====
            # Real-data finding (14405 kHz S8/S9 captures): HF voice sits only 0-4 dB
            # above the noise floor. The old multiplicative gate (ramp 0→1 over 0-5 dB)
            # was killing 96.9% of ALL spectral scores on real S8/S9 voice — making
            # formant and voiceband detection completely useless.
            #
            # Fix: use a very soft gate. Only block spectral scores when SNR is deeply
            # negative (< -3 dB = signal clearly below noise floor). Above -3 dB, pass
            # through with a mild scale factor. The formant/voiceband structure itself
            # provides the voice vs. noise discrimination — the gate just prevents
            # crediting structure that can't possibly exist in pure silence.
            #
            # Sensitivity-aware floor:
            #   L1: -4 dB  L2: -3 dB  L3: -2 dB  L4: -1 dB  L5: 0 dB
            snr_floors = {1: -4.0, 2: -3.0, 3: -2.0, 4: -1.0, 5: 0.0}
            snr_floor = snr_floors.get(self._sensitivity_level, -2.0)
            # Ramp over 6 dB above the floor — soft gate, not a cliff
            if snr_percentile <= snr_floor:
                snr_spectral_gate = 0.0
            elif snr_percentile >= snr_floor + 6.0:
                snr_spectral_gate = 1.0
            else:
                snr_spectral_gate = (snr_percentile - snr_floor) / 6.0
            
            formant_score   *= snr_spectral_gate
            voiceband_score *= snr_spectral_gate
            
            confidence += formant_score
            confidence += voiceband_score
            
            if debug:
                print(f"  Voice Band: +{voiceband_score:.0f}pts  SNR gate: {snr_spectral_gate:.2f} (SNR={snr_percentile:.1f}dB floor={snr_floor:.0f}dB)")
            
            # ===== COMPONENT 4: PITCH DETECTION (15 points) =====
            pitch_score = self._detect_pitch(audio) if confidence > 10 else 0.0
            # Rescale pitch output from 0-35 to 0-15
            pitch_score = (pitch_score / 35.0) * 15.0
            confidence += pitch_score
            self._last_pitch_score = pitch_score
            
            if debug:
                print(f"  Pitch: +{pitch_score:.0f}pts")
            
            # ===== COMPONENT 5: MODULATION (10 points) =====
            if confidence > 15:
                chunk_size = len(audio) // 10
                if chunk_size > 50:
                    rms_values = []
                    for i in range(10):
                        start = i * chunk_size
                        end = start + chunk_size if i < 9 else len(audio)
                        chunk = audio[start:end]
                        chunk_rms = np.sqrt(np.mean(chunk ** 2))
                        rms_values.append(chunk_rms)
                    
                    rms_array = np.array(rms_values, dtype=np.float32)
                    rms_mean = np.mean(rms_array)
                    
                    if rms_mean > 1e-10:
                        cv = np.std(rms_array) / rms_mean
                    else:
                        cv = 0.0
                    
                    # Score modulation: CV in speech syllable range (0.15-0.50) gets bonus.
                    # v1.21: Cap at CV=0.50 — very high CV (>0.60) is noise bursts, not speech.
                    # Speech syllables: CV 0.15-0.50. Broadband noise bursts: CV 0.60-1.5+
                    if cv < 0.05:
                        mod_score = 0.0
                    elif cv < 0.15:
                        mod_score = (cv / 0.15) * 4.0
                    elif cv < 0.50:
                        mod_score = 4.0 + ((cv - 0.15) / 0.35) * 6.0  # Ramps 4→10 pts
                    elif cv < 0.70:
                        mod_score = 10.0 - ((cv - 0.50) / 0.20) * 5.0  # Ramps 10→5 pts
                    else:
                        mod_score = 5.0 - min(5.0, ((cv - 0.70) / 0.30) * 5.0)  # Ramps 5→0
                    
                    confidence += mod_score
                    if debug:
                        print(f"  Modulation: +{mod_score:.0f}pts (CV={cv:.3f})")
            
            if debug:
                print(f"[VOICE v1.25] SNR={snr_percentile:.1f}dB | Clusters={formant_count} | gate={snr_spectral_gate:.2f} | Confidence={confidence:.0f}/100")
            
            return np.clip(confidence, 0.0, 100.0)
            
        except Exception as e:
            print(f"[VAD ERROR] {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _detect_pitch(self, audio: np.ndarray) -> float:
        """Detect pitch/periodicity in voice range 85-250Hz.
        
        Voice has fundamental frequency. Static bursts have no coherent pitch.
        Weight raised to 35pts - strongest discriminator for SSB radio.
        Silence autocorrelation peak: 0.10-0.20. Voice: 0.22+ (medium) to 0.87 (strong).
        Returns: 0-35 points (0 = no pitch, 35 = strong pitch in voice range)
        """
        try:
            if len(audio) < 512:
                return 0.0
            
            # Compute autocorrelation to find pitch
            # Voice pitch is typically 85 Hz (male) to 250 Hz (female/child)
            # At 44100 Hz sampling: 85 Hz = ~519 samples, 250 Hz = ~176 samples
            
            audio_work = audio - np.mean(audio)  # Remove DC
            if np.max(np.abs(audio_work)) < 1e-10:
                return 0.0
            
            # Autocorrelation using FFT (fast)
            fft = np.fft.fft(audio_work, n=2*len(audio_work))
            power = fft * np.conj(fft)
            autocorr = np.fft.ifft(power)[0:len(audio_work)]
            autocorr = np.real(autocorr)
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for peaks in pitch range
            # Min lag: 44100 Hz / 250 Hz = 176 samples
            # Max lag: 44100 Hz / 85 Hz = 519 samples
            min_lag = max(10, int(self._sample_rate / 250))  # ~176 @ 44.1kHz
            max_lag = min(len(autocorr)-1, int(self._sample_rate / 85))  # ~519 @ 44.1kHz
            
            if max_lag <= min_lag:
                return 0.0
            
            autocorr_pitch = autocorr[min_lag:max_lag]
            
            # Find strongest peak in pitch range
            peak_idx = np.argmax(autocorr_pitch) if len(autocorr_pitch) > 0 else 0
            peak_strength = autocorr_pitch[peak_idx] if len(autocorr_pitch) > 0 else 0.0
            
            # Calibrated from real HF SSB captures (14405 kHz, FlexRadio DAX):
            # SSB compresses voice — autocorr max observed was 0.298 on S8/S9 voice.
            # Old threshold (>=0.45 for full score) was NEVER reachable on SSB.
            # New thresholds: <=0.08 = 0pts (silence), >=0.25 = 35pts (SSB voice)
            # Linear ramp 0.08→0.25. Anything above 0.25 is strong voice.
            if peak_strength >= 0.25:
                return 35.0
            elif peak_strength <= 0.08:
                return 0.0
            else:
                return (peak_strength - 0.08) / 0.17 * 35.0
        except:
            return 0.0  # No credit on error — silence is safer than false positive
    
    
    def _score_formants(self, audio: np.ndarray, max_points: float = 40.0) -> tuple:
        """Detect speech formants in 300-4000 Hz band.
        
        Speech formants (F1-F4) are acoustic resonances visible as peaks:
        F1: 200-900 Hz     (back of tongue position)
        F2: 700-2200 Hz    (front of tongue position)
        F3: 1500-3500 Hz   (lip rounding)
        F4: 2400-4000 Hz   (rare, upper frequencies)
        
        v1.24 changes:
        - Receives a longer audio window (up to 4 blocks ~185ms) for reliable peaks
        - In-band flatness pre-gate: if the 300-4000 Hz region itself is flat
          (flatness > 0.72), return 0 immediately — real voice always has structure
        - Cluster tightness-aware scoring for single-cluster weak voice
        
        Returns: (score_0_to_40, cluster_count)
        """
        try:
            if len(audio) < 512:
                return 0.0, 0
            
            # --- IN-BAND FLATNESS PRE-GATE ---
            # Compute spectral flatness of ONLY the 300-4000 Hz band.
            # Noise in this band is still broadband-flat (flatness 0.75-0.90).
            # Real voice has prominent formant peaks (flatness 0.15-0.55).
            # Gate: if band flatness > 0.72, this is almost certainly noise.
            # (Sensitivity-aware: level 1 allows up to 0.80 for very weak voice)
            preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
            flatness_gates = {1: 0.80, 2: 0.76, 3: 0.72, 4: 0.68, 5: 0.62}
            flatness_gate = flatness_gates.get(self._sensitivity_level, 0.72)
            
            try:
                sos_bp = butter(4, [300, 4000], btype='band', fs=self._sample_rate, output='sos')
                band_audio = sosfiltfilt(sos_bp, audio)
                _, pxx = welch(band_audio, fs=self._sample_rate, nperseg=min(512, len(band_audio)))
                pxx = np.maximum(pxx, 1e-12)
                in_band_flatness = np.exp(np.mean(np.log(pxx))) / np.mean(pxx)
                if in_band_flatness > flatness_gate:
                    return 0.0, 0  # Band is flat — noise, not voice
            except:
                pass  # If band flatness check fails, continue with peak analysis
            
            # FFT in formant region
            window = np.hanning(len(audio))
            fft_result = np.fft.rfft(audio * window)
            freqs = np.fft.rfftfreq(len(audio), 1/self._sample_rate)
            magnitude = np.abs(fft_result)
            magnitude_db = 20 * np.log10(magnitude + 1e-9)
            
            # Extract 300-4000 Hz band
            formant_mask = (freqs > 300) & (freqs < 4000)
            formant_freqs = freqs[formant_mask]
            formant_mag_db = magnitude_db[formant_mask]
            
            if len(formant_mag_db) == 0:
                return 0.0, 0
            
            # Find peaks - raised prominence to be sensitivity-dependent (v1.22)
            # Level 1-2: lower prominence to catch weak voice formants in noise
            # Level 4-5: higher prominence to tightly reject noise bumps
            preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
            threshold_db = preset.get("formant_threshold_db", 5)
            prominence_db = preset.get("formant_prominence_db", 6.0)
            formant_floor = np.median(formant_mag_db) + threshold_db
            
            peaks, properties = find_peaks(
                formant_mag_db,
                height=formant_floor,
                distance=max(1, int(150 / (self._sample_rate / len(audio)))),  # Min 150 Hz apart
                prominence=prominence_db  # Sensitivity-driven: 3.5 (L1) to 9.0 (L5)
            )
            
            if len(peaks) == 0:
                return 0.0, 0
            
            # --- CLUSTER DEDUPLICATION ---
            # Merge peaks within 300 Hz of each other into one cluster.
            # Speech formants are separated by 700-1500 Hz; peaks closer than 300 Hz
            # are just the same formant resonance and should count as one.
            peak_freqs = formant_freqs[peaks]
            peak_freqs_sorted = np.sort(peak_freqs)
            
            clusters = []
            current_cluster = [peak_freqs_sorted[0]]
            for freq in peak_freqs_sorted[1:]:
                if freq - current_cluster[-1] < 300.0:  # Same cluster
                    current_cluster.append(freq)
                else:  # New cluster
                    clusters.append(current_cluster)
                    current_cluster = [freq]
            clusters.append(current_cluster)
            
            cluster_count = len(clusters)
            
            # --- FORMANT SPREAD GATE ---
            # For 2+ clusters, require minimum total span of 400 Hz.
            # Real speech F1-F2 gap is always > 400 Hz; noise clusters that survive
            # deduplication but are still close together get merged back to 1.
            if cluster_count >= 2:
                cluster_centers = [np.mean(c) for c in clusters]
                total_span = max(cluster_centers) - min(cluster_centers)
                if total_span < 400.0:
                    # Too close — collapse to single cluster
                    cluster_count = 1
            
            # --- SINGLE-CLUSTER TIGHTNESS SCORING ---
            # Key insight from real FlexRadio IQ captures:
            #   Extremely weak voice: all peaks within 129 Hz (tight single formant)
            #   Strong voice 2:       all peaks within 217 Hz
            #   Pure noise:           peaks scattered over 17870 Hz
            # A single tight cluster is a REAL formant, not noise.
            # Score based on bandwidth of the tightest cluster.
            if cluster_count == 1:
                cluster_bw = max(clusters[0]) - min(clusters[0]) if len(clusters[0]) > 1 else 0.0
                if cluster_bw <= 150.0:
                    # Extremely tight — characteristic of weak SSB voice (e.g. 129 Hz)
                    score = 22.0
                elif cluster_bw <= 250.0:
                    # Tight — typical of single formant voice (e.g. 217 Hz)
                    score = 17.0
                else:
                    # Loose single cluster — marginal
                    score = 10.0
            elif cluster_count == 2:
                score = 22.0
            elif cluster_count == 3:
                score = 36.0
            elif cluster_count >= 4:
                score = max_points
            else:
                score = 0.0
            
            return score, cluster_count
        
        except:
            return 0.0, 0
    
    def _score_voice_band(self, audio: np.ndarray, max_points: float = 20.0) -> float:
        """Score spectral organization in 300-3000 Hz voice band.
        
        Voice band should have organized structure (formants, harmonics).
        Noise has flat/uniform spectrum in this band.
        
        Scoring based on spectral flatness:
        Voice: < 0.4 flatness = organized
        Noise: > 0.6 flatness = flat/unstructured
        """
        try:
            # Bandpass 300-3000 Hz
            sos = butter(4, [300, 3000], btype='band', fs=self._sample_rate, output='sos')
            filtered = sosfiltfilt(sos, audio)
            
            # Spectral flatness
            freqs, pxx = welch(filtered, fs=self._sample_rate, nperseg=min(512, len(filtered)))
            pxx = np.maximum(pxx, 1e-12)
            
            geom_mean = np.exp(np.mean(np.log(pxx)))
            arith_mean = np.mean(pxx)
            
            if arith_mean > 0:
                flatness = geom_mean / arith_mean
            else:
                return 5.0
            
            # Scoring: structured (low flatness) to flat (high flatness)
            # Tightened thresholds: SSB radio noise is also bandpass-shaped (not white)
            # so noise flatness can be 0.3-0.5, voice is typically 0.1-0.3
            if flatness > 0.50:
                return 0.0  # Flat = noise
            elif flatness < 0.25:
                return max_points  # Very structured = voice
            else:
                # Linear between 0.25-0.50
                return (0.50 - flatness) / 0.25 * max_points
        
        except:
            return 0.0  # Defensive: no credit on error

    # ============================================================================
    # Audio Processing Methods
    # ============================================================================
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio processing filters based on user settings."""
        processed = audio.copy()
        
        # 0. Spectral gate (aggressive out-of-band noise removal)
        # Applied first as a surgical filter before other processing
        processed = self._apply_spectral_gate(processed)
        
        # 1. Bandpass filter (300-3000 Hz SSB speech)
        if self._audio_settings.get("use_bandpass", False):
            processed = self._apply_bandpass_filter(processed)
        
        # 2. Enhanced denoising
        if self._audio_settings.get("use_denoise", True):
            strength = self._audio_settings.get("denoise_strength", 6)
            processed = self._apply_enhanced_denoise(processed, strength)
        
        # 3. Voice extraction (keep only clear voice segments)
        if self._audio_settings.get("voice_extract", False):
            processed = self._extract_voice_only(processed)
        
        # 4. Silence removal (optional)
        if self._audio_settings.get("remove_silence", False):
            processed = self._remove_silence_gaps(processed)
        
        return processed
    
    def _apply_spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply aggressive spectral gating to suppress out-of-band noise.
        
        Uses FFT to identify and suppress frequencies outside voice band (300-3000 Hz).
        Preserves voice formants while suppressing static and HF noise.
        """
        try:
            # Process in overlapping frames for smoothness
            frame_size = 2048
            overlap = frame_size // 2
            hop_size = frame_size - overlap
            
            if len(audio) < frame_size:
                return audio
            
            output = np.zeros_like(audio)
            window = np.hamming(frame_size)
            
            # Frequency bands for gating
            # Voice: 300-3000 Hz (strong gate)
            # Sub-band: 100-300 Hz (medium gate, allows bass)
            # Super-band: 3000-5000 Hz (light gate, allows harmonics)
            
            for start in range(0, len(audio) - frame_size, hop_size):
                end = start + frame_size
                frame = audio[start:end] * window
                
                # FFT
                spectrum = np.fft.rfft(frame)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                
                # Create frequency array
                freqs = np.fft.rfftfreq(frame_size, 1.0 / self._sample_rate)
                
                # Gating envelope (soft knees for smoothness)
                gate = np.ones_like(freqs)
                
                # Sub-voice band (0-100 Hz): reduce by 70%
                sub_band = freqs < 100
                gate[sub_band] = 0.3
                
                # Super-voice band (5000+ Hz): reduce by 80%
                super_band = freqs > 5000
                gate[super_band] = 0.2
                
                # Very high freq (>7000 Hz): reduce by 95% (hiss/crackle)
                very_high = freqs > 7000
                gate[very_high] = 0.05
                
                # Apply smooth transition at edges of voice band
                # 300-500 Hz: ramp from 30% → 100%
                voice_start_edge = (freqs >= 100) & (freqs < 300)
                gate[voice_start_edge] = 0.3 + (freqs[voice_start_edge] - 100) / 200 * 0.7
                
                # 3000-5000 Hz: ramp from 100% → 20%
                voice_end_edge = (freqs > 3000) & (freqs <= 5000)
                gate[voice_end_edge] = 1.0 - (freqs[voice_end_edge] - 3000) / 2000 * 0.8
                
                # Apply gating
                spectrum_gated = spectrum * gate
                
                # IFFT
                frame_out = np.fft.irfft(spectrum_gated, n=frame_size)
                frame_out = frame_out * window
                
                # Overlap-add
                output[start:end] += frame_out
            
            # Handle final frame if needed
            if end < len(audio):
                remaining = len(audio) - end
                output[end:] = audio[end:] * 0.5  # Reduce tail
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = output / max_val
            
            return np.clip(output, -1.0, 1.0)
        except Exception as e:
            if self._meter_debug:
                print(f"Spectral gate error: {e}")
            return audio
    
    def _apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter 300-3000 Hz for SSB speech using scipy."""
        try:
            # Butterworth bandpass (proven stable design)
            sos = butter(4, [300, 3000], btype='band', fs=self._sample_rate, output='sos')
            filtered = sosfiltfilt(sos, audio)
            return np.clip(filtered, -1.0, 1.0)
        except Exception as e:
            if self._meter_debug:
                print(f"Bandpass filter error: {e}")
            return audio
    
    def _apply_enhanced_denoise(self, audio: np.ndarray, strength: int) -> np.ndarray:
        """Apply enhanced noise reduction using noisereduce library (proven stable).
        
        Strength 1-10 maps to noise reduction aggressiveness:
        1-3: Light (preserve quality)
        4-6: Balanced
        7-10: Aggressive (noise removal)
        """
        if not HAS_NOISEREDUCE:
            return audio
        
        try:
            # Map strength 1-10 to denoising parameter
            prop_decrease = 0.3 + (strength / 10.0) * 0.5  # 0.3 to 0.8
            
            # Use noisereduce with stationary=False for varying noise
            reduced = nr.reduce_noise(
                y=audio,
                sr=self._sample_rate,
                prop_decrease=prop_decrease,
                n_fft=512,
                stationary=False
            )
            
            return np.clip(reduced, -1.0, 1.0)
        except Exception as e:
            if self._meter_debug:
                print(f"Denoise error: {e}")
            return audio
    
    def _extract_voice_only(self, audio: np.ndarray) -> np.ndarray:
        """Extract only clear voice segments using VAD, silence the rest (proven method)."""
        try:
            # Process in chunks
            chunk_size = BLOCK_SIZE
            voice_audio = np.zeros_like(audio)
            
            # Get voice extraction threshold from settings (default 58%)
            threshold = self._audio_settings.get("voice_extract_threshold", 58)
            
            for i in range(0, len(audio), chunk_size):
                end = min(i + chunk_size, len(audio))
                chunk = audio[i:end]
                
                # Use VAD to detect if this chunk is voice
                # Threshold adjustable via Audio Processing Settings
                score = self._detect_voice(chunk)
                if score > threshold:  # User-adjustable threshold for voice extraction
                    voice_audio[i:end] = chunk
            
            return voice_audio
        except Exception as e:
            if self._meter_debug:
                print(f"Voice extraction error: {e}")
            return audio
    
    def _remove_silence_gaps(self, audio: np.ndarray) -> np.ndarray:
        """Remove silent gaps between speech bursts (proven simple method)."""
        try:
            # Detect silence threshold (10% of max amplitude)
            threshold = 0.1 * np.max(np.abs(audio))
            if threshold < 0.001:
                threshold = 0.001
            
            # Find regions above threshold
            above_threshold = np.abs(audio) > threshold
            
            # Keep consecutive samples with minimum 50ms burst length
            chunk_size = int(0.05 * self._sample_rate)
            
            result = np.zeros_like(audio)
            in_speech = False
            speech_start = 0
            
            for i in range(len(above_threshold)):
                if above_threshold[i] and not in_speech:
                    speech_start = i
                    in_speech = True
                elif not above_threshold[i] and in_speech:
                    # End of speech burst
                    if i - speech_start > chunk_size:
                        result[speech_start:i] = audio[speech_start:i]
                    in_speech = False
            
            # Handle final burst
            if in_speech and len(above_threshold) - speech_start > chunk_size:
                result[speech_start:] = audio[speech_start:]
            
            return result
        except Exception as e:
            if self._meter_debug:
                print(f"Silence removal error: {e}")
            return audio

    def _update_meter(self):
        """v1.06: Confidence-based VAD with word-level detection and 10s post-roll.
        
        Detection states:
        - START: Confidence ≥ 65% (default Level 3)
        - CONTINUE: Confidence ≥ 30% OR within 10s of last detected word
        - WORD PEAK: Confidence ≥ 75% (detected word/syllable, resets post-roll timer)
        - STOP: Confidence < 20% AND > 10s since last word peak
        """
        db = self._peak_db
        self.audio_meter.set_level(db)
        self.meter_label.setText(f"{db:+.1f} dB")
        
        # Get current audio chunk for VAD (brief lock)
        current_chunk = None
        with self._lock:
            if self._ring_buffer:
                current_chunk = self._ring_buffer[-1]
        
        if current_chunk is not None:
            # Get confidence score (0-100) instead of boolean
            confidence = self._detect_voice(current_chunk)
            
            # Get thresholds for current sensitivity level
            preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
            start_threshold = preset.get("confidence_start", 65)
            continue_threshold = preset.get("confidence_continue", 30)
            stop_threshold = preset.get("confidence_stop", 20)
            word_peak_threshold = preset.get("word_peak_threshold", 75)
            post_roll_seconds = preset.get("post_roll_seconds", 10)
            
            # Smooth confidence with exponential moving average
            # EMA = (confidence * weight) + (previous_ema * (1 - weight))
            ema_weight = 0.25  # Faster response than binary history
            if not hasattr(self, '_confidence_ema'):
                self._confidence_ema = confidence
            else:
                self._confidence_ema = (confidence * ema_weight) + (self._confidence_ema * (1 - ema_weight))
            
            # Track word peak (syllable detection)
            now = time.time()
            if not hasattr(self, '_last_word_peak_time'):
                self._last_word_peak_time = now - post_roll_seconds  # Start fresh
            
            # WORD PEAK DETECTION: High confidence = detected word/syllable
            # Only reset timer if ABOVE recording threshold (sustained voice, not noise spike)
            if self._recording and confidence >= word_peak_threshold and confidence > 48:
                self._last_word_peak_time = now
                if self._meter_debug:
                    print(f"[Word Peak] SUSTAINED {confidence:.0f}% >= {word_peak_threshold}% → hangover held at {post_roll_seconds}s")

            # Hangover run gate (v1.37): uses SEPARATE hangover_repin_run_chunks, decoupled
            # from confirm_min_run_chunks. Confirm gate is strict (e.g. 6 chunks) to block noise
            # before recording starts. Hangover repin is lenient (3 chunks = ~0.14s) so voice
            # with natural dips/pauses can keep a recording open without needing a sustained
            # 6-chunk run on every word — which caused fragmented/choppy recordings on Flex DAX.
            frame_duration = BLOCK_SIZE / self._sample_rate if hasattr(self, '_sample_rate') else 0.046
            hangover_repin_run = preset.get("hangover_repin_run_chunks", 3)
            if self._recording:
                if confidence >= word_peak_threshold and confidence > 48:
                    self._hangover_voice_run += 1
                else:
                    self._hangover_voice_run = 0

                if self._hangover_voice_run >= hangover_repin_run:
                    # Sustained voice — hold hangover at full post-roll duration
                    self._hangover_remaining = float(post_roll_seconds)
                    if self._meter_debug and self._hangover_voice_run == hangover_repin_run:
                        print(f"[HANGOVER HELD] Sustained voice run={self._hangover_voice_run} chunks (need>={hangover_repin_run}), reset to {post_roll_seconds}s")
                elif self._hangover_remaining > 0:
                    # No sustained voice — count down
                    self._hangover_remaining = max(0.0, self._hangover_remaining - frame_duration)
            
            # DEBUG: Show state every ~0.5s
            if self._meter_debug and not hasattr(self, '_last_debug_print'):
                self._last_debug_print = now - 0.3  # Skip first one
            if self._meter_debug and (now - self._last_debug_print) > 0.5:
                if not self._recording:
                    print(f"[IDLE] Confidence={confidence:.0f}% | SNR_floor={self._noise_floor_db:.1f}dB | Threshold={start_threshold}% (L{self._sensitivity_level}) | Hangover={self._hangover_remaining:.2f}s | Phase={self._learning_phase}")
                self._last_debug_print = now
            
            # ===== RECORDING START/CONTINUE/STOP LOGIC =====
            # v1.12c: Requires 2.0+ seconds of sustained high confidence for faint radio voices
            # (lowered from 3.5s to handle natural speech pauses in SSB radio)
            # Lowered confidence thresholds for weak radio signals
            
            # Use preset thresholds instead of hardcoded values
            if self._recording and self._hangover_remaining > 0:
                # During hangover: use word_peak_threshold to keep hangover alive if voice returns
                threshold_for_accumulate = word_peak_threshold
            elif self._recording:
                # During normal recording: use continue threshold from preset
                threshold_for_accumulate = continue_threshold
            else:
                # Before recording: use start threshold from preset
                threshold_for_accumulate = start_threshold

            # Voice confidence accumulator - tracks total voice time for upload validation
            if confidence > threshold_for_accumulate:
                # Accumulate high-confidence time
                if self._last_high_confidence_time is None:
                    self._last_high_confidence_time = time.time()
                    self._voice_confidence_duration = 0.0
                else:
                    dt = time.time() - self._last_high_confidence_time
                    self._voice_confidence_duration += dt
                    self._last_high_confidence_time = time.time()
                self._low_confidence_frames = 0
            else:
                # Low confidence frame
                self._low_confidence_frames += 1
                # Preserve voice duration — reset only after 5s of continuous idle silence
                if not self._recording and self._last_high_confidence_time is not None:
                    if time.time() - self._last_high_confidence_time > 5.0:
                        self._voice_confidence_duration = 0.0
                        self._last_high_confidence_time = None
            
            # Voice is detected if confidence exceeds threshold OR we're in hangover window
            # Convert to native Python bool to avoid numpy.bool_ type issues
            # v1.27: After hangover expires, use EMA-smoothed confidence for the stop decision.
            # EMA(alpha=0.25) ≈ 168ms time constant prevents single quiet frames from killing a recording,
            # while simultaneously preventing NB-processed noise (EMA ≈41%) from holding recordings
            # open indefinitely when stop_threshold (L3=42%) sits just above the noise EMA.
            if self._hangover_remaining > 0:
                voice_detected = True
            elif self._recording:
                voice_detected = bool(self._confidence_ema > stop_threshold)
            else:
                voice_detected = bool(confidence >= start_threshold)
            voice_detected = bool(voice_detected)  # Ensure native Python bool
            snr_display = getattr(self, '_last_snr_db', -60.0)
            
            # START RECORDING - use sensitivity preset thresholds
            snr_now = getattr(self, '_last_snr_db', 0.0)
            pitch_score = getattr(self, '_last_pitch_score', 0.0)
            
            # CONFIRMING / START RECORDING state machine (v1.36 — chunk-count window)
            # IDLE: single frame >= start_threshold → enter CONFIRMING (no file opened yet).
            # CONFIRMING: accumulate EXACTLY confirm_window_chunks frames, then evaluate gates.
            # Gate fires only after N chunks — wall clock is NOT used for the gate decision.
            # This prevents audio-callback burst batching from spoofing the window:
            # a noise burst of 6 frames arriving in 0.1s can't satisfy a 65-chunk window.
            # Data: 11175 noise max 7/65 hits (10.8%), max run=3 → both gates stay closed.
            confirm_window_secs   = preset.get("confirm_window_seconds", 3.0)
            confirm_ratio         = preset.get("confirm_min_ratio", 0.20)
            confirm_min_run       = preset.get("confirm_min_run_chunks", 5)
            sr = getattr(self, '_sample_rate', 44100)
            confirm_window_chunks = max(10, round(confirm_window_secs * sr / BLOCK_SIZE))

            if not self._recording and not self._confirming:
                if confidence >= start_threshold:
                    # Enter CONFIRMING — do not open a file yet
                    self._confirming = True
                    self._confirm_start_time = now  # wall clock for display only
                    self._confirm_hit_count = 1
                    self._confirm_total_count = 1
                    self._confirm_current_run = 1
                    self._confirm_max_run = 1
                    self.status_bar.showMessage(f"Confirming voice... ({confidence:.0f}%)")
                    if self._meter_debug:
                        print(f">>> CONFIRMING <<< Score {confidence:.0f}/100 >= {start_threshold}%  window={confirm_window_chunks}chk")

            elif not self._recording and self._confirming:
                self._confirm_total_count += 1
                if confidence >= start_threshold:
                    self._confirm_hit_count += 1
                    self._confirm_current_run += 1
                    if self._confirm_current_run > self._confirm_max_run:
                        self._confirm_max_run = self._confirm_current_run
                else:
                    self._confirm_current_run = 0  # streak broken
                ratio_now = self._confirm_hit_count / self._confirm_total_count

                # Gate fires after exactly confirm_window_chunks frames — no wall-clock shortcut
                if self._confirm_total_count >= confirm_window_chunks:
                    passes_ratio = ratio_now >= confirm_ratio
                    passes_run   = self._confirm_max_run >= confirm_min_run
                    if passes_ratio and passes_run:
                        # Both gates passed — this is real voice, start recording
                        if self._meter_debug:
                            print(f">>> RECORDING START <<< {self._confirm_hit_count}/{self._confirm_total_count}chk ratio={ratio_now:.0%}>={confirm_ratio:.0%}, run={self._confirm_max_run}>={confirm_min_run}")
                        self._confirming = False
                        self._confirm_hit_count = 0
                        self._confirm_total_count = 0
                        self._confirm_current_run = 0
                        self._confirm_max_run = 0
                        self._start_recording()
                        self.status_bar.showMessage(f"Recording started | Voice confirmed (SNR={snr_display:.1f}dB)")
                    else:
                        # At least one gate failed — discard as noise crash or blip
                        reason = []
                        if not passes_ratio: reason.append(f"ratio {ratio_now:.0%}<{confirm_ratio:.0%}")
                        if not passes_run:   reason.append(f"run {self._confirm_max_run}<{confirm_min_run}chk")
                        blip_info = ", ".join(reason)
                        if self._meter_debug:
                            print(f">>> DISCARDED <<< {blip_info}")
                        self._confirming = False
                        self._confirm_hit_count = 0
                        self._confirm_total_count = 0
                        self._confirm_current_run = 0
                        self._confirm_max_run = 0
                        self.status_bar.showMessage(f"Discarded ({blip_info})")
                else:
                    # Still accumulating — show live chunk-based progress
                    pct = int(100 * self._confirm_total_count / confirm_window_chunks)
                    self.status_bar.showMessage(
                        f"Confirming... ({self._confirm_hit_count}/{self._confirm_total_count}chk {pct}%, run={self._confirm_max_run}/{confirm_min_run})"
                    )

            # CONTINUE RECORDING - while voice is detected (threshold met or in hangover)
            elif self._recording:
                # Safety cap: enforce max_recording_duration from preset
                rec_duration = (now - self._voice_started_at) if self._voice_started_at else 0
                max_rec_dur = preset.get("max_recording_duration", 300)
                if rec_duration >= max_rec_dur:
                    if self._meter_debug:
                        print(f">>> RECORDING STOP <<< Max duration {max_rec_dur}s reached")
                    self._finalize_recording()
                    self.status_bar.showMessage(f"Recording stopped | Max duration reached ({max_rec_dur:.0f}s)")
                elif voice_detected:
                    # Still recording - voice or within hangover window
                    self.status_bar.showMessage(
                        f"Recording... | Score: {confidence:.0f}/100 | EMA: {self._confidence_ema:.0f}% | Hangover: {self._hangover_remaining:.2f}s"
                    )
                else:
                    # EMA dropped below stop_threshold after hangover — STOP RECORDING
                    if self._meter_debug:
                        print(f">>> RECORDING STOP <<< EMA {self._confidence_ema:.0f}% <= stop {stop_threshold}%")
                    self._finalize_recording()
                    self.status_bar.showMessage(f"Recording stopped | EMA {self._confidence_ema:.0f}% below threshold")
            
            # Update voice indicator  
            if voice_detected != self._voice_detected:
                self._voice_detected = voice_detected
                # Ensure we emit a native Python bool, not numpy.bool_
                self._signals.voice.emit(bool(voice_detected))


    def _start_recording(self):
        # Safety: clear any dangling confirming state
        self._confirming = False
        self._confirm_start_time = None
        self._confirm_hit_count = 0
        self._confirm_total_count = 0
        self._confirm_current_run = 0
        self._confirm_max_run = 0
        self._voice_started_at = time.time()
        self._voice_silence_at = None
        self._silence_timer_remaining = 0

        # CRITICAL FIX: Reset confidence duration accumulator when recording starts
        # This ensures only voice time DURING recording counts, not static accumulated before
        self._voice_confidence_duration = 0.0
        self._last_high_confidence_time = None
        
        # Track that voice was confirmed
        now = time.time()
        self._last_confirmed_voice_time = now
        
        # HANGOVER FIX: Seed _last_word_peak_time to NOW so the full post-roll window
        # is available immediately when recording starts. Without this, the word-peak
        # timer starts at (now - post_roll_seconds), making post_roll_remaining = 0
        # and hangover never initialises on faint signals that never hit word_peak_threshold.
        self._last_word_peak_time = now

        # Seed hangover to full post-roll duration immediately — voice was just confirmed
        # via the 3s dual-gate, so it deserves the full window from the start.
        # Also pre-prime _hangover_voice_run to hangover_repin_run_chunks so the VERY FIRST
        # voice frame holds the hangover rather than needing to rebuild a run from scratch.
        # Without this: hangover=0 at start, EMA dip in opening frames closes the file.
        _preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
        self._hangover_remaining = float(_preset.get("post_roll_seconds", 10))
        self._hangover_voice_run = _preset.get("hangover_repin_run_chunks", 3)  # pre-primed
        if hasattr(self, '_hangover_started'):
            delattr(self, '_hangover_started')
        self._post_roll_silence_frames = 0
        
        # Copy pre-roll buffer to recording buffer atomically under BOTH locks
        # to prevent the audio callback from writing between the two operations
        with self._lock:
            with self._audio_buffer_lock:
                self._audio_buffer = list(self._ring_buffer)
        
        # Set recording flag AFTER buffer is initialized, so callback appends correctly
        self._recording = True
        
        self._signals.status.emit(
            f"Recording... | VAD: Voice | Pre-roll: {len(self._audio_buffer) * BLOCK_SIZE / self._sample_rate:.1f}s"
        )
        self._signals.recording.emit(True)  # Trigger animation start

    def _finalize_recording(self):
        # Reset confirmation state (safety — may be mid-confirm when forced-stop)
        self._confirming = False
        self._confirm_start_time = None
        self._confirm_hit_count = 0
        self._confirm_total_count = 0
        self._confirm_current_run = 0
        self._confirm_max_run = 0
        self._recording = False
        self._voice_started_at = None
        self._voice_silence_at = None
        self._signals.recording.emit(False)  # Stop animation

        # Reset hangover completely so voice LED clears and next recording starts clean
        self._hangover_remaining = 0.0
        self._hangover_voice_run = 0
        if hasattr(self, '_hangover_started'):
            delattr(self, '_hangover_started')
        
        # Reset post-roll silence counter
        self._post_roll_silence_frames = 0
        
        # Capture voice confirmation duration before clearing
        voice_duration = self._voice_confidence_duration
        
        # Get audio buffer without holding lock during concatenation
        with self._audio_buffer_lock:
            buffer_copy = list(self._audio_buffer)
            self._audio_buffer = []
        
        # Concatenate outside lock (can be slow)
        if buffer_copy:
            audio_data = np.concatenate(buffer_copy)
        else:
            audio_data = np.array([])
        
        if len(audio_data) > self._sample_rate * MIN_RECORDING_DURATION:
            threading.Thread(target=self._encode_and_upload, args=(audio_data, voice_duration), daemon=True).start()
            self._signals.status.emit("Uploading...")

    def _encode_and_upload(self, audio_data: np.ndarray, voice_duration: float = 0.0):
        try:
            # MINIMAL ARTIFACT FILTERING: Only reject obvious silence
            # Noise floor threshold tied to sensitivity setting (1-5)
            # Short bursts (static, sweeps, etc.) are allowed through
            # Better to upload an artifact than miss real voice
            
            rms_db = 20 * np.log10(np.sqrt(np.mean(audio_data ** 2)) + 1e-10)
            
            # Get noise floor threshold from current sensitivity preset
            preset = SENSITIVITY_PRESETS.get(self._sensitivity_level, SENSITIVITY_PRESETS[3])
            noise_floor = preset.get("noise_floor_db", -60.0)
            
            # REJECT: Only if signal is below sensitivity-based noise floor
            if rms_db < noise_floor:
                self._signals.status.emit(f"Filtered: Signal too quiet ({rms_db:.1f} dB < {noise_floor:.1f} dB threshold)")
                if self._meter_debug:
                    print(f"[ARTIFACT REJECTED] Too quiet: {rms_db:.1f} dB (threshold: {noise_floor:.1f} dB)")
                return
            
            # Otherwise: Upload even if it's a brief burst
            if self._meter_debug:
                print(f"[UPLOAD APPROVED] Brief burst recorded (RMS:{rms_db:.1f}dB, threshold:{noise_floor:.1f}dB) - Uploading")
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
            
            # Apply audio processing (bandpass, denoise, voice extraction, etc.)
            audio_data = self._process_audio(audio_data)
            
            # Write WAV with Windows-safe filename (no colons)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"osintcom_{self._frequency}_{timestamp}.wav"
            filepath = os.path.join(self._save_dir, filename)
            import scipy.io.wavfile as wav
            wav.write(filepath, self._sample_rate, (audio_data * 32767).astype(np.int16))
            self._signals.status.emit(f"Saved: {os.path.basename(filepath)}")
            
            # Upload to Discord (all enabled webhooks)
            enabled_webhooks = [w for w in self._webhooks if w.get("enabled", True)]
            if enabled_webhooks:
                self._upload_to_discord(filepath)
                self._signals.status.emit(f"Uploaded to {len(enabled_webhooks)} webhook(s) | Frequency: {self._frequency}")
        except Exception as e:
            self._signals.error.emit(f"Encoding/upload failed: {e}")

    def _check_upload_syllabic_modulation(self, audio: np.ndarray) -> float:
        """Check for speech-like syllabic modulation in recording.
        
        Speech has clear envelope modulation at syllable rate (~3-8 Hz).
        Static/noise has random or flat envelope.
        
        Returns: 0-1 score (higher = more speech-like)
        """
        try:
            # Compute envelope via RMS in short windows
            window = 256  # ~6ms at 44kHz
            envelope = []
            
            for i in range(0, len(audio) - window, window):
                rms = np.sqrt(np.mean(audio[i:i+window] ** 2))
                envelope.append(rms)
            
            if len(envelope) < 10:
                return 0.3
            
            envelope = np.array(envelope, dtype=np.float32)
            
            # Normalize envelope
            env_mean = np.mean(envelope)
            if env_mean < 1e-10:
                return 0.2
            
            envelope_norm = envelope / (env_mean + 1e-10)
            
            # Check for variation (speech varies, static is flat)
            envelope_energy = np.var(envelope_norm)  # Higher = more variation
            
            # Speech recordings have envelope variance ~0.05-0.15
            # Static recordings have variance ~0.01-0.03
            
            if envelope_energy > 0.10:
                modulation = 1.0  # Strong modulation
            elif envelope_energy > 0.05:
                modulation = (envelope_energy - 0.05) / 0.05  # Partial modulation
            elif envelope_energy > 0.02:
                modulation = (envelope_energy - 0.02) / 0.03 * 0.5  # Weak modulation
            else:
                modulation = 0.1  # Flat envelope = static
            
            if self._meter_debug:
                print(f"    [Modulation energy: {envelope_energy:.4f} → {modulation:.0%}]")
            
            return np.clip(modulation, 0.0, 1.0)
        except:
            return 0.3

    def _check_upload_spectral_consistency(self, audio: np.ndarray) -> float:
        """Check spectral shape consistency across recording.
        
        Voice has consistent formant structure; noise/static varies.
        Compares spectral shape of first vs second half of recording.
        
        Returns: 0-1 score (higher = more consistent = more voice-like)
        """
        try:
            # Split audio in half
            mid = len(audio) // 2
            first_half = audio[:mid]
            second_half = audio[mid:]
            
            if len(first_half) < 512 or len(second_half) < 512:
                return 0.5
            
            # Compute spectra with Welch method
            from scipy.signal import welch
            
            f1, p1 = welch(first_half, fs=self._sample_rate, nperseg=512)
            f2, p2 = welch(second_half, fs=self._sample_rate, nperseg=512)
            
            # Normalize power
            p1 = p1 / (np.max(p1) + 1e-10)
            p2 = p2 / (np.max(p2) + 1e-10)
            
            # Compute correlation between spectral shapes (0-1)
            # Voice: high correlation (~0.7-0.95), Static: low (~0.3-0.6)
            correlation = np.corrcoef(p1[:256], p2[:256])[0, 1]
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=0.0)
            correlation = np.clip(correlation, 0.0, 1.0)
            
            # Map correlation to score: <0.4 = 0%, >0.7 = 100%
            if correlation > 0.70:
                score = 1.0
            elif correlation > 0.50:
                score = (correlation - 0.50) / 0.20
            elif correlation > 0.30:
                score = (correlation - 0.30) / 0.20 * 0.5
            else:
                score = 0.0
            
            if self._meter_debug:
                print(f"    [Spectral correlation: {correlation:.2f} → {score:.0%}]")
            
            return score
        except:
            return 0.5

    def _upload_to_discord(self, filepath: str):
        try:
            # Verify file exists and is readable before attempting upload
            if not os.path.exists(filepath):
                self._signals.error.emit(f"Upload failed: File not found - {filepath}")
                return
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                self._signals.error.emit(f"Upload failed: File is empty - {filepath}")
                return
            
            if self._meter_debug:
                print(f"[UPLOAD] Starting upload of {os.path.basename(filepath)} ({file_size} bytes)")
            
            # Send to all enabled webhooks
            for webhook in self._webhooks:
                if webhook.get("enabled", True):
                    try:
                        # Build payload with webhook-specific role_id
                        role_id = webhook.get("role_id", "")
                        role_part = f"<@&{role_id}>" if role_id else ""
                        
                        payload = {
                            'content': f"{role_part} {self._webhook_message}",
                            'embeds': [{
                                'title': f"{self._frequency} kHz",
                                'color': 3066993,
                                'timestamp': datetime.datetime.utcnow().isoformat()
                            }]
                        }
                        
                        # Read file once into memory before posting
                        try:
                            with open(filepath, 'rb') as f:
                                file_data = f.read()
                        except Exception as e:
                            raise Exception(f"Failed to read file: {e}")
                        
                        if len(file_data) == 0:
                            raise Exception("File read returned empty data")
                        
                        # Now upload with file data already in memory
                        files = {'file1': (os.path.basename(filepath), file_data)}
                        data = {'payload_json': json.dumps(payload)}
                        
                        response = requests.post(webhook.get("url"), data=data, files=files, timeout=30)
                        response.raise_for_status()  # Raise exception for bad HTTP status
                        
                        nickname = webhook.get("nickname", "Unknown")
                        if self._meter_debug:
                            print(f"[UPLOAD] Successfully uploaded to '{nickname}' (HTTP {response.status_code})")
                    
                    except Exception as e:
                        nickname = webhook.get("nickname", "Unknown")
                        self._signals.error.emit(f"Upload to '{nickname}' failed: {e}")
                        if self._meter_debug:
                            print(f"[UPLOAD ERROR] Failed to upload to '{nickname}': {e}")
                            traceback.print_exc()
        except Exception as e:
            self._signals.error.emit(f"Discord upload failed: {e}")
            if self._meter_debug:
                traceback.print_exc()

    def _on_level(self, db):
        pass

    def _on_voice(self, detected):
        if detected:
            self.voice_label.setText("voice")
            self.voice_label.setStyleSheet("color: green; font-weight: bold; font-size: 12px;")
        else:
            self.voice_label.setText("static")
            self.voice_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
        self._update_compact_display()

    def _on_recording(self, is_recording):
        """Handle recording state change - control ticker animation."""
        if is_recording:
            self.ticker.start_animation()
        else:
            self.ticker.stop_animation()

    def _on_status(self, msg):
        self.status_bar.showMessage(msg)
        self._update_compact_display()

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def _save_config(self):
        """Save all user settings to a JSON configuration file."""
        config = {
            "device": self.device_combo.currentText(),
            "sensitivity": self.sens_slider.value(),
            "frequency": self.freq_display.text(),
            "webhooks": self._webhooks,
            "custom_message": self.message_edit.text(),
            "file_location": self._save_dir,
            "audio_settings": self._audio_settings
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", "Settings saved successfully!")
            self.status_bar.showMessage("Settings saved successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def _load_config(self):
        """Load settings from configuration file if it exists."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                
                # Load all settings
                if "device" in config:
                    index = self.device_combo.findText(config["device"])
                    if index != -1:
                        self.device_combo.setCurrentIndex(index)
                
                if "sensitivity" in config:
                    self.sens_slider.setValue(int(config["sensitivity"]))
                
                if "frequency" in config:
                    self.freq_display.setText(config["frequency"])
                    self._frequency = config["frequency"].split()[0]  # Extract just the number
                
                # Load webhooks (new format) or convert from old format
                if "webhooks" in config:
                    self._webhooks = config["webhooks"]
                    enabled_count = sum(1 for w in self._webhooks if w.get("enabled", True))
                    total_count = len(self._webhooks)
                    if total_count > 0:
                        self.webhook_edit.setText(f"✓ {enabled_count}/{total_count} webhooks active")
                    self.webhook_edit.setReadOnly(True)
                elif "webhook_url" in config and config["webhook_url"]:  # Backwards compatibility
                    old_url = config["webhook_url"]
                    if old_url.startswith("✓"):
                        old_url = old_url.split("(")[1].rstrip(")")
                    self._webhooks = [{
                        "nickname": "Legacy Webhook",
                        "url": old_url,
                        "enabled": True
                    }]
                    self.webhook_edit.setText("✓ 1/1 webhooks active")
                    self.webhook_edit.setReadOnly(True)
                
                if "custom_message" in config:
                    self.message_edit.setText(config["custom_message"])
                    self._webhook_message = config["custom_message"]
                
                if "file_location" in config:
                    self._save_dir = config["file_location"]
                
                if "audio_settings" in config:
                    self._audio_settings.update(config["audio_settings"])
                
                self.status_bar.showMessage("Settings loaded from config")
            except Exception as e:
                print(f"Warning: Could not load config file: {str(e)}")

    def closeEvent(self, event):
        self._meter_timer.stop()
        if self._running:
            self._on_stop()
        event.accept()

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OSINTCOMWindow()
    window.show()
    sys.exit(app.exec_())
