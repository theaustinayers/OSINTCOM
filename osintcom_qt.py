"""
OSINTCOM - Cross-platform Voice Activity Detection and Recording Tool
PyQt5 GUI with real-time audio monitoring, VAD, recording, and Discord integration.
"""

import sys
import os
import io
import time
import threading
import datetime
import json
import collections
import traceback
import wave
import struct
import uuid

import numpy as np
import sounddevice as sd
import requests
# scipy imports removed - using numpy-only signal processing now
# from scipy.signal import butter, sosfiltfilt, welch, get_window
# from scipy.fftpack import fft
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QDialog, QLineEdit,
    QGroupBox, QStatusBar, QSlider, QTextEdit, QDialogButtonBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPainter, QColor, QLinearGradient

# ============================================================================
# Constants
# ============================================================================
CONFIG_FILE = "osintcom_config.json"

CHANNELS = 1
BLOCK_SIZE = 2048
PRE_ROLL_SECONDS = 3.0  # 3 seconds of audio before VAD triggers
POST_ROLL_SECONDS = 10.0  # 10 seconds of silence/decay after last word peak
MIN_VOICE_DURATION = 0.5
MIN_RECORDING_DURATION = 1.5
VAD_SMOOTHING_WINDOW = 3  # Confidence smoothing window
VAD_WORD_PEAK_THRESHOLD = 75  # Confidence needed to consider it a word
VAD_START_CONFIDENCE = 65  # Start recording threshold
VAD_CONTINUE_CONFIDENCE = 30  # Continue recording threshold
VAD_STOP_CONFIDENCE = 20  # Stop recording threshold

# Confidence-Based Professional VAD (v1.06)
# Scores voice 0-100 based on: energy, band dominance, spectral entropy, ZCR, pitch
# Detects word peaks and uses 10-second post-roll from last word
SENSITIVITY_PRESETS = {
    # Level 1: Maximum sensitivity - catches faintest voices
    1: {"confidence_start": 55, "confidence_continue": 20, "confidence_stop": 10,
        "word_peak_threshold": 65, "post_roll_seconds": 10,
        "energy_weight": 0.30, "band_weight": 0.25, "entropy_weight": 0.20,
        "zcr_weight": 0.15, "pitch_weight": 0.10},
    # Level 2: Very sensitive - good for weak radio
    2: {"confidence_start": 60, "confidence_continue": 25, "confidence_stop": 15,
        "word_peak_threshold": 70, "post_roll_seconds": 10,
        "energy_weight": 0.30, "band_weight": 0.25, "entropy_weight": 0.20,
        "zcr_weight": 0.15, "pitch_weight": 0.10},
    # Level 3: Balanced (default)
    3: {"confidence_start": 65, "confidence_continue": 30, "confidence_stop": 20,
        "word_peak_threshold": 75, "post_roll_seconds": 10,
        "energy_weight": 0.30, "band_weight": 0.25, "entropy_weight": 0.20,
        "zcr_weight": 0.15, "pitch_weight": 0.10},
    # Level 4: Strict - rejects static
    4: {"confidence_start": 70, "confidence_continue": 40, "confidence_stop": 25,
        "word_peak_threshold": 80, "post_roll_seconds": 10,
        "energy_weight": 0.30, "band_weight": 0.25, "entropy_weight": 0.20,
        "zcr_weight": 0.15, "pitch_weight": 0.10},
    # Level 5: Voice only - maximum rejection
    5: {"confidence_start": 75, "confidence_continue": 50, "confidence_stop": 30,
        "word_peak_threshold": 85, "post_roll_seconds": 10,
        "energy_weight": 0.30, "band_weight": 0.25, "entropy_weight": 0.20,
        "zcr_weight": 0.15, "pitch_weight": 0.10},
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
# Dialogs
# ============================================================================
class WebhookDialog(QDialog):
    def __init__(self, current_url: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Discord Webhook URL")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter your Discord Webhook URL:"))
        self.url_edit = QLineEdit(current_url)
        self.url_edit.setPlaceholderText("https://discord.com/api/webhooks/...")
        layout.addWidget(self.url_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_url(self) -> str:
        return self.url_edit.text().strip()

class WebhookCustomizeDialog(QDialog):
    def __init__(self, role_id: str = "", message_template: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Webhook Message")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Role ID to ping (leave blank for none):"))
        self.role_edit = QLineEdit(role_id)
        self.role_edit.setPlaceholderText("e.g. 1474631083042541730")
        layout.addWidget(self.role_edit)
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
        self.role_edit.textChanged.connect(self._update_preview)
        self.msg_edit.textChanged.connect(self._update_preview)
        self._update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self, *_):
        role = self.role_edit.text().strip()
        msg = self.msg_edit.toPlainText().strip() or "EMERGENCY ACTION MESSAGE INCOMING"
        role_part = f"@Role({role}) " if role else ""
        freq = "8992 kHz"
        self.preview_label.setText(f"{role_part}{msg}\n─ {freq} | {datetime.datetime.utcnow().isoformat()}Z")

    def get_role_id(self) -> str:
        return self.role_edit.text().strip()

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
        
        layout.addSpacing(12)
        layout.addWidget(QLabel("ℹ️ Voice-Only mode sends cleanest audio but may cut very quiet voices.", 
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
        }

# ============================================================================
# Main Window
# ============================================================================
class OSINTCOMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSINTCOM")
        self.setMinimumSize(700, 620)
        self.setFont(QFont("Segoe UI", 10))
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
        
        # State
        self._running = False
        self._recording = False
        self._stream = None
        self._sample_rate = 44100
        self._lock = threading.Lock()
        self._ring_buffer = collections.deque(maxlen=int(PRE_ROLL_SECONDS * self._sample_rate / BLOCK_SIZE))
        self._audio_buffer_lock = threading.Lock()
        self._audio_buffer = []
        self._peak_db = -60.0
        self._voice_detected = False
        self._voice_history = collections.deque(maxlen=VAD_SMOOTHING_WINDOW)
        self._voice_started_at = None
        self._voice_silence_at = None
        self._silence_timer_remaining = 0
        self._sensitivity_level = 3
        self._previous_energy_db = -60.0  # Track energy for drop-off detection
        self._webhook_url = ""
        self._webhook_role_id = ""
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
        self._adaptive_snr_threshold = 8.0  # Lowered from 11 - Silero is primary (was 13 in v1.08.3)
        self._adaptive_cv_min = 0.25  # Default, updated by calibration
        self._adaptive_cv_max = 0.60  # Default, updated by calibration
        
        # SSB-friendly SNR gate: use rolling minimum of last 2s
        self._snr_percentile_window = 87  # ~2 seconds of SNR samples
        
        # Voice confirmation timer - requires 3.5 seconds of sustained high confidence
        self._voice_confidence_duration = 0.0  # Cumulative seconds of high confidence
        self._voice_confirmation_threshold = 3.5  # Seconds of voice confidence needed
        self._last_high_confidence_time = None  # Timestamp of last high confidence frame
        
        # Signal-based VAD (no ML dependencies)
        self._silero_model = None
        self._silero_ready = False
        
        # Audio processing settings
        self._audio_settings = {
            "use_bandpass": False,
            "use_denoise": True,
            "denoise_strength": 6,
            "remove_silence": False,
            "voice_extract": False,
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
        version_label = QLabel("v1.10c")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        layout.addWidget(version_label)

        # HFGCS Frequencies
        self.hfgcs_frequencies = [4724.0, 6739.0, 8992.0, 11175.0, 13200.0, 15016.0, 18046.0]
        freq_group = QGroupBox("HFGCS Frequencies")
        freq_layout = QHBoxLayout(freq_group)
        self.freq_buttons = []
        for freq in self.hfgcs_frequencies:
            btn = QPushButton(f"{freq:.0f} kHz")
            btn.setCheckable(True)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda checked, f=freq: self._on_freq_selected(f))
            freq_layout.addWidget(btn)
            self.freq_buttons.append(btn)
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
        audio_layout.addLayout(sens_row)
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
        webhook_btn = QPushButton("Webhook")
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
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.file_btn)
        controls_layout.addWidget(self.audio_settings_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.calibrate_btn)
        layout.addWidget(controls_group)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | VAD: Idle | Recording: Stopped | Pre-roll: 5s")

        self.setCentralWidget(central)

    def _connect_signals(self):
        self._signals.level.connect(self._on_level)
        self._signals.voice.connect(self._on_voice)
        self._signals.status.connect(self._on_status)
        self._signals.error.connect(self._on_error)

    def _init_timer(self):
        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(50)
        self._meter_timer.timeout.connect(self._update_meter)

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
        self._save_config()  # Auto-save frequency

    def _on_sensitivity_changed(self, value):
        self._sensitivity_level = value
        self.sens_label.setText(SENSITIVITY_LABELS[value])

    def _open_webhook_dialog(self):
        dlg = WebhookDialog(self._webhook_url, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhook_url = dlg.get_url()
            self.webhook_edit.setText("✓ Webhook configured (" + self._webhook_url[-20:] + ")")
            self.webhook_edit.setReadOnly(True)
            self._save_config()  # Auto-save webhook

    def _open_customize_dialog(self):
        dlg = WebhookCustomizeDialog(self._webhook_role_id, self._webhook_message, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhook_role_id = dlg.get_role_id()
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
        if not self._webhook_url:
            QMessageBox.warning(self, "Warning", "Please configure Discord webhook first.")
            return
        self._running = True
        self._learning_phase = "startup"  # Start with initial learning
        self._last_learning_time = time.time()
        self._last_relearn_time = time.time()
        self._noise_samples.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._meter_timer.start()
        self._start_audio_stream()

    def _on_stop(self):
        self._running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._meter_timer.stop()
        self._stop_audio_stream()

    def _on_calibrate_noise(self):
        """Record 10 seconds of ambient noise, analyze it, and auto-adjust VAD thresholds."""
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
        
        # After 10 seconds, analyze and adjust
        QTimer.singleShot(10000, self._analyze_calibration)

    def _analyze_calibration(self):
        """Analyze recorded noise and adjust VAD thresholds."""
        self._calibration_active = False
        
        if len(self._calibration_samples) < 100:
            self.status_bar.showMessage("Calibration failed: insufficient audio samples")
            self.calibrate_btn.setEnabled(True)
            return
        
        try:
            # Concatenate all samples
            audio_data = np.concatenate(self._calibration_samples)
            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            
            # Analyze noise properties
            rms_noise = np.sqrt(np.mean(audio_data ** 2))
            snr_db_noise = 20 * np.log10((rms_noise + 1e-10) / (self._noise_floor_rms + 1e-10))
            
            # Check modulation (CV) of noise
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
            
            # Adjust thresholds based on noise characteristics
            # If noise is high energy, raise SNR gate
            # If noise has high modulation (variable), tighten CV range
            
            old_snr = self._open_threshold
            old_cv_min = 0.28
            old_cv_max = 0.55
            
            # Adaptive SNR: noise SNR was X, so require human voice to be at least +4dB above that
            new_snr = max(13.0, snr_db_noise + 4.0)  # At least 13 dB floor
            
            # Adaptive CV: if noise CV is high (variable), tighten CV acceptance
            if cv_noise > 0.45:
                new_cv_min = 0.32  # Tighten range
                new_cv_max = 0.50
            else:
                new_cv_min = 0.28
                new_cv_max = 0.55
            
            # Log the changes
            msg = f"📊 Calibration Complete:\n\n"
            msg += f"Noise SNR: {snr_db_noise:.1f} dB\n"
            msg += f"Noise CV: {cv_noise:.2f}\n\n"
            msg += f"Adjusted thresholds:\n"
            msg += f"SNR: {old_snr:.1f} → {new_snr:.1f} dB\n"
            msg += f"CV: {old_cv_min:.2f}-{old_cv_max:.2f} → {new_cv_min:.2f}-{new_cv_max:.2f}"
            
            # Store new thresholds in VAD
            self._adaptive_snr_threshold = new_snr
            self._adaptive_cv_min = new_cv_min
            self._adaptive_cv_max = new_cv_max
            
            self.status_bar.showMessage(f"✓ Calibrated: SNR={new_snr:.1f}dB, CV={new_cv_min:.2f}-{new_cv_max:.2f}")
            
            QMessageBox.information(self, "Calibration Complete", msg)
            
        except Exception as e:
            self.status_bar.showMessage(f"Calibration error: {str(e)[:50]}")
            QMessageBox.warning(self, "Calibration Error", f"Failed to analyze noise: {str(e)[:100]}")
        finally:
            self.calibrate_btn.setEnabled(True)
            self._calibration_samples = []

    def _start_audio_stream(self):
        try:
            device_data = self.device_combo.currentData()
            device_idx = device_data if device_data is not None else None
            device_info = sd.query_devices(device_idx)
            self._sample_rate = int(device_info['default_samplerate'])
            self._stream = sd.InputStream(
                device=device_idx, channels=CHANNELS, samplerate=self._sample_rate,
                blocksize=BLOCK_SIZE, callback=self._audio_callback
            )
            self._stream.start()
            self._signals.status.emit(f"Listening on {device_info['name']}")
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
        """v1.11 Intelligent Voice Detection - Signal-Based Analysis.
        
        Uses 4-layer detection (no ML dependencies):
        1. SNR gate (noise floor relative calculation)
        2. Pitch detection (voice fundamental ~85-250Hz)
        3. Spectral entropy (voice organized, static chaotic)
        4. Zero-crossing rate (voice smooth, static choppy)
        
        Returns: 0-100 confidence where 50+ = voice
        """
        if len(audio) < 512:
            return 0.0
        
        try:
            debug = self._meter_debug
            
            # ===== SAFETY: CLIP EXTREME AUDIO =====
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # ===== INITIALIZATION =====
            if not hasattr(self, '_noise_floor_rms'):
                self._noise_floor_rms = 0.001
                self._snr_history = collections.deque(maxlen=300)
                self._hangover_remaining = 0.0
                self._close_threshold = 2.0  # dB SNR (very lenient during hangover, was 5.0)
                self._open_threshold = 8.0  # dB SNR (was 11.0, Silero is now primary)
            
            # ===== LEARNING PHASE =====
            if self._learning_phase == "startup":
                now = time.time()
                if now - self._last_learning_time < self._noise_learning_time:
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms < 0.01:
                        self._noise_floor_rms = rms * 0.9
                    return 0.0
                else:
                    self._learning_phase = "periodic"
                    print(f"✓ LEARNING COMPLETE: Noise floor = {20*np.log10(self._noise_floor_rms+1e-10):.1f} dB")
                    self._last_relearn_time = now
                    return 0.0
            
            # ===== PERIODIC NOISE FLOOR UPDATE =====
            elif self._learning_phase == "periodic" and not self._recording:
                now = time.time()
                rms = np.sqrt(np.mean(audio ** 2))
                if now - self._last_relearn_time > self._noise_relearn_interval and rms < 0.005:
                    self._noise_floor_rms = self._noise_floor_rms * 0.99 + rms * 0.01
                    self._last_relearn_time = now
            
            # ===== 4-LAYER INTELLIGENT VOICE DETECTION =====
            # Layer 1: SNR gate (foundation)
            # Layer 2: Pitch detection (voice has fundamental frequency ~85-250Hz)
            # Layer 3: Spectral entropy (voice organized, static chaotic)
            # Layer 4: Zero-crossing rate (voice smooth, static choppy)
            
            confidence = 0.0
            rms = np.sqrt(np.mean(audio ** 2))
            snr_db = 20 * np.log10((rms + 1e-10) / (self._noise_floor_rms + 1e-10))
            self._snr_history.append(snr_db)
            self._last_snr_db = snr_db
            
            if len(self._snr_history) > 10:
                snr_percentile = np.percentile(list(self._snr_history), 20)
            else:
                snr_percentile = snr_db
            
            # ===== LAYER 1: SNR GATE =====
            # Determine thresholds based on state
            if self._recording:
                snr_threshold = 0.0
            elif self._hangover_remaining > 0:
                snr_threshold = 2.0
            else:
                snr_threshold = 6.0
            
            snr_passes = snr_percentile > snr_threshold
            
            # Layer 1 score: 0-25 points (gate + bonus)
            if not snr_passes:
                confidence = 0.0  # SNR gate fails - everything fails
            else:
                confidence = 20.0  # Base score for passing SNR
                if snr_percentile > snr_threshold + 3.0:
                    confidence = 25.0  # Bonus for strong SNR
            
            # ===== LAYER 2: PITCH DETECTION =====
            # Voice fundamental ~85-250Hz (can reach 300Hz for children/women high)
            if confidence > 0:
                pitch_score = self._detect_pitch(audio)  # Returns 0-25
                confidence += pitch_score
                if debug:
                    print(f"  Layer 2 Pitch: +{pitch_score:.0f} (total: {confidence:.0f})")
            
            # ===== LAYER 3: SPECTRAL ENTROPY =====
            # Voice: organized spectrum (low entropy), Static: chaotic (high entropy)
            if confidence > 0:
                entropy_score = self._estimate_spectral_entropy(audio)  # Returns 0-25
                confidence += entropy_score
                if debug:
                    print(f"  Layer 3 Entropy: +{entropy_score:.0f} (total: {confidence:.0f})")
            
            # ===== LAYER 4: ZERO-CROSSING RATE =====
            # Voice: smooth transitions (low ZCR), Static: choppy (high ZCR)
            if confidence > 0:
                zcr_score = self._zero_crossing_rate_score(audio)  # Returns 0-25
                confidence += zcr_score
                if debug:
                    print(f"  Layer 4 ZCR: +{zcr_score:.0f} (total: {confidence:.0f})")
            
            # ===== MODULATION CHECK: Boost/Penalty =====
            # Natural voice has 0.25-0.60 coefficient of variation
            if confidence > 0:
                chunk_size = len(audio) // 10
                if chunk_size > 0:
                    rms_values = []
                    for i in range(10):
                        start = i * chunk_size
                        end = start + chunk_size if i < 9 else len(audio)
                        chunk_rms = np.sqrt(np.mean(audio[start:end] ** 2))
                        rms_values.append(chunk_rms)
                    
                    rms_values = np.array(rms_values, dtype=np.float32)
                    rms_mean = np.mean(rms_values)
                    if rms_mean > 1e-10:
                        cv = np.sqrt(np.var(rms_values)) / rms_mean
                    else:
                        cv = 0.0
                    
                    if self._adaptive_cv_min < cv < self._adaptive_cv_max:
                        confidence = min(100.0, confidence + 5.0)
                        if debug:
                            print(f"  Modulation: CV={cv:.2f} natural → +5 (total: {confidence:.0f})")
                    else:
                        confidence = max(5.0, confidence - 5.0)
                        if debug:
                            print(f"  Modulation: CV={cv:.2f} unnatural → -5 (total: {confidence:.0f})")
            
            if debug:
                print(f"[VOICE] SNR={snr_percentile:.1f}dB | Confidence={confidence:.0f}/100 | Hangover={self._hangover_remaining:.2f}s")
            
            return np.clip(confidence, 0.0, 100.0)
            
        except Exception as e:
            print(f"[VAD EXCEPTION] {str(e)[:100]}")
            return 0.0
    
    def _detect_pitch(self, audio: np.ndarray) -> float:
        """Detect pitch/periodicity in voice range 85-250Hz.
        
        Voice has fundamental frequency. Static bursts have no coherent pitch.
        Returns: 0-25 points (0 = no pitch, 25 = strong pitch in voice range)
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
            
            # Voice: peak_strength > 0.6, Static: < 0.3
            # Map: <0.3 = 0 pts, >0.6 = 25 pts, linear between
            if peak_strength > 0.60:
                return 25.0
            elif peak_strength < 0.30:
                return 0.0
            else:
                return (peak_strength - 0.30) / 0.30 * 25.0
        except:
            return 0.0
    
    def _estimate_spectral_entropy(self, audio: np.ndarray) -> float:
        """Estimate spectral entropy - low for voice, high for static.
        
        Voice has organized spectrum with formants.
        Static has relatively flat spectrum.
        Returns: 0-25 points (0 = chaotic/noise, 25 = organized/voice)
        """
        try:
            if len(audio) < 512:
                return 0.0
            
            # Compute power spectral density via simple FFT
            audio_work = audio - np.mean(audio)
            # Use Hamming window to reduce spectral leakage
            window = np.hamming(len(audio_work))
            audio_windowed = audio_work * window
            
            fft_data = np.fft.rfft(audio_windowed)
            power = np.abs(fft_data) ** 2
            power = power / (np.sum(power) + 1e-10)  # Normalize to probability distribution
            
            # Shannon entropy: -sum(p * log(p))
            # Low entropy = concentrated energy (voice)
            # High entropy = spread energy (static)
            power_clipped = np.clip(power, 1e-10, 1.0)
            entropy = -np.sum(power_clipped * np.log(power_clipped + 1e-10))
            
            # Normalize by max entropy (uniform distribution)
            max_entropy = np.log(len(power))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            
            # Voice: entropy ~0.3-0.5, Static: entropy ~0.7-0.9
            # Map: >0.7 = 0 pts (chaotic), <0.5 = 25 pts (organized), linear between
            if normalized_entropy > 0.70:
                return 0.0
            elif normalized_entropy < 0.50:
                return 25.0
            else:
                return (0.70 - normalized_entropy) / 0.20 * 25.0
        except:
            return 0.0
    
    def _zero_crossing_rate_score(self, audio: np.ndarray) -> float:
        """Calculate zero-crossing rate - low for voice, high for static.
        
        Voice has smooth modulation (low ZCR).
        Static has rapid fluctuations (high ZCR).
        Returns: 0-25 points (0 = high ZCR/noise, 25 = low ZCR/voice)
        """
        try:
            if len(audio) < 2:
                return 0.0
            
            # Count sign changes in audio signal
            audio_work = audio - np.mean(audio)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_work)))) / 2.0
            zcr = zero_crossings / len(audio_work)
            
            # Voice: ZCR ~0.08-0.15, Static: ZCR >0.25
            # Map: >0.25 = 0 pts, <0.15 = 25 pts, linear between
            if zcr > 0.25:
                return 0.0
            elif zcr < 0.15:
                return 25.0
            else:
                return (0.25 - zcr) / 0.10 * 25.0
        except:
            return 0.0
    
    def _check_syllabic_modulation(self, audio: np.ndarray) -> float:
        """Check for speech-like modulation at 3-8 Hz (syllabic rate).
        
        Human speech has energy modulation around syllable rate.
        Static bursts don't have this pattern.
        
        Returns: 0-1 score, higher = more speech-like modulation
        """
        try:
            audio_safe = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # Compute envelope (RMS over short windows)
            window = 128  # ~3ms at 44kHz
            envelope = []
            for i in range(0, len(audio_safe) - window, window):
                e = np.sqrt(np.mean(audio_safe[i:i+window] ** 2))
                envelope.append(e)
            
            if len(envelope) < 10:
                return 0.3
            
            envelope = np.array(envelope, dtype=np.float32)
            
            # FFT of envelope (looking for 3-8 Hz modulation)
            # envelope dt ≈ 128/44000 ≈ 3ms, so Fs ≈ 333 Hz
            try:
                from scipy.signal import welch
                freqs, pxx = welch(envelope, fs=len(envelope) * (self._sample_rate / len(audio_safe)))
                
                # Protect against NaN values
                pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Find energy in 3-8 Hz band
                mask = (freqs >= 3) & (freqs <= 8)
                speech_band_energy = np.mean(pxx[mask]) if np.any(mask) else 0
                
                # Find energy in silence band (0-1 Hz and >15 Hz)
                silence_mask = ((freqs >= 0) & (freqs <= 1)) | (freqs > 15)
                silence_band_energy = np.mean(pxx[silence_mask]) if np.any(silence_mask) else 1
                
                # Speech modulation ratio
                speech_band_energy = np.nan_to_num(speech_band_energy, nan=0.0, posinf=0.0, neginf=0.0)
                silence_band_energy = np.nan_to_num(silence_band_energy, nan=1e-10, posinf=1.0, neginf=1e-10)
                
                ratio = speech_band_energy / (silence_band_energy + 1e-10)
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                
                return np.clip(ratio, 0, 1.0)
            except:
                return 0.3
        except:
            return 0.3
    
    def _check_formant_structure(self, audio: np.ndarray) -> float:
        """Check for formant-like structure (peaks in spectrogram).
        
        Speech has formants (concentrated energy bands).
        Noise has relatively flat energy distribution.
        
        Returns: 0-1 score, higher = more formant-like (voice-like)
        """
        try:
            audio_safe = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # Compute spectrum
            freqs, pxx = welch(audio_safe, fs=self._sample_rate, nperseg=min(512, len(audio_safe)))
            
            # Protect against NaN and extreme values
            pxx = np.nan_to_num(pxx, nan=0.0, posinf=1e-10, neginf=0.0)
            
            # Normalize
            pxx_max = np.max(pxx)
            if pxx_max > 0:
                pxx = pxx / np.max([pxx_max, 1e-10])
            
            # Check if energy is concentrated (voice) vs uniform (noise)
            # Voice: peaks with valleys (kurtosis > 2)
            # Noise: relatively flat (kurtosis ≈ 0-1)
            
            # Simple: count number of local maxima in spectrum
            from scipy.signal import argrelextrema
            peaks = argrelextrema(pxx, np.greater, order=10)[0]
            
            # More peaks = more structure = more voice-like
            peak_ratio = len(peaks) / (len(pxx) / 50.0 + 1)  # Normalize by expected peaks
            peak_ratio = np.nan_to_num(peak_ratio, nan=0.0, posinf=0.0, neginf=0.0)
            
            return np.clip(peak_ratio, 0, 1.0)
        except:
            return 0.3
    
    def _calculate_confidence(self, audio: np.ndarray, preset: dict) -> float:
        """Calculate voice confidence 0-100 from 5 weighted features.
        
        Features:
        1. Energy in speech band (300-3000 Hz): 30 points
        2. Band dominance (speech band % of total energy): 25 points
        3. Spectral entropy (low = structured/voice, high = flat/noise): 20 points
        4. Zero-crossing rate (voice varies, noise constant): 15 points
        5. Pitch/periodicity (voice has pitch, noise doesn't): 10 points
        
        Total: 100 points
        """
        try:
            confidence = 0.0
            debug = self._meter_debug
            
            # Feature 1: Energy in speech band (300-3000 Hz) - 30 points
            energy_db = self._energy_db(audio)
            threshold_db = self._noise_floor_db + preset.get("confidence_start", 65)
            
            # Map energy to 0-30 points: below threshold = 0, at threshold + 6dB = 30
            db_above_threshold = energy_db - threshold_db
            energy_points = np.clip((db_above_threshold / 6.0) * 30, 0, 30)
            confidence += energy_points
            
            # Feature 2: Band dominance - 25 points
            band_energy = self._extract_speech_band_energy(audio)
            total_energy_linear = np.sqrt(np.mean(audio ** 2))
            if total_energy_linear > 1e-10:
                band_dominance = band_energy / total_energy_linear
                # Voice: >60% in speech band, static: <30%
                # Map: <30% = 0 pts, 60% = 25 pts, >80% = 25 pts
                if band_dominance < 0.30:
                    band_points = 0
                elif band_dominance < 0.60:
                    band_points = (band_dominance - 0.30) / 0.30 * 25
                else:
                    band_points = 25
                confidence += band_points
            else:
                band_points = 0
            
            # Feature 3: Spectral entropy (low = voice structure) - 20 points
            flatness = self._spectral_flatness(audio)
            # Voice: <0.4 flatness, Noise: >0.6
            # Map: >0.6 = 0 pts, <0.4 = 20 pts, linear between
            if flatness > 0.60:
                entropy_points = 0
            elif flatness < 0.40:
                entropy_points = 20
            else:
                entropy_points = (0.60 - flatness) / 0.20 * 20
            confidence += entropy_points
            
            # Feature 4: Zero-crossing rate (mid-range = voice) - 15 points
            zcr = self._zero_crossing_rate(audio)
            # Voice: 0.1-0.5, Static: >0.5
            # Map: <0.1 = 10 pts, 0.3 = 15 pts, >0.5 = 0 pts
            if zcr < 0.10:
                zcr_points = 10
            elif zcr < 0.30:
                zcr_points = 10 + (zcr - 0.10) / 0.20 * 5
            elif zcr < 0.50:
                zcr_points = 15 - (zcr - 0.30) / 0.20 * 15
            else:
                zcr_points = 0
            confidence += zcr_points
            
            # Feature 5: Pitch/periodicity (voice = periodic) - 10 points
            pitch = self._pitch_periodicity(audio)
            # Voice: >0.4 periodicity, Noise: <0.2
            # Map: <0.2 = 0 pts, >0.4 = 10 pts, linear between
            if pitch > 0.40:
                pitch_points = 10
            elif pitch < 0.20:
                pitch_points = 0
            else:
                pitch_points = (pitch - 0.20) / 0.20 * 10
            confidence += pitch_points
            
            if debug and self._learning_phase == "periodic":
                print(f"  [Features] E:{energy_points:.0f}({energy_db:.1f}dB) B:{band_points:.0f}({band_dominance:.0%}) S:{entropy_points:.0f}({flatness:.2f}) Z:{zcr_points:.0f}({zcr:.2f}) P:{pitch_points:.0f}({pitch:.2f})")
            
            return np.clip(confidence, 0.0, 100.0)
        except Exception as e:
            print(f"Confidence calc error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _extract_speech_band_energy(self, audio: np.ndarray) -> float:
        """Extract RMS energy in speech band (300-3000 Hz)."""
        try:
            # Apply bandpass filter
            sos = butter(4, [300, 3000], btype='band', fs=self._sample_rate, output='sos')
            filtered = sosfiltfilt(sos, audio)
            
            # Return RMS energy
            return np.sqrt(np.mean(filtered ** 2))
        except:
            return np.sqrt(np.mean(audio ** 2)) * 0.5  # Fallback: assume 50% in band
    
    def _energy_db(self, audio: np.ndarray, return_linear: bool = False) -> float:
        """Calculate RMS energy in dB or linear domain.
        
        Args:
            audio: Audio samples
            return_linear: If True, return RMS (linear). If False, return dB.
        
        Returns:
            Energy in dB (default) or RMS (if return_linear=True)
        """
        try:
            rms = np.sqrt(np.mean(audio ** 2))
            if return_linear:
                return rms
            else:
                return 20 * np.log10(rms + 1e-10)
        except:
            return 0.0 if not return_linear else 1e-10
    
    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """Wiener entropy: flat=1 (white noise), structured=0 (voiced). Voice has low flatness."""
        try:
            audio_safe = np.clip(audio, -1.0, 1.0).astype(np.float32)
            freqs, pxx = welch(audio_safe, fs=self._sample_rate, nperseg=min(512, len(audio_safe)))
            
            # Protect against invalid values
            pxx = np.maximum(pxx, 1e-12)
            pxx = np.nan_to_num(pxx, nan=1e-12, posinf=1.0, neginf=1e-12)
            
            geom_mean = np.exp(np.mean(np.log(pxx)))
            arith_mean = np.mean(pxx)
            
            if np.isnan(geom_mean) or np.isnan(arith_mean) or np.isinf(arith_mean):
                return 0.5
            
            flatness = geom_mean / (arith_mean + 1e-12)
            flatness = np.nan_to_num(flatness, nan=0.5, posinf=0.5, neginf=0.5)
            
            return np.clip(flatness, 0.0, 1.0)
        except:
            return 0.5

    def _autocorrelation_periodicity(self, audio: np.ndarray) -> float:
        """Autocorrelation-based periodicity detection. Voice has peaks, noise is flat."""
        try:
            std = np.std(audio)
            if std < 1e-8:
                return 0.4
            audio_norm = (audio - np.mean(audio)) / (std + 1e-10)
            
            # Autocorrelation at pitch period (60-200 Hz typical for speech)
            autocor = np.correlate(audio_norm, audio_norm, mode='full')
            autocor = autocor[len(autocor)//2:]
            autocor = autocor / (autocor[0] + 1e-10)
            
            # Search pitch range (2-18 samples @ 44kHz = ~100-400 Hz)
            pitch_range = autocor[2:18] if len(autocor) > 18 else autocor[2:]
            periodicity = np.max(pitch_range) if len(pitch_range) > 0 else 0.0
            
            return np.clip(periodicity, 0.0, 1.0)
        except:
            return 0.3

    def _envelope_variance(self, audio: np.ndarray) -> float:
        """Detects syllabic modulation: voice has varying envelope, static is flat.
        Uses Hilbert transform to extract amplitude envelope."""
        try:
            from scipy.signal import hilbert
            
            # Get instantaneous amplitude via Hilbert transform
            analytical = hilbert(audio)
            envelope = np.abs(analytical)
            
            # Normalize envelope
            env_mean = np.mean(envelope)
            if env_mean < 1e-10:
                return 0.0
            
            # Variance in normalized envelope (voice > 0.005, static < 0.001)
            normalized_env = envelope / (env_mean + 1e-10)
            variance = np.var(normalized_env)
            
            return np.clip(variance, 0.0, 1.0)
        except:
            return 0.005

    def _harmonicity_score(self, audio: np.ndarray) -> float:
        """Detects harmonic structure: voice has harmonics, noise is inharmonic.
        Returns 0-1 where higher = more harmonic (voice-like)."""
        try:
            freqs, pxx = welch(audio, fs=self._sample_rate, nperseg=min(512, len(audio)))
            
            if len(pxx) < 10:
                return 0.3
            
            # Find peaks in power spectrum (potential harmonics)
            threshold = np.mean(pxx) + np.std(pxx)
            peaks = np.where(pxx > threshold)[0]
            
            if len(peaks) < 2:
                return 0.0
            
            # Check if peaks are spaced harmonically (multiples of fundamental)
            peak_freqs = freqs[peaks]
            fundamental = peak_freqs[0] if len(peak_freqs) > 0 else 100
            
            if fundamental < 50:
                return 0.0
            
            # Count harmonics (peaks at ~2f, ~3f, ~4f, etc.)
            harmonic_count = 0
            for i in range(2, 5):  # Check 2nd, 3rd, 4th harmonics
                harmonic_freq = fundamental * i
                # Allow ±20% tolerance
                matches = np.sum((peak_freqs >= harmonic_freq * 0.8) & 
                                (peak_freqs <= harmonic_freq * 1.2))
                harmonic_count += matches
            
            # Harmonicity: ratio of detected harmonics (0-1)
            harmonicity = harmonic_count / 3.0  # max 3 harmonics checked
            return np.clip(harmonicity, 0.0, 1.0)
        except:
            return 0.3

    def _crest_factor(self, audio: np.ndarray) -> float:
        """Peak-to-RMS ratio: voice has defined peaks, noise is flat.
        Voice: 1.5-2.2, Static: 1.0-1.2"""
        try:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 1e-10:
                return 0.0
            
            peak = np.max(np.abs(audio))
            crest = peak / (rms + 1e-10)
            
            # Normalize: 1.0 = no peaks (noise), 2.5+ = strong peaks (voice)
            # Map to 0-1: crest 1.0 -> 0.0, crest 2.5 -> 1.0
            normalized = (crest - 1.0) / 1.5
            return np.clip(normalized, 0.0, 1.0)
        except:
            return 1.0

    def _zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Normalized zero-crossing rate. Voice typically 0.1-0.5; static > 0.5."""
        try:
            zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            return np.clip(zcr, 0.0, 1.0)
        except:
            return 0.3
    
    def _pitch_periodicity(self, audio: np.ndarray) -> float:
        """Enhanced periodicity detection for weak/noisy signals like USB radio.
        Autocorrelation-based: voice has peaks, noise doesn't. Returns 0-1 (higher = more voice-like)."""
        try:
            audio_safe = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            # Normalize (handle very quiet signals)
            std = np.std(audio_safe)
            if std < 1e-8:
                # Signal too quiet, be lenient
                return 0.4
            
            audio_norm = (audio_safe - np.mean(audio_safe)) / (std + 1e-10)
            
            # Autocorrelation with wider search for weak signals
            max_lag = min(len(audio_norm) // 2, 512)
            autocor = np.correlate(audio_norm, audio_norm, mode='full')
            autocor = autocor[len(autocor)//2:]
            
            # Protect against division by zero
            autocor_norm = np.abs(autocor[0]) + 1e-10
            if np.isnan(autocor_norm) or np.isinf(autocor_norm):
                return 0.35
            
            autocor = autocor / autocor_norm
            
            # Wider pitch range for robustness: 50-400 Hz (1-18 samples @ 44kHz)
            # Also check 400-800 Hz for harmonics
            pitch_range_1 = autocor[1:18] if len(autocor) > 18 else autocor[1:]
            pitch_range_2 = autocor[18:36] if len(autocor) > 36 else []
            
            peak1 = np.max(pitch_range_1) if len(pitch_range_1) > 0 else 0.0
            peak2 = np.max(pitch_range_2) if len(pitch_range_2) > 0 else 0.0
            
            # Handle NaN peaks
            peak1 = np.nan_to_num(peak1, nan=0.0, posinf=0.0, neginf=0.0)
            peak2 = np.nan_to_num(peak2, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Combine peaks with bias toward fundamental frequency
            periodicity = max(peak1 * 0.7 + peak2 * 0.3, peak1)
            
            return np.clip(periodicity, 0.0, 1.0)
        except:
            return 0.35

    # ============================================================================
    # Audio Processing Methods
    # ============================================================================
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio processing filters based on user settings."""
        processed = audio.copy()
        
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
    
    def _apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter 300-3000 Hz for SSB speech."""
        try:
            # Butterworth bandpass
            sos = butter(4, [300, 3000], btype='band', fs=self._sample_rate, output='sos')
            filtered = sosfiltfilt(sos, audio)
            return filtered
        except:
            return audio
    
    def _apply_enhanced_denoise(self, audio: np.ndarray, strength: int) -> np.ndarray:
        """Apply enhanced noise reduction."""
        if not HAS_NOISEREDUCE:
            return audio
        
        try:
            # Map strength 1-10 to noise reduction parameters
            # Higher strength = more aggressive denoising
            prop_decrease = 0.4 + (strength / 10.0) * 0.5  # 0.4 to 0.9
            
            # Use noisereduce with aggressive settings
            reduced = nr.reduce_noise(
                y=audio,
                sr=self._sample_rate,
                prop_decrease=prop_decrease,
                n_fft=512,
                stationary=True
            )
            
            return reduced
        except:
            return audio
    
    def _extract_voice_only(self, audio: np.ndarray) -> np.ndarray:
        """Extract only clear voice segments using VAD, silence the rest."""
        try:
            # Process in chunks
            chunk_size = BLOCK_SIZE
            voice_audio = np.zeros_like(audio)
            
            for i in range(0, len(audio), chunk_size):
                end = min(i + chunk_size, len(audio))
                chunk = audio[i:end]
                
                # Use VAD to detect if this chunk is voice (score > 50 = likely voice)
                score = self._detect_voice(chunk)
                if score > 50:  # 50+ = voice territory
                    voice_audio[i:end] = chunk
            
            return voice_audio
        except:
            return audio
    
    def _remove_silence_gaps(self, audio: np.ndarray) -> np.ndarray:
        """Remove silent gaps between speech bursts."""
        try:
            from scipy.signal import find_peaks
            
            # Detect silence threshold (10% of max amplitude)
            threshold = 0.1 * np.max(np.abs(audio))
            
            # Find regions above threshold
            above_threshold = np.abs(audio) > threshold
            
            # Remove small isolated peaks (noise)
            chunk_size = int(0.05 * self._sample_rate)  # 50ms minimum
            
            # Simple approach: keep consecutive samples above threshold
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
        except:
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
            if confidence >= word_peak_threshold:
                self._last_word_peak_time = now
                if self._meter_debug:
                    print(f"[Word Peak] RAW {confidence:.0f}% >= {word_peak_threshold}% threshold → reset post-roll")
            
            # Calculate time since last detected word
            time_since_last_word = now - self._last_word_peak_time
            post_roll_remaining = max(0, post_roll_seconds - time_since_last_word)
            
            # Use post-roll to extend hangover (10 seconds of silence/decay after last word)
            if post_roll_remaining > 0 and self._recording:
                self._hangover_remaining = post_roll_remaining
            
            # DEBUG: Show state every ~0.5s
            if self._meter_debug and not hasattr(self, '_last_debug_print'):
                self._last_debug_print = now - 0.3  # Skip first one
            if self._meter_debug and (now - self._last_debug_print) > 0.5:
                if not self._recording:
                    print(f"[IDLE] RAW={confidence:.0f}% EMA={self._confidence_ema:.0f}% | Thresholds: Start={start_threshold}% (L{self._sensitivity_level}) | Noise floor={self._noise_floor_db:.1f}dB | Phase={self._learning_phase}")
                self._last_debug_print = now
            
            # ===== RECORDING START/CONTINUE/STOP LOGIC =====
            # v1.08.3: Requires 3.5+ seconds of sustained high confidence to prevent false positives
            # Voice confidence accumulator
            if confidence > 60:
                # Accumulate high-confidence time
                if self._last_high_confidence_time is None:
                    self._last_high_confidence_time = time.time()
                    self._voice_confidence_duration = 0.0
                else:
                    dt = time.time() - self._last_high_confidence_time
                    self._voice_confidence_duration += dt
                    self._last_high_confidence_time = time.time()
            else:
                # Reset accumulator when confidence drops
                self._voice_confidence_duration = 0.0
                self._last_high_confidence_time = None
            
            # Voice only counts as "detected" after sustained 3.5 seconds of high confidence
            # This prevents momentary noise spikes from triggering, but catches real transmissions
            voice_qualified = self._voice_confidence_duration >= self._voice_confirmation_threshold
            voice_detected = (voice_qualified) or (self._hangover_remaining > 0)
            snr_display = getattr(self, '_last_snr_db', -60.0)
            
            # START RECORDING (only after 3.5+ seconds of sustained voice confidence)
            if not self._recording and voice_detected and voice_qualified:
                if self._meter_debug:
                    print(f">>> RECORDING START <<< Score {confidence:.0f}/100 (sustained {self._voice_confidence_duration:.1f}s)")
                self._start_recording()
                self.status_bar.showMessage(f"Recording started | Voice detected (SNR={snr_display:.1f}dB)")
            
            # CONTINUE RECORDING
            elif self._recording:
                if voice_detected:
                    # Still within hangover window
                    self.status_bar.showMessage(
                        f"Recording... | Score: {confidence:.0f}/100 | Duration: {self._voice_confidence_duration:.1f}s | Hangover: {self._hangover_remaining:.2f}s"
                    )
                else:
                    # Hangover expired, stop recording
                    if self._meter_debug:
                        print(f">>> RECORDING STOP <<< Hangover expired")
                    self._finalize_recording()
                    self.status_bar.showMessage(f"Recording stopped | Hangover timeout")
            
            # Update voice indicator
            if voice_detected != self._voice_detected:
                self._voice_detected = voice_detected
                self._signals.voice.emit(voice_detected)


    def _start_recording(self):
        self._recording = True
        self._voice_started_at = time.time()
        self._voice_silence_at = None
        self._silence_timer_remaining = 0
        
        # Copy pre-roll buffer to recording buffer (must happen atomically)
        with self._lock:
            self._audio_buffer = list(self._ring_buffer)
        
        with self._audio_buffer_lock:
            # Keep audio buffer; callback will append to it
            pass
        
        self._signals.status.emit(
            f"Recording... | VAD: Voice | Pre-roll: {len(self._audio_buffer) * BLOCK_SIZE / self._sample_rate:.1f}s"
        )

    def _finalize_recording(self):
        self._recording = False
        self._voice_started_at = None
        self._voice_silence_at = None
        
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
            threading.Thread(target=self._encode_and_upload, args=(audio_data,), daemon=True).start()
            self._signals.status.emit("Uploading...")

    def _encode_and_upload(self, audio_data: np.ndarray):
        try:
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
            
            # Upload to Discord
            if self._webhook_url:
                self._upload_to_discord(filepath)
                self._signals.status.emit(f"Uploaded | Frequency: {self._frequency}")
        except Exception as e:
            self._signals.error.emit(f"Encoding/upload failed: {e}")

    def _upload_to_discord(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                # Use file1 as the field name for the attachment
                files = {'file1': (os.path.basename(filepath), f)}
                role_part = f"<@&{self._webhook_role_id}>" if self._webhook_role_id else ""
                
                # Discord webhook requires JSON to be sent as payload_json parameter
                payload = {
                    'content': f"{role_part} {self._webhook_message}",
                    'embeds': [{
                        'title': f"{self._frequency} kHz",
                        'color': 3066993,
                        'timestamp': datetime.datetime.utcnow().isoformat()
                    }]
                }
                
                # Send as multipart form-data with payload_json
                data = {'payload_json': json.dumps(payload)}
                requests.post(self._webhook_url, data=data, files=files, timeout=30)
        except Exception as e:
            self._signals.error.emit(f"Discord upload failed: {e}")

    def _on_level(self, db):
        pass

    def _on_voice(self, detected):
        if detected:
            self.voice_label.setText("voice")
            self.voice_label.setStyleSheet("color: green; font-weight: bold; font-size: 12px;")
        else:
            self.voice_label.setText("static")
            self.voice_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")

    def _on_status(self, msg):
        self.status_bar.showMessage(msg)

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def _save_config(self):
        """Save all user settings to a JSON configuration file."""
        config = {
            "device": self.device_combo.currentText(),
            "sensitivity": self.sens_slider.value(),
            "frequency": self.freq_display.text(),
            "webhook_url": self.webhook_edit.text(),
            "custom_message": self.message_edit.text(),
            "role_id": self._webhook_role_id,
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
                
                if "webhook_url" in config:
                    self.webhook_edit.setText("✓ Webhook configured (" + config["webhook_url"][-20:] + ")")
                    self.webhook_edit.setReadOnly(True)
                    self._webhook_url = config["webhook_url"]
                
                if "custom_message" in config:
                    self.message_edit.setText(config["custom_message"])
                    self._webhook_message = config["custom_message"]
                
                if "role_id" in config:
                    self._webhook_role_id = config["role_id"]
                
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
