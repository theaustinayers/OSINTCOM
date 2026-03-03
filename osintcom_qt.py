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
from scipy.signal import butter, sosfiltfilt, welch, get_window
from scipy.fftpack import fft
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
PRE_ROLL_SECONDS = 5.0
VOICE_SILENCE_TAIL = 10.0
MIN_VOICE_DURATION = 0.5
MIN_RECORDING_DURATION = 1.5
VAD_SMOOTHING_WINDOW = 10

# Hybrid VAD: EAM Watcher algorithms + Adaptive Noise Floor Learning
# Learns actual noise floor during first 3 seconds, then sets thresholds intelligently
# Voice detection: Energy + Spectral Features for low SNR discrimination
SENSITIVITY_PRESETS = {
    # Level 1: Maximum sensitivity - catches faintest voices just above noise floor
    # db_above_noise_floor: how far above learned noise to trigger
    1: {"db_above_noise": 2.0, "autocorr": 0.05, "flatness": 0.70,
        "prominence": 6.0, "envelope_var": 0.0, "harmonicity": 0.0,
        "crest_factor": 0.0, "smoothing_req": 5, "hold_req": 2},
    # Level 2: Very sensitive - adds slight spectral filtering
    2: {"db_above_noise": 3.0, "autocorr": 0.08, "flatness": 0.65,
        "prominence": 7.0, "envelope_var": 0.0, "harmonicity": 0.0,
        "crest_factor": 0.2, "smoothing_req": 5, "hold_req": 2},
    # Level 3: Balanced (default) - some noise rejection
    3: {"db_above_noise": 4.0, "autocorr": 0.10, "flatness": 0.60,
        "prominence": 8.0, "envelope_var": 0.001, "harmonicity": 0.0,
        "crest_factor": 0.5, "smoothing_req": 6, "hold_req": 2},
    # Level 4: Strict - better noise rejection
    4: {"db_above_noise": 5.0, "autocorr": 0.15, "flatness": 0.50,
        "prominence": 9.0, "envelope_var": 0.002, "harmonicity": 0.5,
        "crest_factor": 0.8, "smoothing_req": 7, "hold_req": 3},
    # Level 5: Voice only - maximum static rejection
    5: {"db_above_noise": 6.0, "autocorr": 0.20, "flatness": 0.40,
        "prominence": 10.0, "envelope_var": 0.004, "harmonicity": 1.0,
        "crest_factor": 1.2, "smoothing_req": 8, "hold_req": 3},
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
        self._webhook_url = ""
        self._webhook_role_id = ""
        self._webhook_message = "EMERGENCY ACTION MESSAGE INCOMING"
        self._frequency = ""
        self._save_dir = os.path.join(os.path.expanduser("~"), "Documents", "OSINTCOM")
        self._signals = WorkerSignals()
        
        # Adaptive noise floor learning
        self._noise_floor_db = -25.0  # Initial estimate
        self._noise_samples = collections.deque(maxlen=100)  # Last 100 samples
        self._noise_learning_time = 3.0  # Learn for 3 seconds on startup
        self._noise_relearn_interval = 240.0  # Re-learn every 4 minutes (240 seconds)
        self._last_learning_time = time.time()
        self._last_relearn_time = time.time()
        self._learning_enabled = True
        self._learning_phase = "startup"  # "startup" or "periodic"
        
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
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.file_btn)
        controls_layout.addWidget(self.audio_settings_btn)
        controls_layout.addWidget(self.save_btn)
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

    def _on_sensitivity_changed(self, value):
        self._sensitivity_level = value
        self.sens_label.setText(SENSITIVITY_LABELS[value])

    def _open_webhook_dialog(self):
        dlg = WebhookDialog(self._webhook_url, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhook_url = dlg.get_url()
            self.webhook_edit.setText(self._webhook_url[:50] + "..." if len(self._webhook_url) > 50 else self._webhook_url)

    def _open_customize_dialog(self):
        dlg = WebhookCustomizeDialog(self._webhook_role_id, self._webhook_message, self)
        if dlg.exec_() == QDialog.Accepted:
            self._webhook_role_id = dlg.get_role_id()
            self._webhook_message = dlg.get_message_template()
            self.message_edit.setText(self._webhook_message[:40] + "..." if len(self._webhook_message) > 40 else self._webhook_message)

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
        
        # Buffer recording chunk outside main lock
        if self._recording:
            with self._audio_buffer_lock:
                self._audio_buffer.append(audio_chunk)

    def _detect_voice(self, audio: np.ndarray) -> bool:
        """Hybrid VAD with Adaptive Noise Floor Learning.
        Learns the actual noise floor during first 3 seconds, then detects voice intelligently."""
        if len(audio) < 512:
            return False
        try:
            preset = SENSITIVITY_PRESETS[self._sensitivity_level]
            debug = False  # Disable debug logging for production
            
            # Calculate energy
            energy_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
            
            # ===== NOISE FLOOR LEARNING (first 3 seconds) =====
            if self._learning_phase == "startup":
                elapsed = time.time() - self._last_learning_time
                if elapsed < self._noise_learning_time:
                    # During learning phase, collect quiet samples to estimate noise floor
                    if energy_db < -20:  # Likely noise, not voice
                        self._noise_samples.append(energy_db)
                    
                    # Update learned noise floor (median of collected samples)
                    if len(self._noise_samples) > 10:
                        self._noise_floor_db = np.median(list(self._noise_samples))
                        if debug and elapsed % 1.0 < 0.05:  # Print every ~1 second
                            print(f"[Startup Learning] Noise floor: {self._noise_floor_db:.1f} dB ({len(self._noise_samples)} samples)")
                else:
                    # Startup learning complete
                    self._learning_phase = "periodic"
                    if len(self._noise_samples) > 0:
                        self._noise_floor_db = np.median(list(self._noise_samples))
                    print(f"✓ Startup learning complete: Noise floor = {self._noise_floor_db:.1f} dB")
                    print(f"  Will trigger on signals > {self._noise_floor_db + preset['db_above_noise']:.1f} dB")
                    print(f"  Will re-learn every {self._noise_relearn_interval/60:.0f} minutes during silence")
                    self._last_relearn_time = time.time()
            
            # ===== PERIODIC NOISE FLOOR RE-LEARNING (every 4 minutes) =====
            # Only re-learn during silence (not recording), to adapt to propagation changes
            elif self._learning_phase == "periodic" and not self._recording:
                time_since_relearn = time.time() - self._last_relearn_time
                
                if time_since_relearn >= self._noise_relearn_interval:
                    # Start a new learning cycle (collect for 0.5 seconds)
                    if energy_db < -20:  # Likely noise
                        self._noise_samples.append(energy_db)
                    
                    if len(self._noise_samples) > 5:
                        old_floor = self._noise_floor_db
                        self._noise_floor_db = np.median(list(self._noise_samples))
                        self._noise_samples.clear()
                        self._last_relearn_time = time.time()
                        if self._noise_floor_db != old_floor:
                            print(f"[Periodic Re-learn] Noise floor updated: {old_floor:.1f} dB → {self._noise_floor_db:.1f} dB (propagation change)")
                            print(f"  Threshold now: {self._noise_floor_db + preset['db_above_noise']:.1f} dB")
                
                elif time_since_relearn > self._noise_relearn_interval + 5.0:
                    # If we've been over threshold for too long without collecting samples, reset
                    self._noise_samples.clear()
                    self._last_relearn_time = time.time()
                    if debug:
                        print(f"[Periodic Re-learn] Reset at {time_since_relearn:.0f}s for next cycle")
            
            # ===== VOICE DETECTION AFTER LEARNING =====
            # 1. ENERGY CHECK relative to learned noise floor
            threshold_db = self._noise_floor_db + preset["db_above_noise"]
            if energy_db < threshold_db:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Energy {energy_db:.1f} dB < threshold {threshold_db:.1f} dB (noise floor {self._noise_floor_db:.1f} + {preset['db_above_noise']:.1f})")
                return False
            
            # 2. SPECTRAL FLATNESS - reject pure noise (even if loud)
            # Pure noise = high flatness, voice = low flatness
            flatness = self._spectral_flatness(audio)
            if flatness > preset["flatness"]:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Flatness {flatness:.4f} > {preset['flatness']:.4f} (noise-like spectrum)")
                return False
            
            # 3. AUTOCORRELATION periodicty - voice has pitch structure
            autocorr = self._autocorrelation_periodicity(audio)
            if autocorr < preset["autocorr"]:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Autocorr {autocorr:.3f} < {preset['autocorr']:.3f} (no pitch structure)")
                return False
            
            # 4. ENVELOPE VARIANCE - voice modulates, static doesn't
            env_var = self._envelope_variance(audio)
            if env_var < preset["envelope_var"]:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Env.Var {env_var:.6f} < {preset['envelope_var']:.6f} (too steady)")
                return False
            
            # 5. HARMONICITY - voice has harmonic content
            harmonicity = self._harmonicity_score(audio)
            if harmonicity < preset["harmonicity"]:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Harmonicity {harmonicity:.2f} < {preset['harmonicity']:.2f} (inharmonic)")
                return False
            
            # 6. CREST FACTOR - voice has peaks, static is flat
            crest = self._crest_factor(audio)
            if crest < preset["crest_factor"]:
                if debug and self._learning_phase == "periodic":
                    print(f"VAD FAIL: Crest {crest:.2f} < {preset['crest_factor']:.2f} (flat)")
                return False
            
            if debug and self._learning_phase == "periodic":
                print(f"✓ VOICE: E={energy_db:.1f}dB (thr={threshold_db:.1f}) F={flatness:.3f} A={autocorr:.2f} EV={env_var:.5f} H={harmonicity:.1f} C={crest:.1f}")
            return True
        except Exception as e:
            print(f"VAD Exception: {e}")
            return False
    
    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """Wiener entropy: flat=1 (white noise), structured=0 (voiced). Voice has low flatness."""
        try:
            freqs, pxx = welch(audio, fs=self._sample_rate, nperseg=min(512, len(audio)))
            pxx = np.maximum(pxx, 1e-12)
            geom_mean = np.exp(np.mean(np.log(pxx)))
            arith_mean = np.mean(pxx)
            flatness = geom_mean / (arith_mean + 1e-12)
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
            # Normalize (handle very quiet signals)
            std = np.std(audio)
            if std < 1e-8:
                # Signal too quiet, be lenient
                return 0.4
            audio_norm = (audio - np.mean(audio)) / (std + 1e-10)
            
            # Autocorrelation with wider search for weak signals
            max_lag = min(len(audio) // 2, 512)
            autocor = np.correlate(audio_norm, audio_norm, mode='full')
            autocor = autocor[len(autocor)//2:]
            autocor = autocor / (autocor[0] + 1e-10)
            
            # Wider pitch range for robustness: 50-400 Hz (1-18 samples @ 44kHz)
            # Also check 400-800 Hz for harmonics
            pitch_range_1 = autocor[1:18] if len(autocor) > 18 else autocor[1:]
            pitch_range_2 = autocor[18:36] if len(autocor) > 36 else []
            
            peak1 = np.max(pitch_range_1) if len(pitch_range_1) > 0 else 0.0
            peak2 = np.max(pitch_range_2) if len(pitch_range_2) > 0 else 0.0
            
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
                
                # Use VAD to detect if this chunk is voice
                if self._detect_voice(chunk):
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
        db = self._peak_db
        self.audio_meter.set_level(db)
        self.meter_label.setText(f"{db:+.1f} dB")
        
        # Get current audio chunk for VAD (brief lock)
        current_chunk = None
        with self._lock:
            if self._ring_buffer:
                current_chunk = self._ring_buffer[-1]
        
        if current_chunk is not None:
            voice = self._detect_voice(current_chunk)
            self._voice_history.append(voice)
            detected = sum(self._voice_history) > len(self._voice_history) / 2
            
            if detected != self._voice_detected:
                self._voice_detected = detected
                self._signals.voice.emit(detected)
                if detected and not self._recording:
                    self._start_recording()
                elif not detected and self._recording:
                    self._voice_silence_at = time.time()
                    self._silence_timer_remaining = VOICE_SILENCE_TAIL
            
            # Check for silence timeout and update countdown
            if not detected and self._recording and self._voice_silence_at:
                elapsed = time.time() - self._voice_silence_at
                self._silence_timer_remaining = max(0, VOICE_SILENCE_TAIL - elapsed)
                
                if elapsed > VOICE_SILENCE_TAIL:
                    self._finalize_recording()
                else:
                    # Update status with red countdown
                    self.status_bar.showMessage(
                        f"Recording... | VAD: Silence | >>> TIMEOUT IN {self._silence_timer_remaining:.1f}s <<<"
                    )
            elif self._recording and detected:
                self.status_bar.showMessage(f"Recording... | VAD: Voice | Frequency: {self._frequency}")

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
                
                if "webhook_url" in config:
                    self.webhook_edit.setText(config["webhook_url"])
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
