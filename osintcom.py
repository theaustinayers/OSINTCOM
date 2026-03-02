"""
OSINTCOM: Cross-platform Voice Activity Detection and Recording Tool

Features:
- Listens to Mic In or Loopback WASAPI (Windows)
- Enhanced VAD for SSB (HF radio, USB mode) with 5 sensitivity levels
- Records 5 seconds before and 10 seconds after VAD (countdown resets on new VAD)
- Saves audio as .wav file
- User input for Frequency, Discord Webhook, custom Discord message (role ID for pings)
- Start/Stop/File Location controls
- Optional WAV denoiser filter
"""


import os
import sys
import threading
import queue
import time
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# Platform-specific imports for WASAPI loopback

CONFIG_FILE = "osintcom_config.json"

class OSINTCOMGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSINTCOM")
        self.geometry("520x560")
        self.resizable(False, False)
        self.configure(bg="#232435")
        self.style = ttk.Style(self)
        self.set_dark_theme()
        self.create_widgets()
        self.load_config()

    def set_dark_theme(self):
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#232435")
        self.style.configure("TLabel", background="#232435", foreground="#bfc7e0", font=("Segoe UI", 11))
        self.style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground="#bfc7e0", background="#232435")
        self.style.configure("Section.TLabelframe", background="#232435", foreground="#bfc7e0", font=("Segoe UI", 13, "bold"), borderwidth=2, relief="groove")
        self.style.configure("Section.TLabelframe.Label", foreground="#bfc7e0", background="#232435", font=("Segoe UI", 13, "bold"))
        self.style.configure("TButton", background="#35364a", foreground="#bfc7e0", font=("Segoe UI", 11, "bold"), borderwidth=2)
        self.style.map("TButton", background=[("active", "#2d2e3e")])
        self.style.configure("TEntry", fieldbackground="#232435", foreground="#bfc7e0", background="#232435", font=("Segoe UI", 11))
        self.style.configure("TCheckbutton", background="#232435", foreground="#bfc7e0")
        self.style.configure("TCombobox", fieldbackground="#232435", background="#232435", foreground="#232435", font=("Segoe UI", 11))
        self.style.configure("Horizontal.TScale", background="#232435")

    def create_widgets(self):
        # Header
        ttk.Label(self, text="OSINTCOM", style="Header.TLabel").pack(pady=(16, 8))

        # Audio Interface Section
        audio_frame = ttk.Labelframe(self, text="Audio Interface", style="Section.TLabelframe")
        audio_frame.pack(fill="x", padx=16, pady=(0, 10))

        self.input_var = tk.StringVar()
        self.input_devices = self._get_input_devices()

        # Device row with icon
        device_row = ttk.Frame(audio_frame)
        device_row.pack(fill="x", padx=12, pady=(12, 6))
        self.device_icon_label = tk.Label(device_row, text="🔊", font=("Segoe UI", 14), bg="#232435", fg="#bfc7e0")
        self.device_icon_label.pack(side="left", padx=(0, 8))
        self.input_combo = ttk.Combobox(device_row, textvariable=self.input_var, values=self.input_devices, state="readonly", font=("Segoe UI", 12), width=36)
        self.input_combo.pack(side="left", fill="x", expand=True, padx=(0,8))
        if self.input_devices:
            self.input_var.set(self.input_devices[0])

        # Real-time audio meter (placeholder)
        meter_frame = ttk.Frame(audio_frame)
        meter_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.meter_canvas = tk.Canvas(meter_frame, width=220, height=18, bg="#232435", highlightthickness=0)
        self.meter_canvas.pack(side="left", padx=(0,8))
        self.db_var = tk.StringVar(value="-60.0 dB")
        db_label = tk.Label(meter_frame, textvariable=self.db_var, font=("Consolas", 13), bg="#232435", fg="#e0e0e0", relief="sunken", width=10)
        db_label.pack(side="left")

        # Voice/static indicator (color-coded)
        voice_frame = ttk.Frame(audio_frame)
        voice_frame.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Label(voice_frame, text="Voice:").pack(side="left")
        self.voice_var = tk.StringVar(value="static")
        self.voice_indicator = tk.Label(voice_frame, textvariable=self.voice_var, font=("Segoe UI", 12, "bold"), width=8, bg="#232435", fg="#e74c3c")
        self.voice_indicator.pack(side="left", padx=(8,0))

        # Sensitivity slider

        # Sensitivity slider
        sens_frame = ttk.Frame(audio_frame)
        sens_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Label(sens_frame, text="Sensitivity:", font=("Segoe UI", 11)).pack(side="left")
        self.sensitivity_var = tk.IntVar(value=3)
        sens_slider = ttk.Scale(sens_frame, from_=1, to=5, orient="horizontal", variable=self.sensitivity_var, length=180)
        sens_slider.pack(side="left", padx=8)
        self.sens_value_label = ttk.Label(sens_frame, text="3", font=("Segoe UI", 11, "bold"))
        self.sens_value_label.pack(side="left", padx=8)
        ttk.Label(sens_frame, text="- Balanced (Default)", font=("Segoe UI", 10)).pack(side="left")
        sens_slider.bind("<Motion>", lambda e: self.sens_value_label.config(text=str(int(self.sensitivity_var.get()))))

        # Discord Webhook Section
        discord_frame = ttk.Labelframe(self, text="Discord Webhook", style="Section.TLabelframe")
        discord_frame.pack(fill="x", padx=16, pady=(0, 10))

        ttk.Label(discord_frame, text="Frequency:").pack(anchor="w", padx=12, pady=(12, 0))
        self.freq_var = tk.StringVar()
        ttk.Entry(discord_frame, textvariable=self.freq_var, font=("Segoe UI", 12)).pack(fill="x", padx=12, pady=4)

        freq_hint = ttk.Label(discord_frame, text="e.g. 8992 kHz or 14.295 MHz", font=("Segoe UI", 10))
        freq_hint.pack(anchor="w", padx=12, pady=(0, 6))

        self.webhook_var = tk.StringVar()
        self.message_var = tk.StringVar()
        webhook_row = ttk.Frame(discord_frame)
        webhook_row.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Entry(webhook_row, textvariable=self.webhook_var, font=("Segoe UI", 12), width=18).pack(side="left", fill="x", expand=True)
        ttk.Button(webhook_row, text="Webhook URL", width=16).pack(side="left", padx=8)
        ttk.Entry(webhook_row, textvariable=self.message_var, font=("Segoe UI", 12), width=18).pack(side="left", fill="x", expand=True)
        ttk.Button(webhook_row, text="Customize Message", width=18).pack(side="left", padx=8)

        self.role_var = tk.StringVar()
        ttk.Entry(discord_frame, textvariable=self.role_var, font=("Segoe UI", 12)).pack(fill="x", padx=12, pady=(0, 6))

        self.webhook_status = tk.StringVar(value="Webhook: (not set)")
        ttk.Label(discord_frame, textvariable=self.webhook_status, font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(0, 10))

        # Controls Section
        controls_frame = ttk.Labelframe(self, text="Controls", style="Section.TLabelframe")
        controls_frame.pack(fill="x", padx=16, pady=(0, 10))

        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(padx=12, pady=12)
        for text, cmd, state in [
            ("Start", self.start_recording, "normal"),
            ("Stop", self.stop_recording, "disabled"),
            ("File Location", self.browse_file, "normal"),
            ("Save Settings", self.save_config, "normal")]:
            btn = ttk.Button(btn_frame, text=text, command=cmd, width=16)
            btn.pack(side="left", padx=12, ipadx=4, ipady=4)
            if state == "disabled":
                btn.state(["disabled"])

        self.file_var = tk.StringVar(value="C:/Users/theau/Documents/OSINTCOM Recordings")
        ttk.Label(controls_frame, text="Save to: ", font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(0, 2))
        ttk.Label(controls_frame, textvariable=self.file_var, font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(0, 8))

        # Status bar (real-time system status)
        self.status_var = tk.StringVar(value="Ready | VAD: Idle | Recording: Stopped")
        status_bar = tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 10, "bold"), bg="#232435", fg="#bfc7e0", anchor="w", relief="sunken")
        status_bar.pack(fill="x", side="bottom", ipady=4)

    def _get_input_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    input_devices.append(f"{dev['name']} (ID {idx})")
            # Add WASAPI loopback for Windows
            if sys.platform == "win32":
                input_devices.append("Loopback WASAPI (Windows)")
            return input_devices
        except Exception as e:
            return ["Mic In"]

    def browse_file(self):
        filename = filedialog.askdirectory(title="Select Output Folder")
        if filename:
            self.file_var.set(filename)

    def start_recording(self):
        self.status_var.set("Recording started (functionality to be implemented)")
        messagebox.showinfo("Start", "Recording started (functionality to be implemented)")

    def stop_recording(self):
        self.status_var.set("Recording stopped (functionality to be implemented)")
        messagebox.showinfo("Stop", "Recording stopped (functionality to be implemented)")

    def save_config(self):
        """Save all user settings to a JSON configuration file."""
        config = {
            "device": self.input_var.get(),
            "sensitivity": self.sensitivity_var.get(),
            "frequency": self.freq_var.get(),
            "webhook_url": self.webhook_var.get(),
            "custom_message": self.message_var.get(),
            "role_id": self.role_var.get(),
            "file_location": self.file_var.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.status_var.set("Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_config(self):
        """Load settings from configuration file if it exists."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                
                # Load all settings
                if "device" in config:
                    self.input_var.set(config["device"])
                if "sensitivity" in config:
                    self.sensitivity_var.set(config["sensitivity"])
                if "frequency" in config:
                    self.freq_var.set(config["frequency"])
                if "webhook_url" in config:
                    self.webhook_var.set(config["webhook_url"])
                if "custom_message" in config:
                    self.message_var.set(config["custom_message"])
                if "role_id" in config:
                    self.role_var.set(config["role_id"])
                if "file_location" in config:
                    self.file_var.set(config["file_location"])
                
                self.status_var.set("Settings loaded from config")
            except Exception as e:
                print(f"Warning: Could not load config file: {str(e)}")


def main():
    app = OSINTCOMGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
