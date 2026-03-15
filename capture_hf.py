"""
OSINTCOM HF Capture Tool
========================
Records audio from any input device for a set duration and logs real-time
VAD scores to a companion CSV so the VAD thresholds can be tuned later.

Usage:
    python capture_hf.py

You will be prompted to:
  1. Pick an audio device
  2. Name the session (e.g. "8992_voice" or "8992_noise")
  3. Set duration in minutes (default 60)

Output (saved to ./hf_captures/):
    <session>_<timestamp>.wav      — full raw audio
    <session>_<timestamp>_vad.csv  — per-chunk VAD scores + RMS dB

Send both files so the VAD logic can be audited and thresholds tuned.
"""

import os
import sys
import time
import datetime
import threading
import collections
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import csv

# ── Optional: reuse the existing VAD engine if present ──────────────────────
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from osintcom_qt import OSINTCOMAPP  # noqa – just for path resolution
except Exception:
    pass

try:
    from scipy.signal import butter, sosfiltfilt, welch, find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 48000
BLOCK_SIZE    = 2048          # ~42 ms per chunk @ 48 kHz
CHANNELS      = 1
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "hf_captures")

# ── Sensitivity presets (mirrors osintcom_qt.py) ─────────────────────────────
SENSITIVITY_PRESETS = {
    1: {"confidence_start": 42, "confirm_min_ratio": 0.40, "confirm_min_run_chunks": 16,
        "confirm_window_seconds": 5.0,
        "formant_threshold_db": 3, "formant_prominence_db": 3.5,
        "flatness_gate": 0.80, "vb_flatness_hi": 0.82, "vb_flatness_lo": 0.40,
        "snr_gate_ramp_db": 6.0},
    2: {"confidence_start": 46, "confirm_min_ratio": 0.35, "confirm_min_run_chunks": 14,
        "confirm_window_seconds": 4.0,
        "formant_threshold_db": 4, "formant_prominence_db": 4.5,
        "flatness_gate": 0.76, "vb_flatness_hi": 0.65, "vb_flatness_lo": 0.35,
        "snr_gate_ramp_db": 7.0},
    3: {"confidence_start": 50, "confirm_min_ratio": 0.30, "confirm_min_run_chunks": 10,
        "confirm_window_seconds": 3.0,
        "formant_threshold_db": 5, "formant_prominence_db": 6.0,
        "flatness_gate": 0.72, "vb_flatness_hi": 0.50, "vb_flatness_lo": 0.25,
        "snr_gate_ramp_db": 8.0},
    4: {"confidence_start": 60, "confirm_min_ratio": 0.25, "confirm_min_run_chunks": 6,
        "confirm_window_seconds": 3.0,
        "formant_threshold_db": 7, "formant_prominence_db": 7.5,
        "flatness_gate": 0.68, "vb_flatness_hi": 0.45, "vb_flatness_lo": 0.20,
        "snr_gate_ramp_db": 8.0},
    5: {"confidence_start": 68, "confirm_min_ratio": 0.30, "confirm_min_run_chunks": 8,
        "confirm_window_seconds": 3.0,
        "formant_threshold_db": 8, "formant_prominence_db": 9.0,
        "flatness_gate": 0.62, "vb_flatness_hi": 0.40, "vb_flatness_lo": 0.15,
        "snr_gate_ramp_db": 9.0},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def list_devices():
    print("\n── Audio Devices ────────────────────────────────────────────────")
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = "  "
            print(f"{marker}[{idx:2d}] {d['name']}  (in:{d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)")
    print("──────────────────────────────────────────────────────────────\n")
    return devices


def pick_device(devices):
    while True:
        try:
            raw = input("Enter device ID to record from: ").strip()
            idx = int(raw)
            if 0 <= idx < len(devices) and devices[idx]['max_input_channels'] > 0:
                return idx
            print("  ✘ Invalid device — must be an input device.")
        except ValueError:
            print("  ✘ Enter a number.")


def bar(value_0_1, width=40):
    filled = int(value_0_1 * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ── Lightweight VAD scorer (mirrors osintcom_qt.py v1.36 logic) ──────────────

class LightVAD:
    """
    Stripped-down version of the main-app VAD for offline scoring.
    Returns a dict of component scores per audio chunk.
    Sensitivity-aware: mirrors osintcom_qt.py formant/voiceband/SNR settings.
    """
    def __init__(self, sample_rate=48000, sensitivity=3):
        self.sr = sample_rate
        self.noise_floor_rms = 0.001
        self._formant_buffer = collections.deque(maxlen=4)
        self._snr_history    = collections.deque(maxlen=300)
        self._sensitivity = max(1, min(5, sensitivity))
        self._preset = SENSITIVITY_PRESETS[self._sensitivity]

    def score(self, audio: np.ndarray) -> dict:
        if not HAS_SCIPY or len(audio) < 512:
            return self._null()

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        rms   = float(np.sqrt(np.mean(audio ** 2)))
        snr   = 20 * np.log10((rms + 1e-10) / (self.noise_floor_rms + 1e-10))
        self._snr_history.append(snr)
        rms_db = 20 * np.log10(rms + 1e-10)

        # SNR score (0-15) — matches main app idle state: threshold=0dB, full at +10dB
        snr_threshold = 0.0
        if snr < snr_threshold:
            snr_score = 0.0
        elif snr > snr_threshold + 10.0:
            snr_score = 15.0
        else:
            snr_score = ((snr - snr_threshold) / 10.0) * 15.0

        # Formant score (0-40)
        self._formant_buffer.append(audio)
        formant_audio = np.concatenate(list(self._formant_buffer))
        formant_score, formant_count = self._formants(formant_audio)

        # Voice band score (0-20)
        vb_score = self._voice_band(audio)

        # Flat-spectrum penalty (sensitivity-aware)
        if vb_score == 0.0:
            flat_penalties = {1: 0.65, 2: 0.55, 3: 0.40, 4: 0.25, 5: 0.15}
            formant_score *= flat_penalties.get(self._sensitivity, 0.40)

        # SNR spectral gate — sensitivity-aware floor & ramp (matches main app per-level)
        snr_floors = {1: -4.0, 2: -3.0, 3: -2.0, 4: -1.0, 5: 0.0}
        snr_ramps  = {1: 6.0, 2: 7.0, 3: 8.0, 4: 8.0, 5: 9.0}
        snr_floor = snr_floors.get(self._sensitivity, -2.0)
        snr_ramp  = snr_ramps.get(self._sensitivity, 8.0)
        if snr <= snr_floor:
            gate = 0.0
        elif snr >= snr_floor + snr_ramp:
            gate = 1.0
        else:
            gate = (snr - snr_floor) / snr_ramp
        formant_score *= gate
        vb_score      *= gate

        # Pitch score (0-15)
        pitch_raw = self._pitch(audio)
        pitch_score = (pitch_raw / 35.0) * 15.0

        # Modulation (0-10)
        mod_score = self._modulation(audio)

        confidence = snr_score + formant_score + vb_score + pitch_score + mod_score
        confidence = float(np.clip(confidence, 0.0, 100.0))

        return {
            "rms_db":        round(rms_db, 2),
            "snr_db":        round(snr, 2),
            "confidence":    round(confidence, 1),
            "snr_score":     round(snr_score, 1),
            "formant_score": round(formant_score, 1),
            "formant_count": formant_count,
            "vb_score":      round(vb_score, 1),
            "pitch_score":   round(pitch_score, 1),
            "mod_score":     round(mod_score, 1),
        }

    def _null(self):
        return {"rms_db": -60, "snr_db": -60, "confidence": 0.0,
                "snr_score": 0, "formant_score": 0, "formant_count": 0,
                "vb_score": 0, "pitch_score": 0, "mod_score": 0}

    def _formants(self, audio, max_pts=40.0):
        try:
            # In-band flatness pre-gate (matches main app)
            # Noise in 300-4000 Hz band is flat; real voice has formant peaks.
            flatness_gate = self._preset.get("flatness_gate", 0.72)
            try:
                sos_bp = butter(4, [300, 4000], btype='band', fs=self.sr, output='sos')
                band_audio = sosfiltfilt(sos_bp, audio)
                _, pxx = welch(band_audio, fs=self.sr, nperseg=min(512, len(band_audio)))
                pxx = np.maximum(pxx, 1e-12)
                in_band_flatness = np.exp(np.mean(np.log(pxx))) / np.mean(pxx)
                if in_band_flatness > flatness_gate:
                    return 0.0, 0  # Band is flat — noise, not voice
            except:
                pass

            window = np.hanning(len(audio))
            fft_r  = np.fft.rfft(audio * window)
            freqs  = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
            mag_db = 20 * np.log10(np.abs(fft_r) + 1e-9)

            mask   = (freqs > 300) & (freqs < 4000)
            f_db   = mag_db[mask]
            f_freq = freqs[mask]
            if len(f_db) == 0:
                return 0.0, 0

            # Sensitivity-aware peak detection (L3: median+5dB, prominence=6.0)
            threshold_db  = self._preset.get("formant_threshold_db", 5)
            prominence_db = self._preset.get("formant_prominence_db", 6.0)
            floor   = np.median(f_db) + threshold_db
            dist    = max(1, int(150 / (self.sr / len(audio))))
            peaks, _ = find_peaks(f_db, height=floor, distance=dist, prominence=prominence_db)

            if len(peaks) == 0:
                return 0.0, 0

            pf = np.sort(f_freq[peaks])
            clusters, cur = [], [pf[0]]
            for f in pf[1:]:
                if f - cur[-1] < 300:
                    cur.append(f)
                else:
                    clusters.append(cur); cur = [f]
            clusters.append(cur)
            n = len(clusters)

            if n >= 2:
                centers = [np.mean(c) for c in clusters]
                span = max(centers) - min(centers)
                if span < 400:
                    n = 1

            # Tightness-aware single-cluster scoring (matches main app)
            if n == 1:
                bw = max(clusters[0]) - min(clusters[0]) if len(clusters[0]) > 1 else 0.0
                if bw <= 150.0:
                    score = 22.0   # Extremely tight — weak SSB voice
                elif bw <= 250.0:
                    score = 17.0   # Tight — typical single formant
                else:
                    score = 10.0   # Loose single cluster
            elif n == 2:
                score = 22.0
            elif n == 3:
                score = 36.0
            elif n >= 4:
                score = max_pts
            else:
                score = 0.0
            return score, n
        except:
            return 0.0, 0

    def _voice_band(self, audio, max_pts=20.0):
        try:
            sos      = butter(4, [300, 3000], btype='band', fs=self.sr, output='sos')
            filtered = sosfiltfilt(sos, audio)
            _, pxx   = welch(filtered, fs=self.sr, nperseg=min(512, len(filtered)))
            pxx      = np.maximum(pxx, 1e-12)
            flatness = float(np.exp(np.mean(np.log(pxx))) / np.mean(pxx))
            # Sensitivity-aware thresholds (matching main app)
            # L3: >0.50 = noise, <0.25 = voice. HF radio noise is band-shaped,
            # not white, so tighter thresholds than 0.82 are needed.
            hi = self._preset.get("vb_flatness_hi", 0.50)
            lo = self._preset.get("vb_flatness_lo", 0.25)
            if flatness > hi:
                return 0.0
            elif flatness < lo:
                return max_pts
            return (hi - flatness) / (hi - lo) * max_pts
        except:
            return 0.0  # No credit on error (matches main app)

    def _pitch(self, audio):
        try:
            a = audio - np.mean(audio)
            if np.max(np.abs(a)) < 1e-10:
                return 0.0
            fft   = np.fft.fft(a, n=2 * len(a))
            ac    = np.real(np.fft.ifft(fft * np.conj(fft))[:len(a)])
            ac   /= ac[0] + 1e-10
            lo    = max(10, int(self.sr / 250))
            hi    = min(len(ac) - 1, int(self.sr / 85))
            if hi <= lo:
                return 0.0
            peak = float(np.max(ac[lo:hi]))
            # v1.26: SSB compresses voice — max observed autocorr on S8/S9 = 0.298.
            # Old threshold (>=0.45 full) was never reachable on SSB.
            # New: 0 at <=0.08, full 35pts at >=0.25.
            if peak >= 0.25:
                return 35.0
            if peak <= 0.08:
                return 0.0
            return (peak - 0.08) / 0.17 * 35.0
        except:
            return 0.0

    def _modulation(self, audio, max_pts=10.0):
        try:
            cs = len(audio) // 10
            if cs < 50:
                return 5.0
            rms_vals = [np.sqrt(np.mean(audio[i*cs:(i+1)*cs]**2)) for i in range(10)]
            arr  = np.array(rms_vals, dtype=np.float32)
            mn   = np.mean(arr)
            cv   = float(np.std(arr) / mn) if mn > 1e-10 else 0.0
            if cv < 0.05:   return 0.0
            if cv < 0.15:   return cv / 0.15 * 4.0
            if cv < 0.50:   return 4.0 + (cv - 0.15) / 0.35 * 6.0
            if cv < 0.70:   return 10.0 - (cv - 0.50) / 0.20 * 5.0
            return max(0.0, 5.0 - (cv - 0.70) / 0.30 * 5.0)
        except:
            return 5.0


# ── Main capture loop ────────────────────────────────────────────────────────

def run_capture(device_idx: int, session_name: str, duration_min: float, sensitivity: int = 3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts        = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    wav_path  = os.path.join(OUTPUT_DIR, f"{session_name}_{ts}.wav")
    csv_path  = os.path.join(OUTPUT_DIR, f"{session_name}_{ts}_vad.csv")

    device_info = sd.query_devices(device_idx)
    sr          = int(device_info['default_samplerate'])
    total_sec   = duration_min * 60.0

    vad         = LightVAD(sample_rate=sr, sensitivity=sensitivity)
    audio_chunks = []
    csv_rows     = []
    lock         = threading.Lock()

    # ── Learn noise floor for 3 seconds ──
    print(f"\n  Learning noise floor (3s) — stay quiet / no transmissions...")
    learn_chunks = []

    def _learn_cb(indata, frames, t, status):
        learn_chunks.append(indata[:, 0].copy())

    with sd.InputStream(device=device_idx, channels=CHANNELS,
                        samplerate=sr, blocksize=BLOCK_SIZE,
                        callback=_learn_cb):
        time.sleep(3.0)

    if learn_chunks:
        all_learn = np.concatenate(learn_chunks)
        vad.noise_floor_rms = float(np.median([
            np.sqrt(np.mean(c**2)) for c in learn_chunks
        ])) * 0.9
        nf_db = 20 * np.log10(vad.noise_floor_rms + 1e-10)
        print(f"  Noise floor: {nf_db:.1f} dB  (RMS={vad.noise_floor_rms:.6f})\n")

    # ── Recording callback ──
    chunk_index = [0]
    start_time  = [None]

    def _record_cb(indata, frames, t, status):
        chunk = indata[:, 0].copy()
        with lock:
            audio_chunks.append(chunk)

        scores = vad.score(chunk)
        elapsed = time.time() - start_time[0] if start_time[0] else 0.0

        row = {
            "chunk":         chunk_index[0],
            "elapsed_s":     round(elapsed, 3),
            "rms_db":        scores["rms_db"],
            "snr_db":        scores["snr_db"],
            "confidence":    scores["confidence"],
            "snr_score":     scores["snr_score"],
            "formant_score": scores["formant_score"],
            "formant_count": scores["formant_count"],
            "vb_score":      scores["vb_score"],
            "pitch_score":   scores["pitch_score"],
            "mod_score":     scores["mod_score"],
        }
        csv_rows.append(row)
        chunk_index[0] += 1

    # ── Print header ──
    print(f"  Session : {session_name}")
    print(f"  Device  : [{device_idx}] {device_info['name']} @ {sr} Hz")
    print(f"  VAD     : L{sensitivity} sensitivity")
    print(f"  Duration: {duration_min:.0f} min  ({total_sec:.0f} s)")
    print(f"  WAV     : {wav_path}")
    print(f"  CSV     : {csv_path}")
    print(f"\n  Recording... (Ctrl+C to stop early)\n")
    print(f"  {'Elapsed':>8}  {'Remain':>8}  Level                       Conf   SNR")
    print(f"  {'':─<70}")

    start_time[0] = time.time()
    last_print     = [0.0]

    try:
        with sd.InputStream(device=device_idx, channels=CHANNELS,
                            samplerate=sr, blocksize=BLOCK_SIZE,
                            callback=_record_cb):
            while True:
                elapsed  = time.time() - start_time[0]
                remaining = max(0, total_sec - elapsed)

                if elapsed - last_print[0] >= 0.5:
                    last_print[0] = elapsed

                    # Latest scores for display
                    if csv_rows:
                        r   = csv_rows[-1]
                        lvl = np.clip((r["rms_db"] + 60) / 60.0, 0.0, 1.0)
                        conf = r["confidence"]
                        snr  = r["snr_db"]
                        conf_color = ""
                        if conf >= 55:
                            conf_str = f"\033[92m{conf:5.1f}%\033[0m"   # green
                        elif conf >= 35:
                            conf_str = f"\033[93m{conf:5.1f}%\033[0m"   # yellow
                        else:
                            conf_str = f"\033[90m{conf:5.1f}%\033[0m"   # grey
                        print(
                            f"\r  {int(elapsed)//60:02d}:{int(elapsed)%60:02d}  "
                            f"  {int(remaining)//60:02d}:{int(remaining)%60:02d}  "
                            f"{bar(lvl, 28)}  {conf_str}  {snr:+6.1f}dB",
                            end="", flush=True
                        )

                if elapsed >= total_sec:
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n  Stopped early by user.")

    print(f"\n\n  Saving WAV...")
    with lock:
        all_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])

    if len(all_audio) > 0:
        int16 = (np.clip(all_audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav.write(wav_path, sr, int16)
        duration_actual = len(all_audio) / sr
        size_mb = os.path.getsize(wav_path) / 1024 / 1024
        print(f"  ✓ WAV saved: {duration_actual:.1f}s  ({size_mb:.1f} MB)")
    else:
        print("  ✘ No audio captured.")
        return

    print(f"  Saving VAD CSV ({len(csv_rows)} chunks)...")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  ✓ CSV saved: {csv_path}")

    # ── Quick summary + confirm-gate simulation ──
    confs = [r["confidence"] for r in csv_rows]
    if confs:
        print(f"\n  ── VAD Summary (scored at L{sensitivity}) ──────────────────────────────")
        print(f"  Chunks recorded : {len(confs)}")
        print(f"  Confidence  avg : {np.mean(confs):.1f}%")
        print(f"  Confidence  max : {np.max(confs):.1f}%")
        print(f"  Confidence  min : {np.min(confs):.1f}%")

        # Per-level analysis with confirm gate simulation
        for lvl in [3, 2, 1]:
            p = SENSITIVITY_PRESETS[lvl]
            thr = p["confidence_start"]
            above = sum(1 for c in confs if c >= thr)
            pct = above / len(confs) * 100

            # Max consecutive run
            max_run = 0; cur_run = 0
            for c in confs:
                if c >= thr:
                    cur_run += 1
                    if cur_run > max_run: max_run = cur_run
                else:
                    cur_run = 0

            # Worst sliding window ratio
            win_chunks = max(10, round(p["confirm_window_seconds"] * sr / BLOCK_SIZE))
            worst_ratio = 0.0; worst_win_run = 0
            for i in range(max(1, len(confs) - win_chunks + 1)):
                window = confs[i:i + win_chunks]
                hits = sum(1 for c in window if c >= thr)
                ratio = hits / len(window)
                wr = 0; cr2 = 0
                for c in window:
                    if c >= thr:
                        cr2 += 1
                        if cr2 > wr: wr = cr2
                    else:
                        cr2 = 0
                if ratio > worst_ratio:
                    worst_ratio = ratio; worst_win_run = wr

            # Would confirm gate fire?
            req_ratio = p["confirm_min_ratio"]
            req_run   = p["confirm_min_run_chunks"]
            passes_ratio = worst_ratio >= req_ratio
            passes_run   = max_run >= req_run
            would_fire   = passes_ratio and passes_run

            label = {1: "L1", 2: "L2", 3: "L3 (default)"}[lvl]
            gate_sym = "\033[91m✘ BLOCKED\033[0m" if not would_fire else "\033[92m⚠ WOULD FIRE\033[0m"
            print(f"  Frames >={thr}%    : {above:>4}  ({pct:>5.1f}%)  {label}")
            print(f"    Max run       : {max_run}  (gate needs {req_run})  {'PASS' if passes_run else 'FAIL'}")
            print(f"    Worst w-ratio : {worst_ratio:.1%}  (gate needs {req_ratio:.0%})  {'PASS' if passes_ratio else 'FAIL'}")
            print(f"    Confirm gate  : {gate_sym}")

        print(f"  ────────────────────────────────────────────────────────────")

    print(f"\n  Done. Share both files for VAD analysis:\n"
          f"    {wav_path}\n"
          f"    {csv_path}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

DEFAULT_DEVICE = 7   # DAX Audio RX 1 (FlexRadio SmartSDR)


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     OSINTCOM HF Capture Tool  — VAD Training Aid  v1.36     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    devices = list_devices()

    # Pre-select the DAX device if it exists, otherwise prompt normally
    default_ok = (
        0 <= DEFAULT_DEVICE < len(devices)
        and devices[DEFAULT_DEVICE]['max_input_channels'] > 0
    )
    if default_ok:
        d = devices[DEFAULT_DEVICE]
        print(f"  Default device: [{DEFAULT_DEVICE}] {d['name']}  ({int(d['default_samplerate'])} Hz)")
        raw = input(f"  Press Enter to use default, or enter a different device ID: ").strip()
        device_idx = int(raw) if raw else DEFAULT_DEVICE
    else:
        device_idx = pick_device(devices)

    print("\n  Session name examples:")
    print("    8992_voice_strong   8992_voice_weak   8992_noise   11175_voice")
    session = input("  Session name: ").strip().replace(" ", "_") or "session"

    dur_raw = input("  Duration in minutes [60]: ").strip()
    try:
        duration = float(dur_raw) if dur_raw else 60.0
    except ValueError:
        duration = 60.0

    sens_raw = input("  Sensitivity level 1-5 [3]: ").strip()
    try:
        sens = max(1, min(5, int(sens_raw))) if sens_raw else 3
    except ValueError:
        sens = 3
    print(f"  VAD sensitivity: L{sens}")

    run_capture(device_idx, session, duration, sensitivity=sens)


if __name__ == "__main__":
    main()
