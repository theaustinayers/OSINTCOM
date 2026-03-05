#!/usr/bin/env python3
"""
OSINTCOM VAD Diagnostic Tool
Test voice detection in real-time with full debug output
"""

import numpy as np
import sounddevice as sd
import collections
import time

try:
    import torch
    from silero_vad import load_silero_vad
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    print("[WARNING] Silero VAD not available - heuristic mode only")

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DURATION = 10  # seconds to listen

class VADTester:
    def __init__(self):
        self.noise_floor_rms = 0.001
        self.snr_history = collections.deque(maxlen=300)
        self.silero_model = None
        self.silero_ready = False
        self.adaptive_cv_min = 0.25
        self.adaptive_cv_max = 0.60
        
        # Load Silero if available
        if SILERO_AVAILABLE:
            print("[Silero] Loading model...")
            try:
                self.silero_model = load_silero_vad(onnx=False, force_onnx=False)
                self.silero_ready = True
                print("[Silero] Model loaded successfully!")
            except Exception as e:
                print(f"[Silero] Failed to load: {str(e)}")
                self.silero_ready = False
    
    def detect_voice(self, audio):
        """Test VAD detection"""
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        # SNR calculation
        rms = np.sqrt(np.mean(audio ** 2))
        snr_db = 20 * np.log10((rms + 1e-10) / (self.noise_floor_rms + 1e-10))
        self.snr_history.append(snr_db)
        
        # Rolling percentile
        if len(self.snr_history) > 10:
            snr_percentile = np.percentile(list(self.snr_history), 20)
        else:
            snr_percentile = snr_db
        
        # Modulation check
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
        else:
            cv = 0.0
        
        # Silero detection
        silero_prob = 0.0
        if self.silero_ready and self.silero_model is not None:
            try:
                # Resample to 16kHz
                if SAMPLE_RATE != 16000:
                    resample_factor = SAMPLE_RATE / 16000
                    indices = np.arange(len(audio)) / resample_factor
                    audio_resampled = np.interp(indices, np.arange(len(audio)), audio)
                else:
                    audio_resampled = audio
                
                audio_tensor = torch.from_numpy(audio_resampled).float()
                silero_prob = self.silero_model(audio_tensor, 16000).item()
            except Exception as e:
                print(f"[Silero ERROR] {str(e)}")
        
        return {
            'rms_db': 20 * np.log10(rms + 1e-10),
            'snr_db': snr_db,
            'snr_percentile': snr_percentile,
            'cv': cv,
            'has_modulation': self.adaptive_cv_min < cv < self.adaptive_cv_max,
            'silero_prob': silero_prob
        }

def audio_callback(indata, frames, time_info, status):
    """Audio callback for stream"""
    if status:
        print(f"[AUDIO ERROR] {status}")
    
    audio = indata[:, 0]
    results = tester.detect_voice(audio)
    
    # Display results
    print(f"\n[FRAME] RMS={results['rms_db']:.1f}dB | SNR={results['snr_percentile']:.1f}dB | CV={results['cv']:.2f} | Modulation={results['has_modulation']}", end="")
    
    if tester.silero_ready:
        print(f" | Silero={results['silero_prob']:.2f}", end="")
    
    print()

if __name__ == "__main__":
    print("=" * 70)
    print("OSINTCOM VAD DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Block size: {BLOCK_SIZE} samples")
    print(f"Listening for {DURATION} seconds...\n")
    
    if not SILERO_AVAILABLE:
        print("[WARNING] Silero VAD not installed - install with: pip install silero-vad torch torchaudio")
    else:
        print("[INFO] Silero VAD available for enhanced detection")
    
    print("\nCV Range (natural speech): {:.2f} - {:.2f}".format(0.25, 0.60))
    print("SNR Thresholds: 6dB (open), 2dB (hangover), 0dB (recording)\n")
    
    tester = VADTester()
    
    print("Speak now! Debug output shows RMS, SNR, CV, Modulation detection, Silero confidence...\n")
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
            time.sleep(DURATION)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
