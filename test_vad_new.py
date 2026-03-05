#!/usr/bin/env python3
"""
OSINTCOM VAD Diagnostic - Uses new 4-layer detection engine
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from osintcom_qt import OSINTCOMWindow
import sounddevice as sd
import time

# Create a minimal VAD instance
window = OSINTCOMWindow()
window._meter_debug = True
window._learning_phase = "periodic"

print("=" * 70)
print("OSINTCOM v1.11 VAD DIAGNOSTIC - 4-Layer Detection Test")
print("=" * 70)
print(f"Sample rate: {window._sample_rate} Hz")
print(f"Block size: 2048 samples (~46ms)")
print(f"Using: Pitch + Spectral Entropy + Zero-Crossing Rate + SNR")
print()
print(f"CV Range (natural speech): {window._adaptive_cv_min:.2f} - {window._adaptive_cv_max:.2f}")
print(f"SNR Thresholds: 6dB (open), 2dB (hangover), 0dB (recording)")
print()
print("Speak now! Listening for 10 seconds...")
print()

frames_recorded = 0
confidence_scores = []
start_time = time.time()

def audio_callback(indata, frames_count, time_info, status):
    global frames_recorded, confidence_scores
    
    if status:
        print(f"[Audio Status] {status}")
    
    audio_chunk = indata[:, 0].copy()
    
    # Run new 4-layer VAD detection
    confidence = window._detect_voice(audio_chunk)
    confidence_scores.append(confidence)
    frames_recorded += 1
    
    # Compute audio metrics
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Visual indicator
    if confidence > 50:
        indicator = "🟢 VOICE"
    elif confidence > 30:
        indicator = "🟡 POSSIBLE"
    else:
        indicator = "🔴 STATIC"
    
    print(f"[{frames_recorded:2d}] RMS={rms_db:7.1f}dB | Confidence={confidence:6.1f}% | {indicator}")

try:
    print("[Starting audio stream]")
    with sd.InputStream(channels=1, samplerate=window._sample_rate, blocksize=2048, 
                        callback=audio_callback, device=None):
        time.sleep(10)  # Record for 10 seconds
        
except KeyboardInterrupt:
    print("\n[User interrupted]")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
if confidence_scores:
    avg_confidence = np.mean(confidence_scores)
    max_confidence = np.max(confidence_scores)
    voice_frames = sum(1 for c in confidence_scores if c > 50)
    possible_frames = sum(1 for c in confidence_scores if 30 < c <= 50)
    
    print(f"Frames recorded: {frames_recorded}")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print(f"Max confidence: {max_confidence:.1f}%")
    print(f"Voice detected in {voice_frames} frames ({voice_frames/len(confidence_scores)*100:.1f}%)")
    print(f"Possible voice in {possible_frames} frames ({possible_frames/len(confidence_scores)*100:.1f}%)")
    print()
    if avg_confidence > 50:
        print("✅ VOICE DETECTION WORKING - Clear voice signal detected!")
    elif avg_confidence > 30:
        print("⚠️  MARGINAL DETECTION - Weak signal or high noise")
    else:
        print("❌ NO VOICE DETECTED - Check audio device/level")
