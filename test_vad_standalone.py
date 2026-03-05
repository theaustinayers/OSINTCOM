#!/usr/bin/env python3
"""
OSINTCOM VAD Diagnostic - Standalone 4-layer detection test  
"""
import numpy as np
import sounddevice as sd
import time
import collections

# ============================================================================
# Standalone 4-Layer VAD Implementation
# ============================================================================

class VADTester:
    def __init__(self):
        self._sample_rate = 44100
        self._noise_floor_rms = 0.001
        self._snr_history = collections.deque(maxlen=300)
        self._meter_debug = True
        self._adaptive_cv_min = 0.25
        self._adaptive_cv_max = 0.60
        self._hangover_remaining = 0.0
        self._recording = False
    
    def _detect_pitch(self, audio):
        """Detect pitch/periodicity in voice range 85-250Hz."""
        try:
            if len(audio) < 512:
                return 0.0
            
            audio_work = audio - np.mean(audio)
            if np.max(np.abs(audio_work)) < 1e-10:
                return 0.0
            
            # Autocorrelation via FFT
            fft = np.fft.fft(audio_work, n=2*len(audio_work))
            power = fft * np.conj(fft)
            autocorr = np.fft.ifft(power)[0:len(audio_work)]
            autocorr = np.real(autocorr)
            autocorr = autocorr / autocorr[0]
            
            min_lag = max(10, int(self._sample_rate / 250))
            max_lag = min(len(autocorr)-1, int(self._sample_rate / 85))
            
            if max_lag <= min_lag:
                return 0.0
            
            autocorr_pitch = autocorr[min_lag:max_lag]
            peak_idx = np.argmax(autocorr_pitch) if len(autocorr_pitch) > 0 else 0
            peak_strength = autocorr_pitch[peak_idx] if len(autocorr_pitch) > 0 else 0.0
            
            if peak_strength > 0.60:
                return 25.0
            elif peak_strength < 0.30:
                return 0.0
            else:
                return (peak_strength - 0.30) / 0.30 * 25.0
        except:
            return 0.0
    
    def _estimate_spectral_entropy(self, audio):
        """Spectral entropy - low for voice, high for static."""
        try:
            if len(audio) < 512:
                return 0.0
            
            audio_work = audio - np.mean(audio)
            window = np.hamming(len(audio_work))
            audio_windowed = audio_work * window
            
            fft_data = np.fft.rfft(audio_windowed)
            power = np.abs(fft_data) ** 2
            power = power / (np.sum(power) + 1e-10)
            
            power_clipped = np.clip(power, 1e-10, 1.0)
            entropy = -np.sum(power_clipped * np.log(power_clipped + 1e-10))
            
            max_entropy = np.log(len(power))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            
            if normalized_entropy > 0.70:
                return 0.0
            elif normalized_entropy < 0.50:
                return 25.0
            else:
                return (0.70 - normalized_entropy) / 0.20 * 25.0
        except:
            return 0.0
    
    def _zero_crossing_rate_score(self, audio):
        """Zero-crossing rate - low for voice, high for static."""
        try:
            if len(audio) < 2:
                return 0.0
            
            audio_work = audio - np.mean(audio)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_work)))) / 2.0
            zcr = zero_crossings / len(audio_work)
            
            if zcr > 0.25:
                return 0.0
            elif zcr < 0.15:
                return 25.0
            else:
                return (0.25 - zcr) / 0.10 * 25.0
        except:
            return 0.0
    
    def _detect_voice(self, audio):
        """4-layer intelligent voice detection."""
        if len(audio) < 512:
            return 0.0
        
        try:
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            
            confidence = 0.0
            rms = np.sqrt(np.mean(audio ** 2))
            snr_db = 20 * np.log10((rms + 1e-10) / (self._noise_floor_rms + 1e-10))
            self._snr_history.append(snr_db)
            
            if len(self._snr_history) > 10:
                snr_percentile = np.percentile(list(self._snr_history), 20)
            else:
                snr_percentile = snr_db
            
            # Layer 1: SNR Gate
            if self._recording:
                snr_threshold = 0.0
            elif self._hangover_remaining > 0:
                snr_threshold = 2.0
            else:
                snr_threshold = 6.0
            
            snr_passes = snr_percentile > snr_threshold
            
            if not snr_passes:
                confidence = 0.0
            else:
                confidence = 20.0
                if snr_percentile > snr_threshold + 3.0:
                    confidence = 25.0
            
            # Layer 2: Pitch Detection
            if confidence > 0:
                pitch_score = self._detect_pitch(audio)
                confidence += pitch_score
            
            # Layer 3: Spectral Entropy
            if confidence > 0:
                entropy_score = self._estimate_spectral_entropy(audio)
                confidence += entropy_score
            
            # Layer 4: Zero-Crossing Rate
            if confidence > 0:
                zcr_score = self._zero_crossing_rate_score(audio)
                confidence += zcr_score
            
            # Modulation check
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
                    else:
                        confidence = max(5.0, confidence - 5.0)
            
            return np.clip(confidence, 0.0, 100.0)
        
        except Exception as e:
            print(f"[VAD ERROR] {e}")
            return 0.0

# ============================================================================
# Test Main
# ============================================================================

tester = VADTester()

print("=" * 70)
print("OSINTCOM v1.11 VAD DIAGNOSTIC - 4-Layer Detection Test")
print("=" * 70)
print(f"Sample rate: 44100 Hz")
print(f"Block size: 2048 samples (~46ms)")
print(f"Using: Layer 1 SNR + Layer 2 Pitch + Layer 3 Entropy + Layer 4 ZCR")
print()
print(f"CV Range (natural speech): 0.25 - 0.60")
print(f"SNR Thresholds: 6dB (open), 2dB (hangover), 0dB (recording)")
print()
print("Listen on DAX Audio from FlexRadio. Speak now!")
print("Listening for 10 seconds...")
print()

frames_recorded = 0
confidence_scores = []

def audio_callback(indata, frames_count, time_info, status):
    global frames_recorded, confidence_scores
    
    if status:
        print(f"[Audio Status] {status}")
    
    audio_chunk = indata[:, 0].copy()
    confidence = tester._detect_voice(audio_chunk)
    confidence_scores.append(confidence)
    frames_recorded += 1
    
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    if confidence > 50:
        indicator = "🟢 VOICE"
    elif confidence > 30:
        indicator = "🟡 POSSIBLE"
    else:
        indicator = "🔴 STATIC"
    
    print(f"[{frames_recorded:2d}] RMS={rms_db:7.1f}dB | Confidence={confidence:6.1f}% | {indicator}")

try:
    with sd.InputStream(channels=1, samplerate=44100, blocksize=2048, 
                        callback=audio_callback, device=None):
        time.sleep(10)
        
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
    silent_frames = sum(1 for c in confidence_scores if c <= 30)
    
    print(f"Frames recorded: {frames_recorded}")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print(f"Max confidence: {max_confidence:.1f}%")
    print()
    print(f"Voice detected in {voice_frames} frames ({voice_frames/len(confidence_scores)*100:.1f}%)")
    print(f"Possible voice in {possible_frames} frames ({possible_frames/len(confidence_scores)*100:.1f}%)")
    print(f"Silent/static in {silent_frames} frames ({silent_frames/len(confidence_scores)*100:.1f}%)")
    print()
    if avg_confidence > 50:
        print("✅ VOICE DETECTION WORKING - Clear voice signal detected!")
    elif avg_confidence > 30:
        print("⚠️  MARGINAL DETECTION - Weak signal or high noise")
    else:
        print("❌ NO VOICE DETECTED - Check audio device/level")
