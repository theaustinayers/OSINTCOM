#!/usr/bin/env python3
"""
OSINTCOM v1.11 VAD Debug - Detailed signal analysis
Captures raw metrics: RMS, SNR, Pitch, Entropy, ZCR
"""
import numpy as np
import sounddevice as sd
import time
import collections

class VADDebugger:
    def __init__(self):
        self._sample_rate = 44100
        self._noise_floor_rms = 0.001
        self._snr_history = collections.deque(maxlen=300)
        self._adaptive_cv_min = 0.25
        self._adaptive_cv_max = 0.60
    
    def _detect_pitch(self, audio):
        """Detect pitch with detailed metrics."""
        try:
            if len(audio) < 512:
                return 0.0, 0.0
            
            audio_work = audio - np.mean(audio)
            if np.max(np.abs(audio_work)) < 1e-10:
                return 0.0, 0.0
            
            fft = np.fft.fft(audio_work, n=2*len(audio_work))
            power = fft * np.conj(fft)
            autocorr = np.fft.ifft(power)[0:len(audio_work)]
            autocorr = np.real(autocorr)
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            min_lag = max(10, int(self._sample_rate / 250))
            max_lag = min(len(autocorr)-1, int(self._sample_rate / 85))
            
            if max_lag <= min_lag:
                return 0.0, 0.0
            
            autocorr_pitch = autocorr[min_lag:max_lag]
            peak_idx = np.argmax(autocorr_pitch) if len(autocorr_pitch) > 0 else 0
            peak_strength = autocorr_pitch[peak_idx] if len(autocorr_pitch) > 0 else 0.0
            
            # Score: 0-25
            if peak_strength > 0.45:
                score = 25.0
            elif peak_strength < 0.20:
                score = 0.0
            else:
                score = (peak_strength - 0.20) / 0.25 * 25.0
            
            return score, peak_strength
        except:
            return 0.0, 0.0
    
    def _estimate_spectral_entropy(self, audio):
        """Spectral entropy with detailed metrics."""
        try:
            if len(audio) < 512:
                return 0.0, 0.0
            
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
            
            # Score: 0-25
            if normalized_entropy > 0.65:
                score = 0.0
            elif normalized_entropy < 0.45:
                score = 25.0
            else:
                score = (0.65 - normalized_entropy) / 0.20 * 25.0
            
            return score, normalized_entropy
        except:
            return 0.0, 0.0
    
    def _zero_crossing_rate_score(self, audio):
        """Zero-crossing rate with detailed metrics."""
        try:
            if len(audio) < 2:
                return 0.0, 0.0
            
            audio_work = audio - np.mean(audio)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_work)))) / 2.0
            zcr = zero_crossings / len(audio_work)
            
            # Score: 0-25
            if zcr > 0.30:
                score = 0.0
            elif zcr < 0.10:
                score = 25.0
            else:
                score = (0.30 - zcr) / 0.20 * 25.0
            
            return score, zcr
        except:
            return 0.0, 0.0
    
    def analyze_frame(self, audio):
        """Complete analysis of one audio frame."""
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        # RMS and SNR
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        snr_db = 20 * np.log10((rms + 1e-10) / (self._noise_floor_rms + 1e-10))
        self._snr_history.append(snr_db)
        
        if len(self._snr_history) > 10:
            snr_percentile = np.percentile(list(self._snr_history), 20)
        else:
            snr_percentile = snr_db
        
        # Modulation (CV)
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
            cv = np.sqrt(np.var(rms_values)) / (rms_mean + 1e-10) if rms_mean > 1e-10 else 0.0
        else:
            cv = 0.0
        
        # Layer scores
        pitch_score, pitch_strength = self._detect_pitch(audio)
        entropy_score, entropy_val = self._estimate_spectral_entropy(audio)
        zcr_score, zcr_val = self._zero_crossing_rate_score(audio)
        
        # SNR gate
        snr_threshold = 3.0
        snr_passes = snr_percentile > snr_threshold
        snr_score = 25.0 if snr_passes else 0.0
        
        # Total confidence
        total_confidence = snr_score + pitch_score + entropy_score + zcr_score
        
        # Modulation bonus
        if self._adaptive_cv_min < cv < self._adaptive_cv_max:
            total_confidence = min(100.0, total_confidence + 5.0)
        else:
            total_confidence = max(5.0, total_confidence - 5.0)
        
        return {
            'rms_db': rms_db,
            'snr_db': snr_db,
            'snr_percentile': snr_percentile,
            'snr_passes': snr_passes,
            'cv': cv,
            'pitch_score': pitch_score,
            'pitch_strength': pitch_strength,
            'entropy_score': entropy_score,
            'entropy_val': entropy_val,
            'zcr_score': zcr_score,
            'zcr_val': zcr_val,
            'snr_score': snr_score,
            'total_confidence': total_confidence,
        }

# Main
debugger = VADDebugger()

print("=" * 90)
print("OSINTCOM v1.11 VAD DETAILED DEBUG - Signal Analysis")
print("=" * 90)
print("Speak LOW, NORMAL, then STRONG signals on FlexRadio RX 1")
print("=" * 90)
print()

frames = 0
all_frames = []

def audio_callback(indata, frames_count, time_info, status):
    global frames, all_frames
    
    if status:
        print(f"[Audio Status] {status}")
    
    audio_chunk = indata[:, 0].copy()
    result = debugger.analyze_frame(audio_chunk)
    all_frames.append(result)
    frames += 1
    
    # Display
    indicator = ""
    if result['total_confidence'] > 50:
        indicator = "🟢 VOICE"
    elif result['total_confidence'] > 30:
        indicator = "🟡 MAYBE"
    else:
        indicator = "🔴 STATIC"
    
    print(f"[{frames:3d}] RMS={result['rms_db']:7.1f}dB | SNR={result['snr_percentile']:6.1f}dB | "
          f"Pitch={result['pitch_score']:5.1f} ({result['pitch_strength']:.2f}) | "
          f"Entropy={result['entropy_score']:5.1f} ({result['entropy_val']:.2f}) | "
          f"ZCR={result['zcr_score']:5.1f} ({result['zcr_val']:.2f}) | "
          f"CV={result['cv']:.2f} | TOTAL={result['total_confidence']:6.1f}% {indicator}")

try:
    print("[Starting capture - listening to DAX Audio RX 1]")
    print("[Using device index 7 (DAX Audio RX 1)]")
    print()
    with sd.InputStream(channels=1, samplerate=44100, blocksize=2048, 
                        callback=audio_callback, device=7):  # FORCE Device 7
        input("Press ENTER to start capturing (speak LOW signal first)...\n")
        time.sleep(5)  # 5 seconds LOW
        
        input("\nPress ENTER to capture NORMAL signal...\n")
        time.sleep(5)  # 5 seconds NORMAL
        
        input("\nPress ENTER to capture STRONG signal...\n")
        time.sleep(5)  # 5 seconds STRONG
        
except KeyboardInterrupt:
    print("\n[Interrupted]")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 90)
print("ANALYSIS")
print("=" * 90)

# Divide into 3 segments
segment_size = len(all_frames) // 3

if segment_size > 0:
    segments = [
        all_frames[0:segment_size],
        all_frames[segment_size:2*segment_size],
        all_frames[2*segment_size:],
    ]
    names = ["LOW SIGNAL", "NORMAL SIGNAL", "STRONG SIGNAL"]
    
    for name, segment in zip(names, segments):
        if len(segment) == 0:
            continue
        
        avg_rms = np.mean([f['rms_db'] for f in segment])
        avg_snr = np.mean([f['snr_percentile'] for f in segment])
        avg_pitch = np.mean([f['pitch_score'] for f in segment])
        avg_entropy = np.mean([f['entropy_score'] for f in segment])
        avg_zcr = np.mean([f['zcr_score'] for f in segment])
        avg_total = np.mean([f['total_confidence'] for f in segment])
        
        print(f"\n{name}:")
        print(f"  RMS:          {avg_rms:6.2f} dB")
        print(f"  SNR:          {avg_snr:6.2f} dB")
        print(f"  Pitch Score:  {avg_pitch:6.2f}/25")
        print(f"  Entropy Score:{avg_entropy:6.2f}/25")
        print(f"  ZCR Score:    {avg_zcr:6.2f}/25")
        print(f"  TOTAL CONF:   {avg_total:6.2f}%")
        print(f"  Detected:     {'YES ✓' if avg_total > 50 else 'NO ✗'}")
