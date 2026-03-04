# OSINTCOM v1.08 VAD Architecture

## Overview

**Goal:** Bulletproof voice detection for HF SSB radio with **zero false positives** on static/crashes.

**Philosophy:** Don't rely on a single VAD. Use three independent gates + professional squelch behavior.

---

## Three-Gate Detection Pipeline

### Gate 1: SNR Gate (Cheap, Strict Energy Check)

**Purpose:** Kill random static spikes before VAD even runs.

**Implementation:**

1. **Audio Preprocessing (Pre-VAD)**
   - Convert to mono
   - Resample to 16 kHz
   - Apply bandpass 250–2800 Hz (removes HF garbage and rumble)
   - Process in 20 ms frames (~320 samples@16kHz)

2. **Continuous Noise Floor Tracking**
   - Maintain sliding 30-second window of RMS values
   - Noise floor = 10th percentile RMS over this window
   - Re-learn every 4 minutes (adapts to QSB/QRN changes)

3. **Hysteresis Logic (No Chattering)**
   ```
   IF hangover_active:
       open_threshold = +7 dB SNR    (close threshold, prevents clipping)
   ELSE:
       open_threshold = +12 dB SNR   (open threshold, prevents false opens)
   
   SNR = 20 * log10(current_rms / noise_floor)
   
   IF SNR >= open_threshold AND duration >= 250 ms:
       Proceed to Gate 2
   ELSE:
       Return confidence = 5.0 (failed SNR gate)
   ```

**Parameters (v1.08 Defaults):**
- Open threshold: **+12 dB SNR**
- Close threshold: **+7 dB SNR**
- Attack (sustain before opening): **250 ms**
- Noise floor window: **30 seconds**
- Re-learn interval: **240 seconds (4 minutes)**

---

### Gate 2: WebRTC VAD (Speech Timing)

**Purpose:** Detect speech-like syllable structure.

**Implementation:**

1. **WebRTC VAD** (if available)
   - Mode 3 (most aggressive, lowest false positives)
   - 10–20 ms frames
   - Detects presence of human speech timing

2. **Fallback** (v1.08 uses custom verifier instead)
   - If WebRTC not available, use Gate 3 checks below
   - Current implementation: 3-check speech verification

**Note:** v1.08 uses integrated speech-likeness checks (Gate 3) as Van detection rather than external WebRTC library.

---

### Gate 3: Speech-Likeness Verification (The Difference-Maker)

**Purpose:** Reject "strong noise that looks like voice" (crashes, static bursts, fast AGC artifacts).

**Implementation: Three Parallel Checks (Require 2/3 Pass)**

#### Check 1: Harmonic/Voicing Detection
- **What it detects:** Pitched signals with structured spectral content (human voice)
- **Method:**
  - Autocorrelation peak in 80–300 Hz range (typical speech pitch)
  - OR spectral flatness < 0.55 (speech has peaks; noise is flat)
- **Rule:** 
  ```
  voiced_frames = count(frames where pitch > 0.25 AND flatness < 0.55)
  voicing_ratio = voiced_frames / total_frames (over 0.5–1.0s window)
  
  PASS if: voicing_ratio >= 0.25 (at least 25% voiced)
  ```
- **Rejects:** Static bursts (flat spectrum), crashes (no pitch)

#### Check 2: Syllabic Modulation (3–8 Hz Envelope)
- **What it detects:** Human speech has characteristic amplitude modulation at syllable rate
- **Method:**
  - Extract amplitude envelope (RMS over 3–10 ms windows)
  - FFT the envelope to find frequency peaks
  - Measure energy in 3–8 Hz band (speech syllables)
  - Compare to silence bands (0–1 Hz, >15 Hz)
- **Rule:**
  ```
  modulation_score = energy_in_speech_band / total_energy
  PASS if: modulation_score >= 0.4
  ```
- **Rejects:** Continuous noise (flat envelope), crashes (no modulation)

#### Check 3: Narrowband/Formant Structure
- **What it detects:** Speech clusters energy in formants; noise is uniformly spread
- **Method:**
  - Compute power spectrum (Welch or FFT)
  - Count local maxima (peaks)
  - Peak ratio = peaks / (spectrum_length / 50)
- **Rule:**
  ```
  peak_count = count(local_maxima in spectrum)
  formant_score = peak_count / (spectrum_length / 50)
  PASS if: formant_score >= 0.3
  ```
- **Rejects:** Flat-spectrum noise, uniform crashes

**Gate 3 Confidence Scoring:**
```python
checks_passed = sum([check1, check2, check3])

IF checks_passed < 2:
    confidence = 10.0     # Speech verification failed
ELSE:
    # 2–3 checks passed
    confidence = 50 + (checks_passed / 3) * 50    # 50–100 range
```

---

## Squelch Behavior (Attack/Hangover)

**Purpose:** Prevent word clipping, eliminate machine-gun false opens.

**Implementation:**

1. **Attack (Opening Delay)**
   - Require 250 ms of gates-passed before opening recording
   - Prevents hair-trigger opens on brief noise spikes

2. **Pre-roll Buffer**
   - Ring buffer: 1.0–2.0 seconds of audio BEFORE gate opens
   - When recording starts, write pre-roll first
   - Captures first syllable of speech (critical on SSB where fading is fast)

3. **Hangover (Hold-up After Speech Ends)**
   - When confidence > 60, set hangover timer = 2.0 seconds
   - Continue recording even if gates fail temporarily
   - Decrement hangover each frame: `hangover -= frame_size/sample_rate`
   - Recording stops when: `(confidence <= 60) AND (hangover <= 0)`

**Parameters (v1.08 Defaults):**
- Attack: **250 ms** (20 frames @ 20 ms)
- Pre-roll: **1.0–2.0 seconds**
- Hangover: **2.0 seconds**

---

## Recording Strategy

### Segment Lifecycle

1. **Pre-roll Phase** (1.0–2.0s before gate opens)
   - Audio written to ring buffer
   - Not recorded to disk yet

2. **Gate Opens** (confidence > 60, attack time passed)
   - Flush pre-roll to disk
   - Start writing current frames to disk
   - Set hangover timer

3. **Recording Active** (while confidence > 60 OR hangover > 0)
   - Write all frames to disk
   - Update hangover countdown

4. **Gate Closes** (confidence ≤ 60 AND hangover expired)
   - Stop writing to disk
   - Post-process segment (optional: RNNoise denoising)
   - Validate segment (sanity checks)

### Sanity Checks (After Recording Stops)

```python
IF duration < 1.0 s:
    DELETE segment  # Too short, likely noise

voiced_ratio = voiced_frames / total_frames
IF voiced_ratio < 0.15:
    DELETE segment  # Not enough speech characteristics

OTHERWISE:
    KEEP segment
    LOG: {snr_peak, voiced_ratio, duration, formant_ratio}
```

### File Organization

```
C:\Users\[Username]\Documents\OSINTCOM\
├── YYYY-MM-DD/
│   ├── HHMMSS_[confidence]_[snr].wav    (frequency label optional)
│   ├── HHMMSS_[confidence]_[snr].wav
│   └── ...
└── session.log
```

Optional metadata file next to each WAV:
```json
{
  "timestamp": "2026-03-03T12:34:56Z",
  "duration_ms": 2450,
  "snr_peak_db": 14.2,
  "voiced_ratio": 0.68,
  "modulation_score": 0.58,
  "formant_score": 0.42,
  "confidence_peak": 82
}
```

---

## Parameter Tuning Guide

### Starting Point (Balanced, No False Positives)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **Open threshold** | +12 dB SNR | Strict gate—prevents opens on noise |
| **Close threshold** | +7 dB SNR | Easy close—allows natural speech fades |
| **Attack** | 250 ms | Prevents hair-trigger on spikes |
| **Hangover** | 2.0 s | Protects words during QSB fades |
| **Noise floor window** | 30 s | Adapts to QRN changes |
| **Bandpass** | 250–2800 Hz | Removes DC, HF garbage |
| **Voicing threshold** | >25% | At least 1 in 4 frames must be pitched |
| **Modulation threshold** | >0.4 | Strong 3–8 Hz envelope signature |
| **Formant threshold** | >0.3 | Spectral peaks present |

### Tuning Scenarios

#### Scenario: Too Many False Positives (Recording Trash)
```
Issue: Static crashes, fast-AGC artifacts trigger recording

Adjust (in order):
1. Tighten voice verification:
   - voicing_threshold: 0.25 → 0.35
   - modulation_threshold: 0.4 → 0.5
   - formant_threshold: 0.3 → 0.4

2. Raise open threshold:
   - +12 dB → +14 dB (stricter SNR gate)

3. Lower hangover (if hanging on noise):
   - 2.0 s → 1.5 s

Action: Test on HF with known static sources
```

#### Scenario: Missing Weak Stations (Too Sensitive)
```
Issue: Can't hear distant/fading stations

Adjust (in order):
1. Lower open threshold:
   - +12 dB → +10 dB (more permissive SNR gate)
   - Increase attack time to 500 ms if False opens rise

2. Relax voice verification (last resort):
   - voicing_threshold: 0.25 → 0.20
   - modulation_threshold: 0.4 → 0.35

3. Increase hangover:
   - 2.0 s → 2.5 s (protects weak fading speech)

Action: Log SNR values during weak station tests
```

#### Scenario: Clipped Words (Hangover Too Short)
```
Issue: First syllables cut off, last syllables lost

Adjust:
1. Increase pre-roll:
   - 1.0 s → 2.0 s (more audio before gate opens)

2. Increase hangover:
   - 2.0 s → 2.5–3.0 s (longer hold after voice ends)

3. Increase attack time slightly:
   - 250 ms → 400 ms (prevents slamming shut during SSB fades)

Action: Record test clips, inspect .wav files for clipping
```

---

## Key Settings for HF SSB Success

### If You Have AGC Control

1. **Disable fast AGC** or set to slow (500–1000 ms)
   - Fast AGC makes noise bursts "rise" like speech → #1 cause of false opens
   - Slow AGC lets SNR gate work properly

2. **IF receiver has noise blanker:** Enable it
   - Removes impulse clicks that trigger false opens

### Bandpass Filter Choice

- **Conservative (250–2800 Hz):** Removes DC rumble + HF hash, safe default
- **Tight (300–2700 Hz):** If band is very noisy, further restricts bandwidth
- **Wide (200–3000 Hz):** If you're clipping weak faint voices, try wider

---

## Console Output (Debug Mode)

When running v1.08, you'll see:

```
Frame 2450:
  SNR: 14.2 dB  [open: 12.0, close: 7.0]
  Voicing: 0.68  [threshold: 0.25]  ✓
  Modulation: 0.58  [threshold: 0.40]  ✓
  Formants: 0.42  [threshold: 0.30]  ✓
  Checks passed: 3/3
  Confidence: 100
  Hangover: 1.87 s
  Status: RECORDING

--- SEGMENT END ---
Recording stopped. Duration: 2.45s, Voicing ratio: 0.68, SNR peak: 14.2 dB
```

---

## Practical HF Example

### Scenario: Monitoring HFGCS 8992 kHz

1. **Equipment Setup**
   - IC-7300 + USB audio cable → Windows audio device
   - AGC set to SLOW (or disabled)
   - Receiver AF level adjusted for -20 to -10 dB peak on meter

2. **Initial Run**
   - Load OSINTCOM v1.08
   - Set sensitivity slider to Level 3 (middle)
   - Start monitoring—let it run 30 seconds (noise floor learns)

3. **First Station Heard**
   - Console shows: `SNR: 16 dB, Voicing: 0.75, Modulation: 0.62, Formants: 0.45 → Confidence: 95`
   - Recording starts, captures full transmission
   - ✓ Working

4. **Weak Station (Fading)**
   - Console shows: `SNR: 8.5 dB, Voicing: 0.22, Modulation: 0.35, Formants: 0.25 → Confidence: 45` (fails voice gate)
   - BUT hangover is 1.2 seconds remaining → **Still recording**
   - Captures word transitions smoothly

5. **Static Burst (False Open Risk)**
   - Fast spike in RMS: `SNR: 14 dB` (above 12 dB threshold!)
   - BUT voicing = 0.05, formants = 0.12 (not pitched)
   - Only 1/3 checks pass → `Confidence: 10` (rejected)
   - ✓ No false recording

---

## Future Enhancements

- **WebRTC VAD Integration:** Use external VAD library for additional gate
- **Machine Learning Verifier:** Train small classifier on HF voice vs static
- **Multi-Receiver Support:** Independent pipelines per SDR/radio
- **Post-Processing:** RNNoise denoising on saved WAV files
- **Segment Classification:** Label "clean voice" vs "weak/fading" in metadata

---

## Summary

v1.08 implements **Gate 1** (SNR) + **Gate 3** (speech verification) + **squelch behavior** for bulletproof HF monitoring.

**Design principle:** Cheaper DSP first (energy gate), then intelligence (speech verification), rather than perfect VAD algorithm.

**Expected result:** Near-zero false positives on HF SSB, even with fast QSB/QRN and equipment with aggressive AGC.
