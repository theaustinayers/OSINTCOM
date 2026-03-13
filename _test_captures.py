#!/usr/bin/env python3
"""Quick test: score 3 new captures through v1.24 formant logic."""
import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt, welch

SR = 44100  # device default_samplerate

PRESETS = {
    1: {'formant_prominence_db': 3.5, 'flat_penalty_factor': 0.65, 'formant_threshold_db': 5},
    2: {'formant_prominence_db': 4.5, 'flat_penalty_factor': 0.55, 'formant_threshold_db': 5},
    3: {'formant_prominence_db': 6.0, 'flat_penalty_factor': 0.40, 'formant_threshold_db': 5},
    4: {'formant_prominence_db': 7.5, 'flat_penalty_factor': 0.25, 'formant_threshold_db': 5},
    5: {'formant_prominence_db': 9.0, 'flat_penalty_factor': 0.15, 'formant_threshold_db': 5},
}
flatness_gates = {1: 0.80, 2: 0.76, 3: 0.72, 4: 0.68, 5: 0.62}


def score_formants(audio, sens=3):
    preset = PRESETS.get(sens, PRESETS[3])
    flatness_gate = flatness_gates.get(sens, 0.72)

    # In-band flatness pre-gate
    try:
        sos_bp = butter(4, [300, 4000], btype='band', fs=SR, output='sos')
        band_audio = sosfiltfilt(sos_bp, audio)
        _, pxx = welch(band_audio, fs=SR, nperseg=min(512, len(band_audio)))
        pxx = np.maximum(pxx, 1e-12)
        in_band_flatness = np.exp(np.mean(np.log(pxx))) / np.mean(pxx)
        gated = in_band_flatness > flatness_gate
        print(f"  In-band flatness: {in_band_flatness:.4f}  gate={flatness_gate}", end="")
        print("  -> PRE-GATE BLOCKED" if gated else "  -> pass")
        if gated:
            return 0.0, 0
    except Exception as e:
        print(f"  flatness err: {e}")

    # FFT
    window = np.hanning(len(audio))
    fft_result = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-9)

    formant_mask = (freqs > 300) & (freqs < 4000)
    formant_freqs = freqs[formant_mask]
    formant_mag_db = magnitude_db[formant_mask]

    if len(formant_mag_db) == 0:
        print("  No formant-band samples")
        return 0.0, 0

    prominence_db = preset.get('formant_prominence_db', 6.0)
    threshold_db = preset.get('formant_threshold_db', 5)
    formant_floor = np.median(formant_mag_db) + threshold_db

    peaks, _ = find_peaks(
        formant_mag_db,
        height=formant_floor,
        distance=max(1, int(150 / (SR / len(audio)))),
        prominence=prominence_db
    )
    peak_freqs = formant_freqs[peaks].tolist()
    print(f"  Raw peaks ({len(peaks)}): {[round(f, 0) for f in peak_freqs[:10]]}")

    # Cluster dedup
    clusters = []
    for pf in sorted(peak_freqs):
        if clusters and (pf - clusters[-1]) < 300:
            pass
        else:
            clusters.append(pf)
    print(f"  Clusters  ({len(clusters)}): {[round(c, 0) for c in clusters]}")

    if len(clusters) >= 2:
        span = max(clusters) - min(clusters)
        print(f"  Span: {span:.0f} Hz  (need >= 400)")
        if span < 400:
            clusters = [float(np.mean(clusters))]
            print(f"  -> Collapsed to 1 cluster (span too narrow)")

    nc = len(clusters)
    if nc == 0:
        score = 0.0
    elif nc == 1:
        all_in = peak_freqs
        bw = (max(all_in) - min(all_in)) if all_in else 0
        if bw <= 150:
            score = 22.0
        elif bw <= 250:
            score = 17.0
        else:
            score = 10.0
        print(f"  Single cluster BW={bw:.0f} Hz -> {score} pts")
    elif nc == 2:
        score = 22.0
    elif nc == 3:
        score = 36.0
    else:
        score = 40.0

    print(f"  => Formant score: {score:.1f} / 40   ({nc} clusters)")
    return score, nc


files = [
    ("dax_iq_20260313_131611.npy", "Very strong"),
    ("dax_iq_20260313_131656.npy", "Medium in/out"),
    ("dax_iq_20260313_131733.npy", "Very weak"),
]

for fname, label in files:
    try:
        data = np.load(fname)
    except FileNotFoundError:
        print(f"\n{fname}: NOT FOUND")
        continue

    if data.ndim == 2:
        audio = data[:4 * 2048, 0].astype(np.float32)
    else:
        audio = data[:4 * 2048].astype(np.float32)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    print(f"\n{'=' * 60}")
    print(f"[{label}]  file={fname}  samples={len(audio)}")
    for s in [1, 2, 3, 4, 5]:
        print(f"  -- Sensitivity L{s} --")
        score_formants(audio.copy(), sens=s)
