#!/usr/bin/env python3
"""
Find which device has audio signal
"""
import sounddevice as sd
import numpy as np
import time

print("=" * 80)
print("AUDIO DEVICE FINDER - Which device has signal?")
print("=" * 80)
print()

# List all devices
devices = sd.query_devices()
print(f"Total devices: {len(devices)}\n")

# Test each device
for idx in range(len(devices)):
    device = devices[idx]
    
    # Skip output-only devices
    if device['max_input_channels'] == 0:
        continue
    
    print(f"Testing Device {idx}: {device['name']}")
    
    try:
        # Try to read 1 second of audio
        audio, sr = sd.rec(int(sr := device['default_samplerate']), channels=1, device=idx, duration=1, blocking=True)
        
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        if rms_db > -50:
            print(f"  ✓ HAS AUDIO: RMS = {rms_db:.1f} dB")
        else:
            print(f"  • Silent: RMS = {rms_db:.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
    
    time.sleep(0.2)

print()
print("=" * 80)
print("Use the device index above in debug_vad_detailed.py")
