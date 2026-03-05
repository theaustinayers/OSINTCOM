import sounddevice as sd

print("Available Audio Devices:")
print("=" * 70)
devices = sd.query_devices()
for i, d in enumerate(devices):
    print(f"{i:2d}: {d['name'][:40]:40s} | In: {d['max_input_channels']} | Out: {d['max_output_channels']}")

print("\nDefault input device:", sd.default.device[0])
print("Default output device:", sd.default.device[1])
