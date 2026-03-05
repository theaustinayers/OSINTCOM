# OSINTCOM v1.13

**Professional-grade Voice Activity Detection (VAD) and recording system for HF radio SSB monitoring with intelligent noise rejection and Discord integration.**

![OSINTCOM v1.13 Interface](screenshot.png)

---

## About OSINTCOM

OSINTCOM is a **feature-rich VAD recording tool** designed for **HF radio monitoring, EAM (Emergency Action Message) detection, and emergency communications**. It captures faint voice transmissions in high-noise environments while intelligently filtering out static, QRM, and atmospheric noise.

Perfect for:
- 🎙️ HF radio monitoring (SSB, CW, EAM)
- 🚨 Emergency communications surveillance
- 📡 Signal intelligence operations
- 🔊 Audio archival and research

---

## 🎯 Core Features

### Voice Activity Detection (VAD)
- **5-Level Sensitivity Presets** (L1-L5): Adjust confidence thresholds from 46% to 65%
- **Multi-Layer Confidence Scoring**: SNR + pitch + entropy + zero-crossing rate + spectral variation
- **Intelligent SNR Gating**: Recording starts at SNR > 3dB, hangover resets at SNR > 4dB + confidence > 60%
- **Periodic Auto-Calibration**: Recalibrates noise floor every 5 minutes (skips during active voice)
- **Dynamic Threshold Scheduling**: 55% idle, 52% recording, 60% hangover (adapts by state)

### Recording & Processing
- **Smart Pre/Post-Roll**: 5 seconds pre-voice capture + 10-second hangover post-silence
- **Hangover Reset Logic**: 10-frame silence detector (52% confidence) or new voice triggers recording stop
- **Minimum Duration Filter**: 1.1s of sustained voice required for upload (prevents false positives)
- **Audio Processing Pipeline**:
  - 🎚️ **Voice Extraction Slider** (45-70%): Adjustable confidence threshold for voice-only segments
  - 🔊 **Bandpass Filter** (300-3000 Hz): SSB radio optimized, removes out-of-band noise
  - 🔇 **Enhanced Noise Reduction**: noisereduce library with tunable strength (1-10)
  - 🔇 **Silence Gap Removal**: Eliminates quiet periods between speech bursts
- **WAV Export**: 16-bit PCM mono at original sample rate

### Discord Integration
- **Multi-Webhook Management**:
  - ➕ Add unlimited Discord webhook destinations
  - 🏷️ Custom nicknames for each webhook (e.g., "Main Server", "Backup", "Archive")
  - ✅ Per-webhook enable/disable toggle (selective routing without deletion)
  - 📌 Per-webhook role ID support (customize who gets pinged on each server)
- **Automatic Upload**: Sends completed recordings to all enabled webhooks simultaneously
- **Custom Messages**: Customize Discord message template (e.g., "***EAM INCOMING***")
- **Rich Embeds**: Frequency, timestamp, and color-coded alerts

### User Interface
- **Real-Time Audio Meter**: Live dB level with gradient (green → yellow → red)
- **Animated Alert Ticker**: "***EAM INCOMING***" scrolls with flashing effect on black background
  - ShareTechMono font for authentic emergency broadcast aesthetic
  - Auto-triggers on recording start, auto-stops on recording end
- **Frequency Presets**: HFGCS frequencies (4.724 - 18.046 MHz) + custom frequency entry
- **Status Indicators**:
  - Voice detection status (green/red)
  - Sensitivity level display
  - Recording state with duration counter
  - Hangover countdown timer
  - Confidence score percentage
  - Noise floor measurement
- **Dark Theme GUI**: Modern flat design, easy on eyes during long monitoring sessions

### Configuration & Settings
- **Audio Settings Dialog**:
  - Bandpass filter toggle
  - Denoise strength slider (1-10)
  - Silence removal toggle
  - Voice extraction toggle + confidence slider (45-70%)
- **Auto-Save Config**: Settings persist in `osintcom_config.json`
- **Device Selection**: Choose from all available audio inputs (mic, loopback, USB, etc.)
- **File Location Picker**: Set custom save directory for recordings
- **Sensitivity Calibration**: Manual noise floor measurement (10-second collection)

---

## Installation

### Option 1: Standalone Windows Executable (Recommended)

**No Python installation required!**

1. Download `OSINTCOM.exe` from [GitHub Releases](https://github.com/theaustinayers/OSINTCOM/releases)
2. Run the executable (ignore SmartScreen warning — see below)
3. Select your audio input and frequency
4. Click **Start** to begin monitoring

**Note:** First run may show Windows Defender SmartScreen warning (this is normal for unsigned executables). Click "More info" → "Run anyway"

### Option 2: Python Script

**Requirements:** Python 3.8+ 

```bash
# Clone repository
git clone https://github.com/theaustinayers/OSINTCOM.git
cd OSINTCOM

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run
python osintcom_qt.py
```

---

## Quick Start

### 1. Configure Audio Input
- Click dropdown under **"Audio Interface"**
- Select your input source:
  - 🎤 Microphone (direct mic input)
  - 🔊 Speaker Output / Stereo Mix (loopback for receiver audio)
  - 📱 USB device (RTL-SDR with audio output, etc.)

### 2. Select Frequency
- Click one of the **HFGCS frequency buttons** (preset)
- Or type custom frequency in the frequency field

### 3. Configure Discord (Optional)
- Click **"Webhooks"** button to open webhook manager
- Add Discord webhook URL(s) from your server settings
- Assign nicknames and optional role IDs
- Enable/disable which webhooks receive uploads

### 4. Adjust Sensitivity
- Use **Sensitivity slider** (L1-L5)
  - **L1 (46%):** Ultra-sensitive, catches faint voices, more false positives
  - **L3 (53%):** Balanced default, recommended for most
  - **L5 (65%):** Strict, high-confidence only
- Click **"Calibrate Noise"** to measure environment baseline

### 5. Start Monitoring
- Click **Start** button
- Watch the audio meter and ticker display
- When voice is detected → recording begins + ticker animates
- Recordings post to Discord automatically
- Click **Stop** to end monitoring session

---

## Advanced Configuration

### Audio Processing Settings
1. Click **"Audio Settings"** button
2. Enable/disable as needed:
   - **Bandpass Filter**: Removes frequencies outside 300-3000 Hz (SSB optimized)
   - **Denoise Strength**: 1-10 scale (higher = more aggressive)
   - **Remove Silent Gaps**: Cuts quiet periods between transmissions
   - **Voice-Only Extraction**: Toggle + adjust confidence threshold (45-70%)

### Webhook Manager
1. Click **"Webhooks"** button
2. **Add Webhook**:
   - Enter Discord webhook URL
   - Assign nickname
   - (Optional) Enter role ID to ping
   - Enable checkbox
3. **Update/Remove**: Select webhook from list, edit fields, or delete
4. Click **OK** to save

### Sensitivity Levels Explained
| Level | Threshold | Use Case |
|-------|-----------|----------|
| L1 | 46% | Ultra-faint voices, weak signals |
| L2 | 50% | Quiet transmissions |
| L3 | 53% | **Default**, balanced |
| L4 | 60% | Clear voices only |
| L5 | 65% | Voice-only, no borderline signals |

---

## Technical Details

### VAD Confidence Scoring (0-100%)
- **SNR Gate** (25pt baseline): Filters marginal signals
- **Pitch Detection** (+0-15): Speech-specific frequency patterns
- **Entropy** (+0-25): Spectral energy concentration in 500-3500 Hz band
- **Zero-Crossing Rate** (+0-15): Voice-like pattern analysis
- **CV Modulation** (+/-5): Natural vs noise-like amplitude variation

### Recording Flow
```
┌─ Idle (55% threshold)
│
└─→ Voice Detected + SNR > 3dB
    ├─ Start Recording
    ├─ Begin 10-second Hangover Timer
    │
    └─→ Silence (52% confidence)
        ├─ Hangover Active
        ├─ Check: SNR > 4dB AND Confidence > 60%?
        │  ├─ YES → Reset hangover timer (10s loop)
        │  └─ NO → Continue countdown
        │
        └─→ Hangover Expires OR 10 Silence Frames
            ├─ Finalize Recording
            ├─ Validate: voice_duration >= 1.1s?
            │  ├─ YES → Upload to Discord
            │  └─ NO → Discard (insufficient voice)
            │
            └─→ Ready for next recording
```

### SNR Gating Prevents False Positives
- **Recording Start**: Requires SNR > 3.0 dB (filters noise-only bursts)
- **Hangover Reset**: Requires SNR > 4.0 dB + confidence > 60% (prevents noise tail extension)
- **Real Voice SNR**: Typically 10-30 dB
- **Noise Tail SNR**: Typically 0-2 dB (filtered out)

### Periodic Auto-Calibration
- **Trigger**: Every 5 minutes + not currently recording
- **Voice Skip**: Skips if voice detected in past 5 minutes (avoids disruption)
- **Measurement**: Collects 10 seconds of ambient noise
- **Calculation**: SNR_threshold = measured_noise_snr + 4.0 dB (min 13.0 dB)
- **CV Adjustment**: Tightens detection if noise variability high (>0.45)

---

## Dependencies

**Required:**
- numpy >= 1.19
- scipy >= 1.5
- PyQt5 >= 5.15
- sounddevice >= 0.4.5
- requests >= 2.26

**Optional:**
- noisereduce >= 2.0 (enhanced noise filtering)

All included in standalone `.exe` executable.

---

## Discord Webhook Setup

### Create a Webhook in Discord

1. Go to your Discord server settings
2. Navigate to **Integrations** → **Webhooks**
3. Click **New Webhook**
4. Give it a name (e.g., "OSINTCOM Monitor")
5. (#optional) Select a channel
6. Click **Copy Webhook URL**
7. Paste into OSINTCOM's Webhook Manager

### Example Webhook URLs
- ✅ `https://discordapp.com/api/webhooks/999999999/XXXXX...`
- ✅ `https://discord.com/api/webhooks/999999999/XXXXX...`

---

## File Structure

```
OSINTCOM/
├── osintcom_qt.py          # Main application
├── osintcom.py             # CLI version (alternative)
├── build_exe.py            # Build script for .exe
├── requirements.txt        # Python dependencies
├── ShareTechMono-Regular.ttf  # Ticker font
├── osintcom_config.json    # User settings (auto-generated)
├── README.md               # This file
└── dist/
    └── OSINTCOM.exe        # Standalone executable
```

---

## Troubleshooting

### No Audio Input Detected
- **Solution**: Check Windows Sound settings, enable "Stereo Mix" (for loopback)
- On some systems: Right-click speaker icon → Recording devices → Enable disabled devices

### Recordings Won't Upload to Discord
- Verify webhook URL is correct and recent (webhooks expire)
- Check webhook hasn't been deleted from Discord server
- Ensure bot permissions allow message sending
- Check network connectivity

### False Positive Voice Detections
- Reduce sensitivity (L4 or L5)
- Click "Calibrate Noise" to retrain baseline
- Increase voice extraction threshold slider (toward 70%)
- Enable denoise + bandpass filter

### Missing Audio Data in Recordings
- Increase **pre-roll duration** (via code modification)
- Lower sensitivity level (might be missing faint starts)
- Check that voice duration meets 1.1s minimum

### Font Not Loading
- Monospace fallback used if ShareTechMono not found
- Place `ShareTechMono-Regular.ttf` in project folder to enable custom font

---

## Building from Source

### Create Standalone EXE

```bash
# Install build dependency
pip install pyinstaller

# Build (creates dist/OSINTCOM.exe)
python build_exe.py
```

Output: `dist/OSINTCOM.exe` (~117 MB, includes all dependencies)

---

## Version History

### v1.13 (March 2026)
- ✨ **Animated alert ticker** with ShareTechMono font
- ✨ **Multi-webhook manager** with per-webhook role IDs
- ✨ **Voice extraction confidence slider** (45-70%)
- 🐛 Fixed numpy.bool_ type error
- 🔧 SNR gating improvements (>3dB start, >4dB + 60% hangover reset)
- 🔧 Dynamic threshold scheduling by recording state

### v1.12
- Periodic auto-calibration every 5 minutes
- 5-level sensitivity presets (L1-L5)
- Improved post-roll silence detection

### v1.0
- Initial VAD implementation
- Discord webhook integration
- Audio processing pipeline

---

## License

See [LICENSE](LICENSE) file for details.

---

## Support

- 📖 **Documentation**: See README.md
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💬 **Discussions**: GitHub Discussions

---

**Made for HF radio enthusiasts, OSINT researchers, and emergency communications professionals.**


---

## Installation

### Option 1: Python Script (Windows/Mac/Linux)

**Requirements:** Python 3.8+

```bash
# Clone or download this repository
cd OSINTCOM

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Standalone Windows Executable

Download `OSINTCOM.exe` from GitHub Releases — no Python required!

```bash
OSINTCOM.exe
```

#### "Unknown Publisher" Warning (Windows Defender SmartScreen)

When you first run `OSINTCOM.exe`, you may see a security warning:

> **"Windows Defender SmartScreen prevented an unrecognized app from starting"**

This is **normal and safe** — the executable is unsigned (not code-signed), which triggers SmartScreen for any first-run application.

**To run the application:**
1. Click **"More info"**
2. Click **"Run anyway"**

**Why this happens:**
- OSINTCOM is a free, open-source tool built by the community
- Code-signing requires a paid digital certificate
- Open-source projects typically don't have certificates
- SmartScreen is conservative and warns on unsigned executables

**It's safe to click "Run anyway"** — the source code is available on GitHub for inspection.

---

## Usage

### Running the Application

**Python:**
```bash
python osintcom_qt.py
```

**Standalone EXE:**
```
Double-click OSINTCOM.exe
```

### Using the GUI

1. **Select Audio Source:**
   - Click the dropdown under "Audio Interface"
   - Choose your input: Microphone (🎤), Loopback/Stereo Mix (🔊), or USB device

2. **Select Frequency:**
   - Check one of the HFGCS frequency buttons (4.724 - 18.046 MHz)
   - Or use Custom mode to enter your own

3. **Configure Discord Webhook:**
   - Paste your Discord webhook URL in the "Webhook URL" field
   - Click "Customize" to set role pings and message text

4. **Adjust Sensitivity:**
   - Slide the sensitivity bar (1=Faintest, 5=Voice Only)
   - Monitor the VAD indicator (green=voice, red=no voice)

5. **Start Listening:**
   - Click "Start" to begin monitoring
   - Audio meter shows real-time levels
   - Recordings automatically save to `~/Documents/OSINTCOM/`
   - Click "File Location" to open the folder

---

## VAD (Voice Activity Detection) Sensitivity

The VAD uses **spectral analysis** to distinguish voice from static:

| Level | Name | Best For | Behavior |
|-------|------|----------|----------|
| **1** | Maximum | Weak/faint signals | Catches faintest voice; more false positives on static |
| **2** | Relaxed | Typical SSB | Good balance; catches weak voice with minimal false triggers |
| **3** | Balanced (Default) | SSB radio | Recommended; steady performance on typical conditions |
| **4** | Strict | Clean channels | Rejects more noise; may miss weakest voice |
| **5** | Voice Only | Voice isolation | Maximum static rejection; use only for clear voice |

**How it Works:**
- **Energy Check:** Audio must exceed minimum dB threshold
- **Spectral Flatness:** Static = flat spectrum (rejected); Voice = peaks in frequency content
- **Zero-Crossing Rate:** Distinguishes voice (moderate) from noise (very high)
- **Pitch Periodicity:** Voice has periodic structure; static is random

→ **Tip:** Start at Level 3. If you miss voice, go to 2 or 1. If you catch too much static, go to 4 or 5.

---

## Technical Deep-Dive: VAD Architecture

For detailed information about how the VAD detection pipeline works, see **[VAD-Architecture.md](docs/VAD-Architecture.md)**.

This covers:
- **3-gate detection system** (SNR gate + speech-likeness verification)
- **Parameter tuning guide** for different HF conditions
- **Squelch behavior** (attack/hangover) to prevent clipped words
- **Recording strategy** with pre-roll and sanity checks
- **Advanced troubleshooting** for weak stations and false positives

This is the authoritative reference for v1.08+ VAD behavior.

---

## Troubleshooting

### Audio Not Detected / No Devices Showing

**On Windows:** Check if you have an audio input device available.

Run this in PowerShell:
```powershell
Get-PnpDevice | Where-Object { $_.Class -eq "MEDIA" } | Select-Object Name, Status
```

### Can't Listen to Speakers (Loopback Not Available)

**Problem:** OSINTCOM doesn't see "Stereo Mix" or "Loopback" device.

**Solution 1: Enable Windows Stereo Mix (if available)**
1. Right-click the speaker icon in system tray → **Open Sound settings**
2. Scroll down → **Advanced** → **App volume and device preferences**
3. Look for "Stereo Mix" in the device list
4. If it's disabled (grayed out):
   - Right-click (in older Windows) → **Show Disabled Devices**
   - Right-click "Stereo Mix" → **Enable**
5. Restart OSINTCOM and select "Stereo Mix" from the dropdown

**If "Stereo Mix" doesn't appear:** Your audio driver may not support it.

**Solution 2: Install Virtual Audio Cable** (Recommended if no Stereo Mix)
1. Download and install one of:
   - **VB-Audio Virtual Cable** (free): https://vb-audio.com/Cable/
   - **VoiceMeeter** (free): https://vb-audio.com/Voicemeeter/
   - **Virtual Audio Cable** (paid): https://vb-audio.com/

2. After installation, restart OSINTCOM
3. Select the virtual device from the dropdown (e.g., "VB-Audio Virtual Cable")
4. Route system audio through the virtual cable in Windows Sound Settings

### Still Detecting Static as Voice

1. Increase sensitivity level (4 or 5)
2. Check that your input device is not picking up excessive background noise
3. Test with a known voice source to verify detection works
4. Try the optional denoiser (check "Enable Denoiser") for additional filtering

### Discord Messages Not Sending

1. **Verify webhook URL:**
   - Check that the Discord webhook URL is copied correctly (should start with `https://discord.com/api/webhooks/`)
   - Test in browser: paste URL and you should see a 405 error (that's correct!)

2. **Check Discord channel permissions:**
   - Bot must have "Send Messages" and "Embed Links" permissions
   - Role ID must exist and be valid

3. **Internet connectivity:**
   - Verify your PC can reach Discord (ping discord.com)

### WAV Files Not Saving

1. Ensure `~/Documents/OSINTCOM/` exists and is writable
2. Check disk space (each minute of audio ≈ 5.3 MB)
3. Filenames use format: `YYYYMMDD_HHMMSS.wav`

---

## Discord Setup

### Creating a Webhook

1. Go to your Discord server → **Server Settings** → **Integrations**
2. Click **Webhooks** → **New Webhook**
3. Name it (e.g., "OSINTCOM Radio") and choose a channel
4. Click **Copy Webhook URL**
5. Paste into OSINTCOM's "Webhook URL" field

### Role Pings

To ping a role when voice is detected:
1. In Discord: Right-click the role name → **Copy User ID**
2. In OSINTCOM: Click "Customize" and paste the role ID
3. Messages will include `<@&ROLE_ID>` to notify the role

---

## Dependencies

- **sounddevice**: Audio I/O
- **numpy**: Signal processing
- **scipy**: Spectral analysis & filtering
- **requests**: Discord webhook HTTP
- **noisereduce** (optional): Noise reduction filter
- **PyQt5**: GUI framework

All included in `requirements.txt`.

---

## Building from Source

To build `OSINTCOM.exe` yourself:

```bash
pip install PyInstaller
python build_exe.py
```

Output: `dist/OSINTCOM.exe` (~112 MB)

---

## Configuration Files

- **osintcom_qt.py**: Main application (GUI + VAD)
- **requirements.txt**: Dependencies
- **build_exe.py**: PyInstaller build script
- **.gitignore**: Excludes venv, builds, recordings, temp files

---

## Performance Notes

- **CPU Usage:** Minimal (<5% typical)
- **Memory:** ~100-150 MB (with all dependencies)
- **Latency:** <100 ms from voice detection to recording start
- **Audio Format:** 44.1 kHz, 16-bit mono
- **Recording Overhead:** ~5.3 MB/minute

---

## Limitations

- **Windows & WASAPI:** Loopback audio (speaker output) requires Windows + compatible audio driver
- **macOS/Linux:** Microphone input works; speaker loopback depends on OS audio subsystem
- **Static Rejection:** SSB radio environments with extreme noise may need tuning
- **Real-time Processing:** VAD runs on received audio blocks (~46 ms at 44.1 kHz)

---

## Future Improvements

- [ ] Frequency preset profiles (different sensitivity for each band)
- [ ] Multi-frequency monitoring simultaneously
- [ ] Advanced noise reduction (DeepFilter, SEGAN)
- [ ] Audio archive uploading to cloud storage
- [ ] Web-based dashboard for remote monitoring

---

## License

[Specify your license here - MIT, GPL, etc.]

## Support

For issues or feature requests, please open an issue on GitHub.

---

**OSINTCOM v1.0** | Created for professional HF radio monitoring.
