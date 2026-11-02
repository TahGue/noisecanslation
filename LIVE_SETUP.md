# Live mode: hear only voices from your PC

```
 apps (YouTube, games, Netflix...)                 VoiceOnly                    you
 ───────────> Realtek Digital Output ──loopback──> remove music/SFX ──> WH-1000XM4 headphones
              (Windows default, not listened to)   (RTX 4090)
```

No driver is needed: VoiceOnly records what Windows sends to an output you
don't listen to (WASAPI loopback) and plays the filtered result to the one you do.

## Each time you want voice only

Easiest: double-click `VoiceOnly.bat` and use the app.

- **Normal / Voice only** buttons (or **Ctrl+Alt+V** from any app). The app switches
  Windows' output for you and restores it when you go back to Normal or close it.
- **Latency** presets: Clean, Balanced, Fast, Ultra (see below). Changes apply live.
- **Advanced ▸** opens sliders for step, lookahead, safety buffer and model window
  ("Custom" preset). The status line shows the delay and glitch count while running.

Command line instead: set Windows output to **Realtek Digital Output**, then
`uv run voiceonly live -i loopback:Realtek -o WH-1000XM4 --preset fast`.

Requirements for the "sink" output (`SINK=` in the .bat):
- You must not hear it: nothing connected to Realtek Digital Output, or if you use
  the monitor (`SINK=PHL`), turn the monitor's own speaker volume to zero.
- Keep its **Windows** volume up; if VoiceOnly sounds quiet, raise it.
- It must differ from `OUTPUT=` (VoiceOnly refuses a loop).

`uv run voiceonly devices` lists outputs and loopback sources.

## Latency presets

| Preset | Step | Lookahead | Window | Safety buffer | Engine delay | Voice quality* |
|---|---|---|---|---|---|---|
| Clean | 0.10 s | 0.20 s | 4 s | 0.10 s | ~0.45 s | best (SI-SDR 15.1 dB) |
| Balanced | 0.05 s | 0.10 s | 4 s | 0.08 s | ~0.25 s | -1.5 dB |
| Fast | 0.05 s | 0.05 s | 3 s | 0.06 s | ~0.18 s | -2 to -4 dB |
| Ultra | 0.04 s | 0.02 s | 2 s | 0.05 s | ~0.12 s | -2 dB, more background at word starts |

\*Live-sim on the TTS test clip (HTDemucs, fp16, CUDA graph). Add the device buffer
(shown in the app) and Bluetooth (~0.2 s for WH-1000XM4). Smaller steps mean more GPU
steps per second: if the status line shows glitches climbing, use a slower preset or
raise the safety buffer. Close other GPU-heavy apps for Fast/Ultra.

Command-line flags (`--hop`, `--lookahead`, `--window`, `--crossfade`, `--jitter`)
override the chosen `--preset`.

The buffer self-corrects by changing playback speed by at most 1% (inaudible),
so stalls and clock differences don't cut audio.

Video lip-sync will lag by the total delay; VLC (`J`/`K`) and MPC-HC can shift audio sync.

## Measure it yourself

`scripts/live_e2e.py` plays a test clip into the capture output, runs live mode and
records your headphones, then reports real delay, glitches and background level:

```
uv run python scripts/live_e2e.py --preset fast
```

Close the VoiceOnly app first (both would use the same outputs).

## Test without other apps

```
uv run voiceonly live --input-file some_clip.wav -o WH-1000XM4 --record out.wav
```

## Optional: VB-CABLE

If VB-CABLE is installed, `-i "CABLE Output"` works the same way, with
CABLE Input as the Windows output.

## Known limits (HTDemucs)
- Trained on music: it keeps *all vocals*, so **singing in songs is kept**.
  A dialogue-trained model (BandIt / DnR) is the planned fix.
- Very loud effects overlapping speech may leave traces.
