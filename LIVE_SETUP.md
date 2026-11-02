# Live mode: hear only voices from your PC

```
 apps (YouTube, games, Netflix...)                 VoiceOnly                    you
 ───────────> Realtek Digital Output ──loopback──> remove music/SFX ──> WH-1000XM4 headphones
              (Windows default, not listened to)   (RTX 4090)
```

No driver is needed: VoiceOnly records what Windows sends to an output you
don't listen to (WASAPI loopback) and plays the filtered result to the one you do.

## Each time you want voice only

1. Windows sound settings (Win+Ctrl+V, or the speaker icon) → output **Realtek Digital Output**.
2. Double-click `start-voiceonly.bat`.
3. In the window: **Enter** toggles VOICE ONLY / BYPASS, **Ctrl+C** stops.
4. When done, switch Windows output back to your headphones.

Requirements for the "sink" output (`SINK=` in the .bat):
- You must not hear it: nothing connected to Realtek Digital Output, or if you use
  the monitor (`SINK=PHL`), turn the monitor's own speaker volume to zero.
- Keep its **Windows** volume up; if VoiceOnly sounds quiet, raise it.
- It must differ from `OUTPUT=` (VoiceOnly refuses a loop).

`uv run voiceonly devices` lists outputs and loopback sources.

## Measured on this PC (2026-11-02)

App → Realtek → VoiceOnly → WH-1000XM4, HTDemucs, 0.25 s steps, over 42 s:

| | |
|---|---|
| Underruns / dropped audio | 0 / 0 |
| Delay, app to headphones | 0.89 s (+ Bluetooth's own ~0.2 s) |
| GPU time per 250 ms step | ~80 ms (p99 86 ms) |
| Tonal balance vs offline result | within 1 dB |
| Background during voice pauses | 23 dB below speech (unfiltered mix: 5 dB) |

Video lip-sync will be ~1 s behind; VLC (`J`/`K`) and MPC-HC can shift audio sync.

## Tuning

| Setting | Default | Effect |
|---|---|---|
| `--hop` | 0.25 s | Smaller = less delay, more GPU work |
| `--lookahead` | 0.3 s | More = cleaner speech onsets, more delay |
| `--jitter` | 0.2 s | Safety buffer against stalls; raise if `underruns` climb |

The buffer self-corrects by changing playback speed by at most 1% (inaudible),
so stalls and clock differences don't cut audio.

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
