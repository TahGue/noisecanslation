"""VoiceOnly desktop app: one click (or Ctrl+Alt+V) between Normal and Voice only.

Normal:     Windows plays straight to your headphones; VoiceOnly is idle.
Voice only: Windows plays to the "capture" output (one you don't listen to);
            VoiceOnly records it, keeps the voices and plays them to your headphones.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from . import separators, winaudio
from .live import DEFAULT_PRESET, PRESETS, LiveConfig, LiveEngine, LoopbackCapture, find_device

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceOnly"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULTS = {"listen": "WH-1000XM4", "capture": "Realtek", "model": "htdemucs",
            "preset": DEFAULT_PRESET, "custom": None}
# Only user choices are saved; preset timings come from code so updates take effect.
SAVED_KEYS = ("listen", "capture", "preset", "custom")

# Advanced sliders: (key, label, min, max, step)
ADVANCED = [("hop_s", "Step", 0.02, 0.25, 0.01),
            ("lookahead_s", "Lookahead", 0.0, 0.40, 0.01),
            ("jitter_s", "Safety buffer", 0.02, 0.30, 0.01),
            ("window_s", "Model window", 1.5, 7.5, 0.5)]

COLORS = {"bg": "#16181d", "panel": "#20232b", "text": "#e8eaf0", "muted": "#8a90a0",
          "voice": "#3ddc97", "normal": "#5b8def", "off": "#2c3039", "error": "#ff6b6b"}

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x0312

log = logging.getLogger("voiceonly.app")


def load_config() -> dict:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in SAVED_KEYS}}
        if cfg["preset"] not in PRESETS and not (cfg["preset"] == "custom" and cfg["custom"]):
            cfg["preset"] = DEFAULT_PRESET
        return cfg
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({k: cfg[k] for k in SAVED_KEYS}, indent=2),
                           encoding="utf-8")


def _output_device(name: str):
    try:
        return find_device(name, "output")
    except RuntimeError:
        # sounddevice and soundcard can spell names differently; try the part in brackets.
        inner = name[name.find("(") + 1:name.rfind(")")] if "(" in name else name
        return find_device(inner, "output")


class App:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.mode = "normal"
        self.separator = None
        self.engine: LiveEngine | None = None
        self.engine_thread: threading.Thread | None = None
        self.events: queue.Queue[str] = queue.Queue()
        self.combos: list[ttk.Combobox] = []
        self.next_device_check = 0.0

        self.root = tk.Tk()
        self.root.title("VoiceOnly")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_ui()
        self._fix_leftover_routing()

        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------- UI ----------

    def _build_ui(self) -> None:
        pad = {"padx": 16}
        tk.Label(self.root, text="VoiceOnly", font=("Segoe UI Semibold", 16),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", pady=(14, 0), **pad)
        tk.Label(self.root, text="Hear only the voices from your PC", font=("Segoe UI", 9),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", **pad)

        row = tk.Frame(self.root, bg=COLORS["bg"])
        row.pack(fill="x", pady=14, **pad)
        self.btn_normal = self._mode_button(row, "Normal", "normal")
        self.btn_voice = self._mode_button(row, "Voice only", "voice")
        self.btn_normal.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_voice.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.status = tk.Label(self.root, text="Loading model…", font=("Segoe UI", 9),
                               bg=COLORS["bg"], fg=COLORS["muted"], anchor="w", justify="left",
                               wraplength=330)
        self.status.pack(fill="x", **pad)

        settings = tk.Frame(self.root, bg=COLORS["panel"])
        settings.pack(fill="x", pady=(14, 0), padx=16)
        names = [s.name for s in winaudio.speakers()]
        self.listen_var = tk.StringVar(value=self._match(names, self.cfg["listen"]))
        self.capture_var = tk.StringVar(value=self._match(names, self.cfg["capture"]))
        self._combo(settings, "Listen on", self.listen_var, names, 0)
        self._combo(settings, "Capture via (muted)", self.capture_var, names, 1)
        settings.columnconfigure(1, weight=1)

        self._build_latency(pad)

        self.hint = tk.Label(self.root, text="Ctrl+Alt+V switches mode from any app",
                             font=("Segoe UI", 8), bg=COLORS["bg"], fg=COLORS["muted"])
        self.hint.pack(anchor="w", pady=(8, 14), **pad)
        self._refresh_buttons()
        self._refresh_latency()

    def _build_latency(self, pad: dict) -> None:
        head = tk.Frame(self.root, bg=COLORS["bg"])
        head.pack(fill="x", pady=(14, 4), **pad)
        tk.Label(head, text="Latency", font=("Segoe UI Semibold", 10), bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left")
        self.delay_label = tk.Label(head, text="", font=("Segoe UI", 9), bg=COLORS["bg"],
                                    fg=COLORS["muted"])
        self.delay_label.pack(side="left", padx=8)
        self.adv_button = tk.Button(head, text="Advanced \u25b8", font=("Segoe UI", 8),
                                    relief="flat", bd=0, bg=COLORS["bg"], fg=COLORS["muted"],
                                    cursor="hand2", activebackground=COLORS["bg"],
                                    command=self._toggle_advanced)
        self.adv_button.pack(side="right")

        row = tk.Frame(self.root, bg=COLORS["bg"])
        row.pack(fill="x", **pad)
        self.preset_buttons = {}
        for key, preset in PRESETS.items():
            b = tk.Button(row, text=preset.label, font=("Segoe UI", 9), relief="flat", bd=0,
                          cursor="hand2", width=8, command=lambda k=key: self.set_preset(k))
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.preset_buttons[key] = b

        self.adv_frame = tk.Frame(self.root, bg=COLORS["panel"])
        self.adv_vars = {}
        for i, (key, label, lo, hi, step) in enumerate(ADVANCED):
            tk.Label(self.adv_frame, text=label, font=("Segoe UI", 9), bg=COLORS["panel"],
                     fg=COLORS["muted"]).grid(row=i, column=0, sticky="w", padx=10, pady=2)
            var = tk.DoubleVar(value=self._timing()[key])
            scale = tk.Scale(self.adv_frame, variable=var, from_=lo, to=hi, resolution=step,
                             orient="horizontal", length=200, showvalue=True, bd=0,
                             highlightthickness=0, bg=COLORS["panel"], fg=COLORS["text"],
                             troughcolor=COLORS["off"], font=("Segoe UI", 8))
            scale.grid(row=i, column=1, sticky="ew", padx=(0, 10))
            scale.bind("<ButtonRelease-1>", lambda _e: self._advanced_changed())
            self.adv_vars[key] = var
        self.adv_visible = False

    def _toggle_advanced(self) -> None:
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_frame.pack(fill="x", padx=16, pady=(6, 0), before=self.hint)
            self.adv_button.configure(text="Advanced \u25be")
        else:
            self.adv_frame.pack_forget()
            self.adv_button.configure(text="Advanced \u25b8")

    def _timing(self) -> dict:
        """Current timing values: the custom set, or the selected preset's."""
        if self.cfg["preset"] == "custom" and self.cfg["custom"]:
            return dict(self.cfg["custom"])
        p = PRESETS[self.cfg["preset"]]
        return {"hop_s": p.hop_s, "lookahead_s": p.lookahead_s, "jitter_s": p.jitter_s,
                "window_s": p.window_s, "crossfade_s": p.crossfade_s}

    def _refresh_latency(self) -> None:
        for key, b in self.preset_buttons.items():
            active = self.cfg["preset"] == key
            color = COLORS["voice"] if active else COLORS["off"]
            b.configure(bg=color, activebackground=color,
                        fg="#0d0f12" if active else COLORS["text"])
        t = self._timing()
        delay = t["hop_s"] + t["lookahead_s"] + t.get("crossfade_s", 0.02) + t["jitter_s"]
        name = "Custom" if self.cfg["preset"] == "custom" else PRESETS[self.cfg["preset"]].label
        self.delay_label.configure(text=f"{name} \u00b7 ~{delay:.2f} s + device")
        for key, var in self.adv_vars.items():
            var.set(t[key])

    def set_preset(self, key: str) -> None:
        self.cfg["preset"] = key
        save_config(self.cfg)
        self._refresh_latency()
        self._restart_engine()

    def _advanced_changed(self) -> None:
        t = {k: round(v.get(), 3) for k, v in self.adv_vars.items()}
        t["crossfade_s"] = round(min(0.05, t["hop_s"] / 2), 3)
        # The model window must hold the step, lookahead and crossfade plus some context.
        t["window_s"] = max(t["window_s"],
                            round(t["hop_s"] + t["lookahead_s"] + t["crossfade_s"] + 0.5, 1))
        self.cfg["preset"], self.cfg["custom"] = "custom", t
        save_config(self.cfg)
        self._refresh_latency()
        self._restart_engine()

    def _restart_engine(self) -> None:
        """Apply new timing without touching Windows' output routing."""
        if self.mode != "voice":
            return
        self._halt_engine()
        try:
            self._launch_engine()
        except Exception as e:
            log.exception("restart failed")
            self.set_mode("normal")
            self._set_status(f"Could not apply setting: {e}", error=True)

    def _mode_button(self, parent, text: str, mode: str) -> tk.Button:
        return tk.Button(parent, text=text, font=("Segoe UI Semibold", 12), height=2, width=11,
                         relief="flat", bd=0, cursor="hand2", activeforeground=COLORS["text"],
                         command=lambda: self.set_mode(mode))

    def _combo(self, parent, label: str, var: tk.StringVar, values: list[str], row: int) -> None:
        tk.Label(parent, text=label, font=("Segoe UI", 9), bg=COLORS["panel"],
                 fg=COLORS["muted"]).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=30)
        box.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=6)
        self.combos.append(box)
        box.bind("<<ComboboxSelected>>", lambda _e: self._settings_changed())

    @staticmethod
    def _match(names: list[str], fragment: str) -> str:
        return next((n for n in names if fragment.lower() in n.lower()), names[0] if names else "")

    def _refresh_buttons(self) -> None:
        for btn, mode in ((self.btn_normal, "normal"), (self.btn_voice, "voice")):
            active = self.mode == mode
            color = COLORS[mode] if active else COLORS["off"]
            btn.configure(bg=color, activebackground=color,
                          fg="#0d0f12" if active else COLORS["text"])
        ready = self.separator is not None
        self.btn_voice.configure(state="normal" if ready else "disabled")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, fg=COLORS["error"] if error else COLORS["muted"])

    def _settings_changed(self) -> None:
        self.cfg["listen"], self.cfg["capture"] = self.listen_var.get(), self.capture_var.get()
        save_config(self.cfg)
        if self.mode == "voice":  # apply immediately
            self.set_mode("normal")
            self.set_mode("voice")

    # ---------- modes ----------

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        try:
            if mode == "voice":
                self._start_voice()
            else:
                self._stop_voice()
            self.mode = mode
        except Exception as e:  # never leave the user without sound
            log.exception("mode switch failed")
            self._stop_voice()
            self.mode = "normal"
            self._set_status(f"Could not switch: {e}", error=True)
        self._refresh_buttons()

    def toggle(self) -> None:
        self.set_mode("normal" if self.mode == "voice" else "voice")

    def _start_voice(self) -> None:
        if self.separator is None:
            raise RuntimeError("model still loading")
        listen, capture = self.listen_var.get(), self.capture_var.get()
        if listen == capture:
            raise RuntimeError("'Listen on' and 'Capture via' must be different outputs")
        self._launch_engine()
        winaudio.set_default_speaker(winaudio.find_speaker(capture).id)

    def _launch_engine(self) -> None:
        t = self._timing()
        config = LiveConfig(hop_s=t["hop_s"], lookahead_s=t["lookahead_s"],
                            window_s=t["window_s"], crossfade_s=t.get("crossfade_s", 0.02))
        engine = LiveEngine(self.separator, config, LoopbackCapture(self.capture_var.get()),
                            _output_device(self.listen_var.get()), jitter_s=t["jitter_s"])
        self.engine = engine
        self.engine_thread = threading.Thread(target=self._run_engine, args=(engine,),
                                              daemon=True)
        self.engine_thread.start()
        self._set_status("Voice only: starting\u2026")

    def _halt_engine(self) -> None:
        if self.engine is not None:
            self.engine.running = False
            if self.engine_thread is not None:
                self.engine_thread.join(timeout=3)
            self.engine, self.engine_thread = None, None

    def _run_engine(self, engine: LiveEngine) -> None:
        try:
            engine.run()
        except Exception as e:
            log.exception("engine stopped")
            engine.error = e
            self.events.put("engine_failed")

    def _stop_voice(self) -> None:
        self._halt_engine()
        try:
            names = [sp.name for sp in winaudio.speakers()]
            target = self.listen_var.get() if self.listen_var.get() in names                 else self._fallback_output(names)
            if target:
                winaudio.set_default_speaker(winaudio.find_speaker(target).id)
        except Exception:
            log.exception("could not restore default output")
        if self.separator is not None:
            self._set_status("Normal: sound plays directly to " + self.listen_var.get())

    def _fallback_output(self, names: list[str]) -> str | None:
        """A real output to listen on: Windows' default unless that is the capture device."""
        capture = self.capture_var.get()
        try:
            default = winaudio.default_speaker().name
        except Exception:
            default = None
        if default in names and default != capture:
            return default
        return next((n for n in names if n != capture), None)

    def _check_devices(self) -> None:
        """Follow plugged/unplugged outputs (e.g. Bluetooth headphones turning off)."""
        try:
            names = [sp.name for sp in winaudio.speakers()]
        except Exception:
            log.exception("device scan failed")
            return
        for box in self.combos:
            box.configure(values=names)
        listen, capture = self.listen_var.get(), self.capture_var.get()
        if listen not in names or capture not in names:
            lost = listen if listen not in names else capture
            if self.mode == "voice":
                self.set_mode("normal")
            if listen not in names:
                self.listen_var.set(self._fallback_output(names) or "")
            self._set_status(f"'{lost}' disconnected. Listening on {self.listen_var.get()}.",
                             error=True)
            return
        preferred = self._match(names, self.cfg["listen"])
        if (self.mode == "normal" and preferred != listen and preferred != capture
                and self.cfg["listen"].lower() in preferred.lower()):
            self.listen_var.set(preferred)  # saved choice is back (e.g. headphones on again)
            self._set_status(f"Listening on {preferred} again.")

    def _fix_leftover_routing(self) -> None:
        """If a previous run crashed while in Voice only, give the user their sound back."""
        try:
            if winaudio.default_speaker().name == self.capture_var.get():
                winaudio.set_default_speaker(winaudio.find_speaker(self.listen_var.get()).id)
        except Exception:
            log.exception("startup routing check failed")

    # ---------- background work ----------

    def _load_model(self) -> None:
        try:
            sep = separators.create(self.cfg["model"])
            sep.initialize()
            self.separator = sep
            self.events.put("model_ready")
        except Exception as e:
            log.exception("model load failed")
            self.load_error = e
            self.events.put("model_failed")

    def _hotkey_loop(self) -> None:
        user32 = ctypes.windll.user32
        self.hotkey_thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, ord("V")):
            self.events.put("hotkey_failed")
            return
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.events.put("toggle")
        user32.UnregisterHotKey(None, 1)

    def _poll(self) -> None:
        if time.monotonic() >= self.next_device_check:
            self.next_device_check = time.monotonic() + 2.0
            self._check_devices()
        while not self.events.empty():
            ev = self.events.get()
            if ev == "toggle":
                self.toggle()
            elif ev == "model_ready":
                self._set_status("Ready. Normal: sound plays directly to " + self.listen_var.get())
                self._refresh_buttons()
            elif ev == "model_failed":
                self._set_status(f"Model failed to load: {self.load_error}", error=True)
            elif ev == "engine_failed":
                err = self.engine.error if self.engine else "unknown error"
                self.set_mode("normal")
                self._set_status(f"Voice only stopped: {err}", error=True)
            elif ev == "hotkey_failed":
                self._set_status("Ctrl+Alt+V is used by another app; use the buttons.",
                                 error=True)
        if self.mode == "voice" and self.engine is not None and self.engine.stats.steps:
            s = self.engine.stats
            ms = list(s.step_ms)
            gpu = sum(ms) / len(ms) if ms else 0.0
            glitches = s.underruns + s.dropped_blocks + s.drift_drops
            self._set_status(f"Voice only · delay ~{self.engine.latency_s:.2f} s · "
                             f"GPU {gpu:.0f} ms/step · glitches {glitches}")
        self.root.after(100, self._poll)

    def quit(self) -> None:
        self.set_mode("normal")
        if getattr(self, "hotkey_thread_id", None):
            ctypes.windll.user32.PostThreadMessageW(self.hotkey_thread_id, 0x0012, 0, 0)
        if self.separator is not None:
            self.separator.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=CONFIG_DIR / "voiceonly.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
    except Exception:
        pass
    App().run()


if __name__ == "__main__":
    main()
