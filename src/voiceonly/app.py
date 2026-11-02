"""VoiceOnly desktop app: choose what you hear from your PC, live.

Normal:   Windows plays straight to your headphones; VoiceOnly is idle.
Filtered: Windows plays to the "capture" output (one you don't listen to);
          VoiceOnly separates it into stems (voice, music, effects, ...) and plays
          the mix you chose (voice only, no music, ...) to your headphones.

Three choices drive it: the model (speed vs quality vs which stems exist), what to
keep (a mix mode or custom per-stem levels) and the latency preset.
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
from .live import DEFAULT_PRESET, LiveConfig, LiveEngine, LoopbackCapture, find_device, presets_for
from .mixer import DEFAULT_MODE, MODES, available, gains_for

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceOnly"
CONFIG_PATH = CONFIG_DIR / "config.json"

# (registry key, label, one-line description)
MODEL_CHOICES = [
    ("htdemucs", "Fast · HTDemucs", "Lightest. Stems: voice, drums, bass, other."),
    ("bs-roformer-live", "Best voice · BS-RoFormer",
     "Cleanest voice (~9 dB better). Stems: voice, background."),
    ("mrx", "Cinematic · MRX", "Splits music from sound effects. Stems: voice, music, effects."),
]
MODEL_KEYS = [k for k, _, _ in MODEL_CHOICES]

DEFAULTS = {"listen": "WH-1000XM4", "capture": "Realtek", "model": "htdemucs",
            "preset": DEFAULT_PRESET, "custom": None, "mix": DEFAULT_MODE, "gains": {}}
# Only user choices are saved; preset timings come from code so updates take effect.
SAVED_KEYS = ("listen", "capture", "model", "preset", "custom", "mix", "gains")

# Advanced sliders: (key, label, min, max, step)
ADVANCED = [("hop_s", "Step", 0.02, 0.25, 0.01),
            ("lookahead_s", "Lookahead", 0.0, 0.40, 0.01),
            ("jitter_s", "Safety buffer", 0.02, 0.30, 0.01),
            ("window_s", "Model window", 1.0, 7.5, 0.5)]

COLORS = {"bg": "#16181d", "panel": "#20232b", "text": "#e8eaf0", "muted": "#8a90a0",
          "filter": "#3ddc97", "normal": "#5b8def", "off": "#2c3039", "error": "#ff6b6b",
          "accent": "#f2c14e"}

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x0312

log = logging.getLogger("voiceonly.app")


def load_config() -> dict:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in SAVED_KEYS}}
    except (OSError, ValueError):
        cfg = dict(DEFAULTS)
    if cfg["model"] not in MODEL_KEYS:
        cfg["model"] = DEFAULTS["model"]
    if cfg["preset"] not in presets_for(cfg["model"]) and not (
            cfg["preset"] == "custom" and cfg["custom"]):
        cfg["preset"] = DEFAULT_PRESET
    if cfg["mix"] not in MODES and cfg["mix"] != "custom":
        cfg["mix"] = DEFAULT_MODE
    return cfg


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


def _button(parent, text: str, command, size: int = 9, **kw) -> tk.Button:
    return tk.Button(parent, text=text, font=("Segoe UI", size), relief="flat", bd=0,
                     cursor="hand2", command=command, **kw)


def _paint(button: tk.Button, active: bool, color: str, enabled: bool = True) -> None:
    bg = color if active else COLORS["off"]
    button.configure(bg=bg, activebackground=bg, state="normal" if enabled else "disabled",
                     fg="#0d0f12" if active else (COLORS["text"] if enabled else COLORS["muted"]),
                     disabledforeground=COLORS["muted"])


class App:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.mode = "normal"
        self.separator = None
        self.loading_key: str | None = None
        self.resume_filter = False  # go back to Filtered once a model switch finishes
        self.engine: LiveEngine | None = None
        self.engine_thread: threading.Thread | None = None
        self.events: queue.Queue[tuple] = queue.Queue()
        self.combos: list[ttk.Combobox] = []
        self.next_device_check = 0.0
        self.load_error: Exception | None = None

        self.root = tk.Tk()
        self.root.title("VoiceOnly")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_ui()
        self._fix_leftover_routing()

        self._load_model(self.cfg["model"])
        threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------- layout ----------

    def _build_ui(self) -> None:
        pad = {"padx": 16}
        tk.Label(self.root, text="VoiceOnly", font=("Segoe UI Semibold", 16),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w", pady=(14, 0), **pad)
        tk.Label(self.root, text="Choose what you hear from your PC", font=("Segoe UI", 9),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", **pad)

        row = tk.Frame(self.root, bg=COLORS["bg"])
        row.pack(fill="x", pady=(14, 8), **pad)
        self.btn_normal = _button(row, "Normal", lambda: self.set_mode("normal"), 12,
                                  height=2, width=11)
        self.btn_filter = _button(row, "Voice only", lambda: self.set_mode("filter"), 12,
                                  height=2, width=11)
        self.btn_normal.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_filter.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.status = tk.Label(self.root, text="Loading model…", font=("Segoe UI", 9),
                               bg=COLORS["bg"], fg=COLORS["muted"], anchor="w", justify="left",
                               wraplength=360)
        self.status.pack(fill="x", **pad)

        self._build_keep(pad)
        self._build_model(pad)
        self._build_devices()
        self._build_latency(pad)

        self.hint = tk.Label(self.root, text="Ctrl+Alt+V switches Normal / Filtered from any app",
                             font=("Segoe UI", 8), bg=COLORS["bg"], fg=COLORS["muted"])
        self.hint.pack(anchor="w", pady=(8, 14), **pad)
        self._refresh_all()

    def _section(self, title: str, pad: dict, toggle=None, toggle_text: str = ""):
        head = tk.Frame(self.root, bg=COLORS["bg"])
        head.pack(fill="x", pady=(14, 4), **pad)
        tk.Label(head, text=title, font=("Segoe UI Semibold", 10), bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left")
        note = tk.Label(head, text="", font=("Segoe UI", 9), bg=COLORS["bg"], fg=COLORS["muted"])
        note.pack(side="left", padx=8)
        btn = None
        if toggle:
            btn = _button(head, toggle_text, toggle, 8, bg=COLORS["bg"], fg=COLORS["muted"],
                          activebackground=COLORS["bg"])
            btn.pack(side="right")
        return note, btn

    def _build_keep(self, pad: dict) -> None:
        self.keep_note, self.mixer_button = self._section(
            "Keep", pad, self._toggle_mixer, "Mixer ▸")
        row = tk.Frame(self.root, bg=COLORS["bg"])
        row.pack(fill="x", **pad)
        self.mix_buttons = {}
        for key, mode in MODES.items():
            b = _button(row, mode.label, lambda k=key: self.set_mix(k), 8, width=9)
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.mix_buttons[key] = b
        self.mixer_frame = tk.Frame(self.root, bg=COLORS["panel"])
        self.mixer_vars: dict[str, tk.DoubleVar] = {}
        self.mixer_visible = False

    def _build_model(self, pad: dict) -> None:
        self.model_note, _ = self._section("Model", pad)
        frame = tk.Frame(self.root, bg=COLORS["bg"])
        frame.pack(fill="x", **pad)
        labels = [label for _, label, _ in MODEL_CHOICES]
        self.model_var = tk.StringVar(value=labels[MODEL_KEYS.index(self.cfg["model"])])
        box = ttk.Combobox(frame, textvariable=self.model_var, values=labels, state="readonly")
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>",
                 lambda _e: self.set_model(MODEL_KEYS[labels.index(self.model_var.get())]))
        self.model_desc = tk.Label(frame, text="", font=("Segoe UI", 8), bg=COLORS["bg"],
                                   fg=COLORS["muted"], anchor="w")
        self.model_desc.pack(fill="x", pady=(2, 0))

    def _build_devices(self) -> None:
        settings = tk.Frame(self.root, bg=COLORS["panel"])
        settings.pack(fill="x", pady=(14, 0), padx=16)
        names = [s.name for s in winaudio.speakers()]
        self.listen_var = tk.StringVar(value=self._match(names, self.cfg["listen"]))
        self.capture_var = tk.StringVar(value=self._match(names, self.cfg["capture"]))
        self._combo(settings, "Listen on", self.listen_var, names, 0)
        self._combo(settings, "Capture via (muted)", self.capture_var, names, 1)
        settings.columnconfigure(1, weight=1)

    def _build_latency(self, pad: dict) -> None:
        self.delay_label, self.adv_button = self._section(
            "Latency", pad, self._toggle_advanced, "Advanced ▸")
        self.preset_row = tk.Frame(self.root, bg=COLORS["bg"])
        self.preset_row.pack(fill="x", **pad)
        self.preset_buttons: dict[str, tk.Button] = {}

        self.adv_frame = tk.Frame(self.root, bg=COLORS["panel"])
        self.adv_vars = {}
        for i, (key, label, lo, hi, step) in enumerate(ADVANCED):
            tk.Label(self.adv_frame, text=label, font=("Segoe UI", 9), bg=COLORS["panel"],
                     fg=COLORS["muted"]).grid(row=i, column=0, sticky="w", padx=10, pady=2)
            var = tk.DoubleVar(value=0.0)
            scale = self._scale(self.adv_frame, var, lo, hi, step)
            scale.grid(row=i, column=1, sticky="ew", padx=(0, 10))
            scale.bind("<ButtonRelease-1>", lambda _e: self._advanced_changed())
            self.adv_vars[key] = var
        self.adv_visible = False

    @staticmethod
    def _scale(parent, var, lo, hi, step) -> tk.Scale:
        return tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=step,
                        orient="horizontal", length=220, showvalue=True, bd=0,
                        highlightthickness=0, bg=COLORS["panel"], fg=COLORS["text"],
                        troughcolor=COLORS["off"], font=("Segoe UI", 8))

    def _combo(self, parent, label: str, var: tk.StringVar, values: list[str], row: int) -> None:
        tk.Label(parent, text=label, font=("Segoe UI", 9), bg=COLORS["panel"],
                 fg=COLORS["muted"]).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=30)
        box.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=6)
        self.combos.append(box)
        box.bind("<<ComboboxSelected>>", lambda _e: self._devices_changed())

    @staticmethod
    def _match(names: list[str], fragment: str) -> str:
        return next((n for n in names if fragment.lower() in n.lower()), names[0] if names else "")

    def _toggle_mixer(self) -> None:
        # The mixer sits under the mode buttons row.
        if self.mixer_visible:
            self.mixer_frame.pack_forget()
            self.mixer_button.configure(text="Mixer ▸")
        else:
            self.mixer_frame.pack(fill="x", padx=16, pady=(6, 0),
                                  after=self.mix_buttons[DEFAULT_MODE].master)
            self.mixer_button.configure(text="Mixer ▾")
        self.mixer_visible = not self.mixer_visible

    def _toggle_advanced(self) -> None:
        if self.adv_visible:
            self.adv_frame.pack_forget()
            self.adv_button.configure(text="Advanced ▸")
        else:
            self.adv_frame.pack(fill="x", padx=16, pady=(6, 0), after=self.preset_row)
            self.adv_button.configure(text="Advanced ▾")
        self.adv_visible = not self.adv_visible

    # ---------- state -> widgets ----------

    def _stems(self) -> list[str]:
        return separators.REGISTRY[self.cfg["model"]].info.stems

    def _gains(self) -> dict[str, float]:
        stems = self._stems()
        if self.cfg["mix"] == "custom":
            saved = self.cfg["gains"].get(self.cfg["model"], {})
            return {s: float(saved.get(s, 1.0 if s == "voice" else 0.0)) for s in stems}
        return gains_for(self.cfg["mix"], stems)

    def _mix_label(self) -> str:
        return "Custom mix" if self.cfg["mix"] == "custom" else MODES[self.cfg["mix"]].label

    def _refresh_all(self) -> None:
        self._refresh_buttons()
        self._refresh_keep()
        self._refresh_model()
        self._refresh_latency()

    def _refresh_buttons(self) -> None:
        _paint(self.btn_normal, self.mode == "normal", COLORS["normal"])
        ready = self.separator is not None and self.loading_key is None
        self.btn_filter.configure(text=self._mix_label())
        _paint(self.btn_filter, self.mode == "filter", COLORS["filter"], enabled=ready)

    def _refresh_keep(self) -> None:
        stems = self._stems()
        for key, b in self.mix_buttons.items():
            _paint(b, self.cfg["mix"] == key, COLORS["filter"], enabled=available(key, stems))
        unavailable = [MODES[k].label for k in MODES if not available(k, stems)]
        self.keep_note.configure(
            text=f"{', '.join(unavailable)}: use Cinematic" if unavailable else "")
        # One slider per stem of the current model.
        for child in self.mixer_frame.winfo_children():
            child.destroy()
        self.mixer_vars = {}
        for i, (stem, gain) in enumerate(self._gains().items()):
            tk.Label(self.mixer_frame, text=stem.capitalize(), font=("Segoe UI", 9),
                     bg=COLORS["panel"], fg=COLORS["muted"]).grid(
                row=i, column=0, sticky="w", padx=10, pady=2)
            var = tk.DoubleVar(value=round(gain * 100))
            scale = self._scale(self.mixer_frame, var, 0, 100, 5)
            scale.grid(row=i, column=1, sticky="ew", padx=(0, 10))
            scale.bind("<ButtonRelease-1>", lambda _e: self._mixer_changed())
            self.mixer_vars[stem] = var

    def _refresh_model(self) -> None:
        key = self.loading_key or self.cfg["model"]
        self.model_var.set(next(label for k, label, _ in MODEL_CHOICES if k == key))
        desc = next(d for k, _, d in MODEL_CHOICES if k == key)
        self.model_desc.configure(text=desc)
        self.model_note.configure(text="loading…" if self.loading_key else "")

    def _refresh_latency(self) -> None:
        presets = presets_for(self.cfg["model"])
        if list(self.preset_buttons) != list(presets):
            for b in self.preset_buttons.values():
                b.destroy()
            self.preset_buttons = {}
            for key, preset in presets.items():
                b = _button(self.preset_row, preset.label, lambda k=key: self.set_preset(k), 9,
                            width=8)
                b.pack(side="left", expand=True, fill="x", padx=2)
                self.preset_buttons[key] = b
        for key, b in self.preset_buttons.items():
            _paint(b, self.cfg["preset"] == key, COLORS["filter"])
        t = self._timing()
        delay = t["hop_s"] + t["lookahead_s"] + t.get("crossfade_s", 0.02) + t["jitter_s"]
        name = "Custom" if self.cfg["preset"] == "custom" else presets[self.cfg["preset"]].label
        self.delay_label.configure(text=f"{name} · ~{delay:.2f} s + device")
        for key, var in self.adv_vars.items():
            var.set(t[key])

    def _timing(self) -> dict:
        """Current timing values: the custom set, or the selected preset's."""
        if self.cfg["preset"] == "custom" and self.cfg["custom"]:
            return dict(self.cfg["custom"])
        presets = presets_for(self.cfg["model"])
        p = presets.get(self.cfg["preset"]) or presets[DEFAULT_PRESET]
        return {"hop_s": p.hop_s, "lookahead_s": p.lookahead_s, "jitter_s": p.jitter_s,
                "window_s": p.window_s, "crossfade_s": p.crossfade_s}

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, fg=COLORS["error"] if error else COLORS["muted"])

    # ---------- user choices ----------

    def set_mix(self, key: str) -> None:
        self.cfg["mix"] = key
        save_config(self.cfg)
        self._apply_gains()

    def _mixer_changed(self) -> None:
        gains = {s: round(v.get() / 100, 2) for s, v in self.mixer_vars.items()}
        self.cfg["gains"] = {**self.cfg["gains"], self.cfg["model"]: gains}
        self.cfg["mix"] = "custom"
        save_config(self.cfg)
        self._apply_gains()

    def _apply_gains(self) -> None:
        """New mix takes effect immediately; no restart needed."""
        if self.engine is not None:
            self.engine.set_gains(self._gains())
        self._refresh_buttons()
        self._refresh_keep()

    def set_model(self, key: str) -> None:
        if key == self.cfg["model"] and self.separator is not None:
            return
        self.resume_filter = self.mode == "filter"
        if self.mode == "filter":
            self.set_mode("normal")
        self.cfg["model"] = key
        if self.cfg["preset"] not in presets_for(key) and self.cfg["preset"] != "custom":
            self.cfg["preset"] = DEFAULT_PRESET
        if self.cfg["mix"] != "custom" and not available(self.cfg["mix"], self._stems()):
            self.cfg["mix"] = DEFAULT_MODE
        save_config(self.cfg)
        self._load_model(key)

    def set_preset(self, key: str) -> None:
        self.cfg["preset"] = key
        save_config(self.cfg)
        self._refresh_latency()
        self._restart_engine()

    def _advanced_changed(self) -> None:
        t = {k: round(v.get(), 3) for k, v in self.adv_vars.items()}
        min_hop = separators.REGISTRY[self.cfg["model"]].info.live_min_hop_seconds
        t["hop_s"] = max(t["hop_s"], min_hop)
        t["crossfade_s"] = round(min(0.05, t["hop_s"] / 2), 3)
        # The model window must hold the step, lookahead and crossfade plus some context.
        t["window_s"] = max(t["window_s"],
                            round(t["hop_s"] + t["lookahead_s"] + t["crossfade_s"] + 0.5, 1))
        self.cfg["preset"], self.cfg["custom"] = "custom", t
        save_config(self.cfg)
        self._refresh_latency()
        self._restart_engine()

    def _devices_changed(self) -> None:
        self.cfg["listen"], self.cfg["capture"] = self.listen_var.get(), self.capture_var.get()
        save_config(self.cfg)
        if self.mode == "filter":  # apply immediately
            self.set_mode("normal")
            self.set_mode("filter")

    # ---------- modes ----------

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        try:
            if mode == "filter":
                self._start_filter()
            else:
                self._stop_filter()
            self.mode = mode
        except Exception as e:  # never leave the user without sound
            log.exception("mode switch failed")
            self._stop_filter()
            self.mode = "normal"
            self._set_status(f"Could not switch: {e}", error=True)
        self._refresh_buttons()

    def toggle(self) -> None:
        self.set_mode("normal" if self.mode == "filter" else "filter")

    def _start_filter(self) -> None:
        if self.separator is None or self.loading_key:
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
                            _output_device(self.listen_var.get()), jitter_s=t["jitter_s"],
                            gains=self._gains())
        self.engine = engine
        self.engine_thread = threading.Thread(target=self._run_engine, args=(engine,),
                                              daemon=True)
        self.engine_thread.start()
        self._set_status(f"{self._mix_label()}: starting…")

    def _restart_engine(self) -> None:
        """Apply new timing without touching Windows' output routing."""
        if self.mode != "filter":
            return
        self._halt_engine()
        try:
            self._launch_engine()
        except Exception as e:
            log.exception("restart failed")
            self.set_mode("normal")
            self._set_status(f"Could not apply setting: {e}", error=True)

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
            self.events.put(("engine_failed", e))

    def _stop_filter(self) -> None:
        self._halt_engine()
        try:
            names = [sp.name for sp in winaudio.speakers()]
            target = (self.listen_var.get() if self.listen_var.get() in names
                      else self._fallback_output(names))
            if target:
                winaudio.set_default_speaker(winaudio.find_speaker(target).id)
        except Exception:
            log.exception("could not restore default output")
        if self.separator is not None:
            self._set_status("Normal: sound plays directly to " + self.listen_var.get())

    # ---------- devices ----------

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
            if self.mode == "filter":
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
        """If a previous run crashed while filtering, give the user their sound back."""
        try:
            if winaudio.default_speaker().name == self.capture_var.get():
                winaudio.set_default_speaker(winaudio.find_speaker(self.listen_var.get()).id)
        except Exception:
            log.exception("startup routing check failed")

    # ---------- background work ----------

    def _load_model(self, key: str) -> None:
        self.loading_key = key
        old, self.separator = self.separator, None
        self._refresh_all()
        self._set_status(f"Loading {next(lbl for k, lbl, _ in MODEL_CHOICES if k == key)}…")

        def work() -> None:
            if old is not None:
                old.close()  # free GPU memory before loading the next model
            try:
                sep = separators.create(key)
                sep.initialize()
                self.events.put(("model_ready", key, sep))
            except Exception as e:
                log.exception("model load failed")
                self.events.put(("model_failed", key, e))

        threading.Thread(target=work, daemon=True).start()

    def _hotkey_loop(self) -> None:
        user32 = ctypes.windll.user32
        self.hotkey_thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, ord("V")):
            self.events.put(("hotkey_failed",))
            return
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.events.put(("toggle",))
        user32.UnregisterHotKey(None, 1)

    def _poll(self) -> None:
        if time.monotonic() >= self.next_device_check:
            self.next_device_check = time.monotonic() + 2.0
            self._check_devices()
        while not self.events.empty():
            ev = self.events.get()
            kind = ev[0]
            if kind == "toggle":
                self.toggle()
            elif kind == "model_ready":
                _, key, sep = ev
                if key != self.cfg["model"]:
                    sep.close()  # user picked another model meanwhile
                    continue
                self.separator, self.loading_key = sep, None
                self._refresh_all()
                self._set_status("Ready. Normal: sound plays directly to "
                                 + self.listen_var.get())
                if self.resume_filter:
                    self.resume_filter = False
                    self.set_mode("filter")
            elif kind == "model_failed":
                _, key, err = ev
                if key == self.cfg["model"]:
                    self.loading_key, self.load_error = None, err
                    self._refresh_all()
                    self._set_status(f"Model failed to load: {err}", error=True)
            elif kind == "engine_failed":
                self.set_mode("normal")
                self._set_status(f"Filtering stopped: {ev[1]}", error=True)
            elif kind == "hotkey_failed":
                self._set_status("Ctrl+Alt+V is used by another app; use the buttons.",
                                 error=True)
        if self.mode == "filter" and self.engine is not None and self.engine.stats.steps:
            s = self.engine.stats
            ms = list(s.step_ms)
            gpu = sum(ms) / len(ms) if ms else 0.0
            glitches = s.underruns + s.dropped_blocks + s.drift_drops
            self._set_status(f"{self._mix_label()} · delay ~{self.engine.latency_s:.2f} s"
                             f" · GPU {gpu:.0f} ms/step · glitches {glitches}")
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
    # Without a console (pythonw) sys.stdout/stderr are None, and any library that
    # prints a download progress bar (Hugging Face, audio-separator) crashes on it.
    import sys

    if sys.stdout is None or sys.stderr is None:
        sink = open(CONFIG_DIR / "voiceonly-console.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or sink
        sys.stderr = sys.stderr or sink
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
