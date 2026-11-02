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

import customtkinter as ctk

from . import separators, winaudio
from .doctor import load_benchmarks, report
from .live import DEFAULT_PRESET, LiveConfig, LiveEngine, LoopbackCapture, find_device, presets_for
from .mixer import DEFAULT_MODE, MODES, available, gains_for
from .separators.base import GPU_LOCK

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceOnly"
CONFIG_PATH = CONFIG_DIR / "config.json"

# (registry key, label, one-line description)
MODEL_CHOICES = [
    ("htdemucs", "Fast · HTDemucs", "Lightest on the GPU. Voice, drums, bass, other."),
    ("tiger", "Cinematic · TIGER", "Separates music from sound effects. Voice, music, effects."),
    ("bs-roformer-live", "Best voice · BS-RoFormer", "Cleanest dialogue. Voice, background."),
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

COLORS = {"bg": "#111318", "card": "#1b1e25", "card_hi": "#232733", "line": "#2e3340",
          "text": "#e9ecf2", "muted": "#8a91a2", "filter": "#34d399", "filter_dark": "#0f3d2e",
          "normal": "#5b8def", "off": "#2a2f3a", "error": "#ff6b6b", "warn": "#f2c14e"}
FONT = "Segoe UI"

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x0312

log = logging.getLogger("voiceonly.app")


def load_config() -> dict:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in SAVED_KEYS}}
    except (OSError, ValueError):
        cfg = dict(DEFAULTS)
    if cfg["model"] == "mrx":
        cfg["model"] = "tiger"  # Cinematic moved from MRX to TIGER (better on every stem)
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


def _label(parent, text: str = "", size: int = 12, weight: str = "normal",
           color: str = COLORS["text"], **kw) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(FONT, size, weight),
                        text_color=color, **kw)


def _chip_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(parent, text=text, command=command, height=32, corner_radius=8,
                         font=ctk.CTkFont(FONT, 12), fg_color=COLORS["off"],
                         hover_color=COLORS["card_hi"], text_color=COLORS["text"],
                         text_color_disabled=COLORS["muted"])


def _paint(button: ctk.CTkButton, active: bool, color: str, enabled: bool = True) -> None:
    button.configure(fg_color=color if active else COLORS["off"],
                     hover_color=color if active else COLORS["card_hi"],
                     text_color="#0b0d10" if active else COLORS["text"],
                     state="normal" if enabled else "disabled")


class App:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.mode = "normal"
        self.separator = None
        self.loading_key: str | None = None
        self.resume_filter = False  # go back to Filtered once a model switch finishes
        self.engine: LiveEngine | None = None
        self.engine_thread: threading.Thread | None = None
        # Stopped engines that were still warming up; finished before a model is freed.
        self.stopping: list[threading.Thread] = []
        self.events: queue.Queue[tuple] = queue.Queue()
        self.combos: list[ctk.CTkOptionMenu] = []
        self.next_device_check = 0.0
        self.load_error: Exception | None = None
        self.bench = load_benchmarks().get("models", {})

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk(fg_color=COLORS["bg"])
        self.root.title("VoiceOnly")
        self.root.geometry("480x900")
        self.root.minsize(440, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_ui()
        self._fix_leftover_routing()

        self._load_model(self.cfg["model"])
        threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------- layout ----------

    def _build_ui(self) -> None:
        body = ctk.CTkScrollableFrame(self.root, fg_color=COLORS["bg"], corner_radius=0)
        body.pack(fill="both", expand=True)
        self.body = body
        pad = {"padx": 18}

        head = ctk.CTkFrame(body, fg_color="transparent")
        head.pack(fill="x", pady=(16, 2), **pad)
        _label(head, "VoiceOnly", 22, "bold").pack(side="left")
        self.pill = ctk.CTkLabel(head, text="Loading", corner_radius=10, height=24, width=96,
                                 font=ctk.CTkFont(FONT, 11, "bold"), fg_color=COLORS["off"],
                                 text_color=COLORS["muted"])
        self.pill.pack(side="right")
        _label(body, "Choose what you hear from your PC", 12, color=COLORS["muted"]
               ).pack(anchor="w", **pad)

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(14, 8), **pad)
        row.columnconfigure((0, 1), weight=1, uniform="m")
        self.btn_normal = ctk.CTkButton(row, text="Normal", height=64, corner_radius=14,
                                        font=ctk.CTkFont(FONT, 17, "bold"),
                                        command=lambda: self.set_mode("normal"))
        self.btn_filter = ctk.CTkButton(row, text="Voice only", height=64, corner_radius=14,
                                        font=ctk.CTkFont(FONT, 17, "bold"),
                                        text_color_disabled=COLORS["muted"],
                                        command=lambda: self.set_mode("filter"))
        self.btn_normal.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.btn_filter.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        stats = ctk.CTkFrame(body, fg_color="transparent")
        stats.pack(fill="x", **pad)
        stats.columnconfigure((0, 1, 2), weight=1, uniform="s")
        self.stat_labels = {}
        for i, (key, title) in enumerate((("delay", "Delay"), ("gpu", "GPU / step"),
                                          ("glitches", "Glitches"))):
            box = ctk.CTkFrame(stats, fg_color=COLORS["card"], corner_radius=10)
            box.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0 if i == 2 else 4))
            _label(box, title, 10, color=COLORS["muted"]).pack(pady=(6, 0))
            self.stat_labels[key] = _label(box, "—", 15, "bold")
            self.stat_labels[key].pack(pady=(0, 6))

        self.status = _label(body, "Loading model…", 12, color=COLORS["muted"],
                             anchor="w", justify="left", wraplength=420)
        self.status.pack(fill="x", pady=(8, 0), **pad)

        self._build_keep()
        self._build_model()
        self._build_latency()
        self._build_devices()

        foot = ctk.CTkFrame(body, fg_color="transparent")
        foot.pack(fill="x", pady=(10, 18), **pad)
        self.hint = _label(foot, "Ctrl+Alt+V switches Normal / Filtered from any app", 11,
                           color=COLORS["muted"])
        self.hint.pack(side="left")
        ctk.CTkButton(foot, text="System check", width=110, height=28, corner_radius=8,
                      fg_color=COLORS["off"], hover_color=COLORS["card_hi"],
                      font=ctk.CTkFont(FONT, 11), command=self._show_doctor).pack(side="right")
        self._refresh_all()

    def _card(self, title: str, toggle_text: str | None = None, toggle=None):
        card = ctk.CTkFrame(self.body, fg_color=COLORS["card"], corner_radius=14)
        card.pack(fill="x", padx=18, pady=(12, 0))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 6))
        _label(head, title, 14, "bold").pack(side="left")
        note = _label(head, "", 11, color=COLORS["muted"])
        note.pack(side="left", padx=10)
        switch = None
        if toggle_text:
            switch = ctk.CTkSwitch(head, text=toggle_text, command=toggle, width=40,
                                   font=ctk.CTkFont(FONT, 11), text_color=COLORS["muted"],
                                   progress_color=COLORS["filter"])
            switch.pack(side="right")
        return card, note, switch

    def _build_keep(self) -> None:
        card, self.keep_note, self.mixer_switch = self._card("Keep", "Mixer", self._toggle_mixer)
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure((0, 1, 2), weight=1, uniform="k")
        self.mix_buttons = {}
        for i, (key, mode) in enumerate(MODES.items()):
            b = _chip_button(grid, mode.label, lambda k=key: self.set_mix(k))
            b.grid(row=i // 3, column=i % 3, sticky="ew", padx=3, pady=3)
            self.mix_buttons[key] = b
        self.mix_grid = grid
        self.mixer_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.mixer_vars: dict[str, tk.DoubleVar] = {}
        self.mixer_visible = False

    def _build_model(self) -> None:
        card, self.model_note, _ = self._card("Model")
        self.model_cards: dict[str, dict] = {}
        for key, label, desc in MODEL_CHOICES:
            box = ctk.CTkFrame(card, fg_color=COLORS["card_hi"], corner_radius=12,
                               border_width=2, border_color=COLORS["card_hi"])
            box.pack(fill="x", padx=14, pady=(0, 8))
            top = ctk.CTkFrame(box, fg_color="transparent")
            top.pack(fill="x", padx=12, pady=(10, 0))
            title = _label(top, label, 13, "bold")
            title.pack(side="left")
            badge = _label(top, "", 11, color=COLORS["muted"])
            badge.pack(side="right")
            _label(box, desc, 11, color=COLORS["muted"], anchor="w").pack(fill="x", padx=12)
            bars = ctk.CTkFrame(box, fg_color="transparent")
            bars.pack(fill="x", padx=12, pady=(4, 10))
            bars.columnconfigure((1, 3), weight=1)
            widgets = {"box": box, "badge": badge}
            for col, name in ((0, "Quality"), (2, "Speed")):
                _label(bars, name, 10, color=COLORS["muted"]).grid(row=0, column=col, padx=(0, 6))
                bar = ctk.CTkProgressBar(bars, height=6, progress_color=COLORS["filter"],
                                         fg_color=COLORS["off"])
                bar.grid(row=0, column=col + 1, sticky="ew", padx=(0, 12))
                widgets[name.lower()] = bar
            widgets["facts"] = _label(box, "", 10, color=COLORS["muted"], anchor="w")
            widgets["facts"].pack(fill="x", padx=12, pady=(0, 10))
            for w in (box, top, title, bars):
                w.bind("<Button-1>", lambda _e, k=key: self.set_model(k))
            box.configure(cursor="hand2")
            self.model_cards[key] = widgets
        self.model_var = tk.StringVar(value=self.cfg["model"])

    def _build_latency(self) -> None:
        card, self.delay_label, self.adv_switch = self._card(
            "Latency", "Advanced", self._toggle_advanced)
        self.preset_row = ctk.CTkFrame(card, fg_color="transparent")
        self.preset_row.pack(fill="x", padx=14, pady=(0, 4))
        _label(card, "Lowest delay: Fast · HTDemucs + Instant (~0.08 s). Bluetooth "
                     "headphones add ~0.2–0.3 s on top; wired ones add none.",
               10, color=COLORS["muted"], anchor="w", justify="left", wraplength=400
               ).pack(fill="x", padx=14, pady=(0, 12))
        self.preset_buttons: dict[str, ctk.CTkButton] = {}
        self.adv_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.adv_vars, self.adv_values = {}, {}
        for i, (key, label, lo, hi, step) in enumerate(ADVANCED):
            _label(self.adv_frame, label, 11, color=COLORS["muted"]).grid(
                row=i, column=0, sticky="w", padx=(14, 8), pady=4)
            var = tk.DoubleVar(value=0.0)
            slider = self._slider(self.adv_frame, var, lo, hi, step,
                                  lambda _v, k=key: self._show_adv_value(k))
            slider.grid(row=i, column=1, sticky="ew", pady=4)
            slider.bind("<ButtonRelease-1>", lambda _e: self._advanced_changed())
            value = _label(self.adv_frame, "", 11)
            value.grid(row=i, column=2, padx=(8, 14))
            self.adv_vars[key], self.adv_values[key] = var, value
        self.adv_frame.columnconfigure(1, weight=1)
        self.adv_visible = False

    def _build_devices(self) -> None:
        card, _, _ = self._card("Audio")
        names = [s.name for s in winaudio.speakers()]
        self.listen_var = tk.StringVar(value=self._match(names, self.cfg["listen"]))
        self.capture_var = tk.StringVar(value=self._match(names, self.cfg["capture"]))
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure(1, weight=1)
        for row, (label, var) in enumerate((("Listen on", self.listen_var),
                                            ("Capture via (muted)", self.capture_var))):
            _label(grid, label, 11, color=COLORS["muted"]).grid(row=row, column=0, sticky="w",
                                                               padx=(0, 10), pady=4)
            menu = ctk.CTkOptionMenu(grid, variable=var, values=names, dynamic_resizing=False,
                                     fg_color=COLORS["off"], button_color=COLORS["off"],
                                     button_hover_color=COLORS["card_hi"],
                                     font=ctk.CTkFont(FONT, 11),
                                     command=lambda _v: self._devices_changed())
            menu.grid(row=row, column=1, sticky="ew", pady=4)
            self.combos.append(menu)

    @staticmethod
    def _slider(parent, var, lo, hi, step, command) -> ctk.CTkSlider:
        return ctk.CTkSlider(parent, variable=var, from_=lo, to=hi,
                             number_of_steps=max(1, round((hi - lo) / step)), command=command,
                             progress_color=COLORS["filter"], button_color=COLORS["text"],
                             button_hover_color=COLORS["filter"], fg_color=COLORS["off"])

    @staticmethod
    def _match(names: list[str], fragment: str) -> str:
        return next((n for n in names if fragment.lower() in n.lower()), names[0] if names else "")

    def _toggle_mixer(self) -> None:
        self.mixer_visible = not self.mixer_visible
        if self.mixer_visible:
            self.mixer_frame.pack(fill="x", padx=14, pady=(0, 12), after=self.mix_grid)
        else:
            self.mixer_frame.pack_forget()

    def _toggle_advanced(self) -> None:
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_frame.pack(fill="x", pady=(0, 12), after=self.preset_row)
        else:
            self.adv_frame.pack_forget()

    def _show_adv_value(self, key: str) -> None:
        v = self.adv_vars[key].get()
        self.adv_values[key].configure(text=f"{v:.1f} s" if key == "window_s" else f"{v * 1000:.0f} ms")

    def _show_doctor(self) -> None:
        win = ctk.CTkToplevel(self.root, fg_color=COLORS["bg"])
        win.title("VoiceOnly · System check")
        win.geometry("760x520")
        box = ctk.CTkTextbox(win, font=ctk.CTkFont("Consolas", 12), fg_color=COLORS["card"],
                             wrap="word")
        box.pack(fill="both", expand=True, padx=14, pady=14)
        box.insert("1.0", "Checking…")
        win.after(50, lambda: (box.delete("1.0", "end"), box.insert("1.0", report()),
                               box.configure(state="disabled")))
        win.transient(self.root)
        win.focus()

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
        if self.loading_key or not ready:
            pill, fg, color = "Loading", COLORS["off"], COLORS["muted"]
        elif self.mode == "filter":
            pill, fg, color = "Filtering", COLORS["filter_dark"], COLORS["filter"]
        else:
            pill, fg, color = "Ready", COLORS["off"], COLORS["text"]
        self.pill.configure(text=pill, fg_color=fg, text_color=color)
        if self.mode != "filter":
            for label in self.stat_labels.values():
                label.configure(text="—", text_color=COLORS["text"])

    def _refresh_keep(self) -> None:
        stems = self._stems()
        for key, b in self.mix_buttons.items():
            _paint(b, self.cfg["mix"] == key, COLORS["filter"], enabled=available(key, stems))
        unavailable = [MODES[k].label for k in MODES if not available(k, stems)]
        self.keep_note.configure(
            text=f"{' & '.join(unavailable)}: needs Cinematic" if unavailable else "")
        # One slider per stem of the current model.
        for child in self.mixer_frame.winfo_children():
            child.destroy()
        self.mixer_vars = {}
        self.mixer_frame.columnconfigure(1, weight=1)
        for i, (stem, gain) in enumerate(self._gains().items()):
            _label(self.mixer_frame, stem.capitalize(), 11, color=COLORS["muted"]).grid(
                row=i, column=0, sticky="w", padx=(0, 10), pady=4)
            var = tk.DoubleVar(value=round(gain * 100))
            value = _label(self.mixer_frame, f"{round(gain * 100)}%", 11)
            slider = self._slider(self.mixer_frame, var, 0, 100, 5,
                                  lambda v, lbl=value: lbl.configure(text=f"{round(v)}%"))
            slider.grid(row=i, column=1, sticky="ew", pady=4)
            slider.bind("<ButtonRelease-1>", lambda _e: self._mixer_changed())
            value.grid(row=i, column=2, padx=(8, 0))
            self.mixer_vars[stem] = var

    def _refresh_model(self) -> None:
        selected = self.loading_key or self.cfg["model"]
        self.model_var.set(selected)
        self.model_note.configure(text="loading…" if self.loading_key else "")
        for key, w in self.model_cards.items():
            active = key == selected
            w["box"].configure(border_color=COLORS["filter"] if active else COLORS["card_hi"])
            b = self.bench.get(key, {})
            live = b.get("live", {}).get("balanced") or next(iter(b.get("live", {}).values()), {})
            voice = live.get("voice") or b.get("offline", {}).get("voice")
            w["quality"].set(min(1.0, max(0.0, (voice or 0) / 25)))
            w["speed"].set(min(1.0, max(0.0, (live.get("headroom") or 0) / 6)))
            if live:
                # Measured real-time delay (engine + device) when available.
                rt = b.get("realtime", {}).get("balanced", {})
                delay = rt.get("delay_s", live["engine_delay_s"])
                w["facts"].configure(text=(
                    f"Voice {voice:.1f} dB  ·  {delay:.2f} s delay  ·  "
                    f"{live['step_ms']:.0f} ms/step  ·  {live['vram_mb'] / 1024:.1f} GB"
                    f"  ·  {b.get('params_m', 0):.1f} M params"))
            else:
                w["facts"].configure(text="Not benchmarked yet (scripts/model_benchmark.py)")
            w["badge"].configure(text="loading…" if key == self.loading_key else
                                 ("selected" if active else ""))

    def _refresh_latency(self) -> None:
        presets = presets_for(self.cfg["model"])
        if list(self.preset_buttons) != list(presets):
            for b in self.preset_buttons.values():
                b.destroy()
            self.preset_buttons = {}
            for i, (key, preset) in enumerate(presets.items()):
                b = _chip_button(self.preset_row, preset.label, lambda k=key: self.set_preset(k))
                b.grid(row=0, column=i, sticky="ew", padx=3)
                self.preset_row.columnconfigure(i, weight=1, uniform="p")
                self.preset_buttons[key] = b
        for key, b in self.preset_buttons.items():
            _paint(b, self.cfg["preset"] == key, COLORS["filter"])
        t = self._timing()
        delay = t["hop_s"] + t["lookahead_s"] + t.get("crossfade_s", 0.02) + t["jitter_s"]
        name = "Custom" if self.cfg["preset"] == "custom" else presets[self.cfg["preset"]].label
        self.delay_label.configure(text=f"{name} · ~{delay:.2f} s + device")
        for key, var in self.adv_vars.items():
            var.set(t[key])
            self._show_adv_value(key)

    def _timing(self) -> dict:
        """Current timing values: the custom set, or the selected preset's."""
        if self.cfg["preset"] == "custom" and self.cfg["custom"]:
            return dict(self.cfg["custom"])
        presets = presets_for(self.cfg["model"])
        p = presets.get(self.cfg["preset"]) or presets[DEFAULT_PRESET]
        return {"hop_s": p.hop_s, "lookahead_s": p.lookahead_s, "jitter_s": p.jitter_s,
                "window_s": p.window_s, "crossfade_s": p.crossfade_s}

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, text_color=COLORS["error"] if error else COLORS["muted"])
        if error:
            self.pill.configure(text="Error", fg_color="#3a1d22", text_color=COLORS["error"])

    def _show_stats(self, engine: LiveEngine) -> None:
        s = engine.stats
        ms = list(s.step_ms)
        gpu = sum(ms) / len(ms) if ms else 0.0
        hop_ms = engine.proc.hop / engine.proc.sr * 1000
        glitches = s.underruns + s.dropped_blocks + s.drift_drops
        self.stat_labels["delay"].configure(text=f"{engine.latency_s:.2f} s")
        self.stat_labels["gpu"].configure(
            text=f"{gpu:.0f} ms", text_color=COLORS["warn"] if gpu > 0.7 * hop_ms else COLORS["text"])
        self.stat_labels["glitches"].configure(
            text=str(glitches), text_color=COLORS["warn"] if glitches else COLORS["text"])

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
        # Keep the intent across back-to-back switches (the first one already stopped us).
        self.resume_filter = self.resume_filter or self.mode == "filter"
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
                # Usually ends within 50 ms; during CUDA-graph warm-up it can take a few
                # seconds. Don't freeze the window: track it and let it finish.
                self.engine_thread.join(timeout=0.5)
                if self.engine_thread.is_alive():
                    self.stopping.append(self.engine_thread)
            self.engine, self.engine_thread = None, None
        self.stopping = [t for t in self.stopping if t.is_alive()]

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

        stopping = list(self.stopping)

        def work() -> None:
            for t in stopping:
                t.join()  # an engine still using the old model must finish first
            if old is not None:
                with GPU_LOCK:
                    old.close()  # free GPU memory before loading the next model
            try:
                sep = separators.create(key)
                with GPU_LOCK:
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
            self._show_stats(self.engine)
            self._set_status(f"{self._mix_label()}: filtering "
                             f"{self.capture_var.get()} → {self.listen_var.get()}")
        self.root.after(100, self._poll)

    def quit(self) -> None:
        """Close in order: sound back to normal, threads stopped, GPU memory freed."""
        if getattr(self, "closing", False):
            return
        self.closing = True
        self.root.withdraw()  # the window disappears at once; cleanup follows
        try:
            self.set_mode("normal")  # stops the engine and restores Windows' output
        except Exception:
            log.exception("restoring normal mode on close failed")
        # Engines still warming up (CUDA-graph capture) get a moment to finish.
        deadline = time.monotonic() + 3.0
        for t in [*self.stopping, self.engine_thread]:
            if t is not None and t.is_alive():
                t.join(max(0.0, deadline - time.monotonic()))
        if getattr(self, "hotkey_thread_id", None):
            ctypes.windll.user32.PostThreadMessageW(self.hotkey_thread_id, 0x0012, 0, 0)
        if self.separator is not None:
            with GPU_LOCK:
                self.separator.close()  # weights, CUDA graphs, caches, cuBLAS workspaces
            self.separator = None
        log.info("closed cleanly")
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
    # The models run on the GPU; PyTorch would otherwise start one CPU worker per core
    # (70+ threads) that spin and compete with the audio threads. Set before import.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    if not _single_instance():
        _focus_existing_window()
        return
    App().run()
    # Everything was released in App.quit(); end the process even if a library left
    # a thread behind, so nothing keeps holding GPU memory after the window closed.
    logging.shutdown()
    os._exit(0)


_INSTANCE_MUTEX = None


def _single_instance() -> bool:
    """True if this is the only VoiceOnly app; a second copy would fight over the GPU,
    the hotkey and Windows' output routing."""
    global _INSTANCE_MUTEX
    _INSTANCE_MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\VoiceOnlyApp")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def _focus_existing_window() -> None:
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW("TkTopLevel", "VoiceOnly")
    if hwnd:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)


if __name__ == "__main__":
    main()
