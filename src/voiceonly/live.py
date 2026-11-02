"""Live speech isolation.

Each step the model sees a sliding window of the most recent audio:

    |<------------- context ------------->|<- hop ->|<- lookahead ->|
                                     emitted region ^ (hop frames)

The emitted region lies `lookahead` frames before the newest sample, so the
model always hears some future audio for every sample it emits. Consecutive
emitted regions are joined with a short crossfade. Total delay is
hop + lookahead + crossfade (+ device buffers).
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .separators import SeparationError, SpeechSeparator


@dataclass
class LiveConfig:
    hop_s: float = 0.1
    lookahead_s: float = 0.2
    window_s: float | None = None  # None: the model's preferred live window
    crossfade_s: float = 0.05

    def frames(self, sr: int, model_window_s: float) -> tuple[int, int, int, int]:
        hop, look, fade = (int(round(v * sr)) for v in (self.hop_s, self.lookahead_s,
                                                        self.crossfade_s))
        window = int(round((self.window_s or model_window_s) * sr))
        if hop <= 0 or look < 0 or fade < 0:
            raise ValueError("hop must be > 0; lookahead and crossfade >= 0")
        if window < hop + look + fade:
            raise ValueError(f"window ({window / sr:.2f}s) must cover hop + lookahead + crossfade")
        return window, hop, look, fade

    def latency_s(self) -> float:
        return self.hop_s + self.lookahead_s + self.crossfade_s


@dataclass(frozen=True)
class Preset:
    """A named latency/quality trade-off. Delay ~= hop + lookahead + crossfade + jitter."""

    label: str
    hop_s: float
    lookahead_s: float
    crossfade_s: float
    window_s: float
    jitter_s: float

    def config(self) -> LiveConfig:
        return LiveConfig(hop_s=self.hop_s, lookahead_s=self.lookahead_s,
                          window_s=self.window_s, crossfade_s=self.crossfade_s)

    @property
    def delay_s(self) -> float:
        return self.hop_s + self.lookahead_s + self.crossfade_s + self.jitter_s


# Window lengths and lookaheads were chosen from live-sim sweeps on an RTX 4090
# (docs/decisions/0005-latency-presets.md has the measurements).
PRESETS: dict[str, Preset] = {
    "clean": Preset("Clean", hop_s=0.10, lookahead_s=0.20, crossfade_s=0.05, window_s=4.0,
                    jitter_s=0.10),
    "balanced": Preset("Balanced", hop_s=0.05, lookahead_s=0.10, crossfade_s=0.02,
                       window_s=4.0, jitter_s=0.08),
    "fast": Preset("Fast", hop_s=0.05, lookahead_s=0.05, crossfade_s=0.02, window_s=3.0,
                   jitter_s=0.06),
    "ultra": Preset("Ultra", hop_s=0.04, lookahead_s=0.02, crossfade_s=0.01, window_s=2.0,
                    jitter_s=0.05),
}
DEFAULT_PRESET = "clean"


class StreamingProcessor:
    """Feed `hop` frames in, get `hop` filtered frames out, delayed by lookahead + crossfade."""

    def __init__(self, separator: SpeechSeparator, config: LiveConfig) -> None:
        info = separator.info
        if info.live_window_seconds is None and config.window_s is None:
            raise SeparationError(f"model '{info.key}' does not support live mode")
        self.separator = separator
        self.sr, self.channels = info.sample_rate, info.channels
        self.window, self.hop, self.look, self.fade = config.frames(
            self.sr, info.live_window_seconds)
        self.buf = np.zeros((self.window, self.channels), np.float32)
        self.tail = np.zeros((self.fade, self.channels), np.float32)
        ramp = (np.arange(self.fade) + 0.5) / max(self.fade, 1)
        self.ramp = ramp[:, None].astype(np.float32)
        self.bypass = False

    def step(self, block: np.ndarray) -> np.ndarray:
        if block.shape != (self.hop, self.channels):
            raise ValueError(f"expected block {(self.hop, self.channels)}, got {block.shape}")
        self.buf = np.concatenate([self.buf[self.hop:], block])
        est = self.buf if self.bypass else np.asarray(self.separator.separate(self.buf),
                                                      dtype=np.float32)
        end = self.window - self.look
        seg = est[end - self.hop - self.fade:end].copy()
        if self.fade:
            seg[:self.fade] = self.tail * (1 - self.ramp) + seg[:self.fade] * self.ramp
            self.tail = seg[self.hop:]
        return seg[:self.hop]

    @property
    def delay_frames(self) -> int:
        """Frames between an input sample entering and its filtered version leaving."""
        return self.look + self.fade


def simulate(audio: np.ndarray, separator: SpeechSeparator, config: LiveConfig) -> tuple[np.ndarray, dict]:
    """Run the live engine over a file as if it arrived in real time (without waiting).

    Returns output aligned to the input (delay removed) and per-step compute stats.
    """
    proc = StreamingProcessor(separator, config)
    hop, delay = proc.hop, proc.delay_frames
    n = audio.shape[0]
    padded = np.concatenate([audio, np.zeros((delay + hop, audio.shape[1]), np.float32)])
    out, times = [], []
    for start in range(0, n + delay, hop):
        block = padded[start:start + hop]
        if block.shape[0] < hop:
            block = np.pad(block, ((0, hop - block.shape[0]), (0, 0)))
        t0 = time.perf_counter()
        out.append(proc.step(block))
        times.append(time.perf_counter() - t0)
    y = np.concatenate(out)[delay:delay + n]
    t = np.array(times[1:] or times)  # first step includes CUDA warm-up
    hop_s = hop / proc.sr
    return y, {
        "hop_s": hop_s,
        "algorithmic_latency_s": (hop + delay) / proc.sr,
        "step_ms_mean": float(t.mean() * 1000),
        "step_ms_p99": float(np.percentile(t, 99) * 1000),
        "realtime_headroom": float(hop_s / np.percentile(t, 99)),
    }


@dataclass
class Stats:
    steps: int = 0
    underruns: int = 0
    dropped_blocks: int = 0
    drift_drops: int = 0
    step_ms: collections.deque = field(default_factory=lambda: collections.deque(maxlen=50))


class LiveEngine:
    """Device input -> worker thread -> device output, with bounded queues."""

    def __init__(self, separator: SpeechSeparator, config: LiveConfig,
                 input_device=None, output_device=None, jitter_s: float = 0.1,
                 device_latency: str | float = "low") -> None:
        self.proc = StreamingProcessor(separator, config)
        self.config = config
        self.input_device, self.output_device = input_device, output_device
        self.jitter = int(round(jitter_s * self.proc.sr))
        # PortAudio buffer request ("low"/"high"/seconds); the granted value is read back.
        self.device_latency = device_latency
        self.output_latency_s = 0.0
        self.stats = Stats()
        # Up to 4 hops of unprocessed input; beyond that the GPU is not keeping up.
        self.inq: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.pending = np.zeros((0, self.proc.channels), np.float32)
        self.max_pending = 4 * self.proc.hop
        self.out = collections.deque()
        self.out_frames = 0
        self.out_lock = threading.Lock()
        self.started_output = False
        self.running = False
        self.error: Exception | None = None
        self.recorded: list[np.ndarray] | None = None  # set to [] to keep a copy of output

    @property
    def latency_s(self) -> float:
        """Expected delay from capture to the speaker, excluding Bluetooth/codec delay."""
        engine = (self.proc.hop + self.proc.delay_frames + self.jitter) / self.proc.sr
        return engine + self.output_latency_s

    def _on_input(self, indata, frames, time_info, status) -> None:
        try:
            self.inq.put_nowait(indata.copy())
        except queue.Full:
            self.stats.dropped_blocks += 1

    def _on_output(self, outdata, frames, time_info, status) -> None:
        # Output arrives in hop-sized bursts. Playback starts with `jitter` frames of
        # silence in front of the first burst, so the buffer bottoms out at `jitter`,
        # not zero, just before each new burst lands. After an underrun nothing extra
        # is inserted; the rate controller in _worker brings the delay back.
        with self.out_lock:
            if not self.started_output:
                if self.out_frames == 0:
                    outdata.fill(0)
                    return
                self.out.appendleft(np.zeros((self.jitter, self.proc.channels), np.float32))
                self.out_frames += self.jitter
                self.started_output = True
            filled = 0
            while filled < frames and self.out:
                block = self.out[0]
                take = min(frames - filled, block.shape[0])
                outdata[filled:filled + take] = block[:take]
                if take == block.shape[0]:
                    self.out.popleft()
                else:
                    self.out[0] = block[take:]
                filled += take
            self.out_frames -= filled
            if filled < frames:
                outdata[filled:].fill(0)
                self.stats.underruns += 1

    def _worker(self) -> None:
        hop = self.proc.hop
        while self.running:
            try:
                chunk = self.inq.get(timeout=0.1)
            except queue.Empty:
                continue
            self.pending = np.concatenate([self.pending, chunk])
            if self.pending.shape[0] > self.max_pending + hop:
                # Overloaded: skip oldest audio rather than let delay grow forever.
                excess = self.pending.shape[0] - self.max_pending
                self.pending = self.pending[excess:]
                self.stats.dropped_blocks += 1
            while self.pending.shape[0] >= hop:
                block, self.pending = self.pending[:hop], self.pending[hop:]
                t0 = time.perf_counter()
                y = self.proc.step(block)
                self.stats.step_ms.append((time.perf_counter() - t0) * 1000)
                self.stats.steps += 1
                y = np.clip(y, -1.0, 1.0)
                if self.recorded is not None:
                    self.recorded.append(y)
                with self.out_lock:
                    trough = self.out_frames if self.started_output else self.jitter
                y = self._rate_adjust(y, trough)
                with self.out_lock:
                    self.out.append(y)
                    self.out_frames += y.shape[0]
                    # Last resort if the delay ran far away (e.g. a long system stall).
                    excess = self.out_frames - (self.jitter + 3 * hop)
                    if excess > 0:
                        self._drop_oldest(excess)
                        self.stats.drift_drops += 1

    def _rate_adjust(self, y: np.ndarray, trough: int) -> np.ndarray:
        """Stretch or squeeze a block by at most 1% to hold the buffer at `jitter`.

        `trough` is the buffer level just before this block lands; ideally `jitter`.
        Stalls and clock differences move it; a 1% speed change is inaudible and
        undoes 100 ms of extra delay in about 10 s without cutting audio.
        """
        self.trough_avg = 0.8 * getattr(self, "trough_avg", float(trough)) + 0.2 * trough
        n = y.shape[0]
        limit = n // 100
        delta = int(np.clip(round((self.trough_avg - self.jitter) * 0.05), -limit, limit))
        if delta == 0:
            return y
        m = n - delta
        pos = np.linspace(0, n - 1, m)
        return np.stack([np.interp(pos, np.arange(n), y[:, c]) for c in range(y.shape[1])],
                        axis=1).astype(np.float32)

    def _drop_oldest(self, n: int) -> None:
        """Discard exactly n queued output frames (caller holds out_lock)."""
        while n > 0 and self.out:
            block = self.out[0]
            if block.shape[0] <= n:
                self.out.popleft()
                n -= block.shape[0]
                self.out_frames -= block.shape[0]
            else:
                self.out[0] = block[n:]
                self.out_frames -= n
                n = 0

    def warm_up(self) -> None:
        """Run the model once so CUDA kernel setup does not stall the first live steps."""
        bypass, self.proc.bypass = self.proc.bypass, False
        self.proc.separator.separate(np.zeros_like(self.proc.buf))
        self.proc.bypass = bypass

    def _feed_file(self, audio: np.ndarray, block: int) -> None:
        """Stand-in for a capture device: deliver `audio` in real time."""
        start = time.perf_counter()
        sent = 0
        while self.running and sent < audio.shape[0]:
            chunk = audio[sent:sent + block]
            if chunk.shape[0] < block:
                chunk = np.pad(chunk, ((0, block - chunk.shape[0]), (0, 0)))
            self._on_input(chunk, block, None, None)
            sent += block
            wait = start + sent / self.proc.sr - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        # Flush the lookahead so the end of the file is heard, then stop.
        tail = self.proc.delay_frames + self.proc.hop + self.jitter
        self._on_input(np.zeros((tail, audio.shape[1]), np.float32), tail, None, None)
        time.sleep(tail / self.proc.sr + 0.3)
        self.running = False

    def run(self, on_status=None, input_audio: np.ndarray | None = None,
            duration_s: float | None = None) -> None:
        import contextlib

        import sounddevice as sd

        sr, ch = self.proc.sr, self.proc.channels
        self.warm_up()
        extra_out = sd.WasapiSettings(auto_convert=True) if _is_wasapi(self.output_device) else None
        self.running = True
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        try:
            with contextlib.ExitStack() as stack:
                out_stream = stack.enter_context(sd.OutputStream(
                    device=self.output_device, samplerate=sr, channels=ch, dtype="float32",
                    callback=self._on_output, extra_settings=extra_out,
                    latency=self.device_latency))
                self.output_latency_s = float(out_stream.latency)
                if isinstance(self.input_device, LoopbackCapture):
                    threading.Thread(target=self.input_device.run, args=(self,),
                                     daemon=True).start()
                elif input_audio is None:
                    extra = (sd.WasapiSettings(auto_convert=True)
                             if _is_wasapi(self.input_device) else None)
                    stack.enter_context(sd.InputStream(
                        device=self.input_device, samplerate=sr, channels=ch, dtype="float32",
                        callback=self._on_input, extra_settings=extra,
                        latency=self.device_latency))
                else:
                    threading.Thread(target=self._feed_file, args=(input_audio, sr // 100),
                                     daemon=True).start()
                end = time.perf_counter() + duration_s if duration_s else None
                last_status = 0.0
                while self.running and (end is None or time.perf_counter() < end):
                    time.sleep(0.05)  # short, so stopping is near-instant
                    if on_status and time.perf_counter() - last_status >= 0.5:
                        last_status = time.perf_counter()
                        on_status(self)
        finally:
            self.running = False
            worker.join(timeout=2)
        if self.error:
            raise RuntimeError(f"capture failed: {self.error}") from self.error


class LoopbackCapture:
    """Records what Windows plays to an output device (WASAPI loopback), no driver needed.

    Point Windows' default output at a device nobody listens to (muted monitor,
    unconnected digital out) and capture it here.
    """

    def __init__(self, name: str) -> None:
        import soundcard as sc

        self.mic = sc.get_microphone(name, include_loopback=True)
        if not self.mic.isloopback:
            raise RuntimeError(f"'{self.mic.name}' is not an output device")
        self.name = self.mic.name

    def run(self, engine: "LiveEngine") -> None:
        import ctypes

        # soundcard talks to WASAPI through COM, which must be set up per thread.
        ctypes.windll.ole32.CoInitializeEx(None, 0)
        sr, ch = engine.proc.sr, engine.proc.channels
        block = sr // 100
        try:
            with self.mic.recorder(samplerate=sr, channels=2, blocksize=block) as rec:
                while engine.running:
                    data = np.asarray(rec.record(numframes=block), dtype=np.float32)
                    if data.ndim == 1:
                        data = data[:, None]
                    if data.shape[1] != ch:
                        data = data.mean(axis=1, keepdims=True) if ch == 1 else data[:, :ch]
                    engine._on_input(data, data.shape[0], None, None)
        except Exception as e:  # surface capture failures instead of hanging silently
            engine.error = e
            engine.running = False


def _is_wasapi(device) -> bool:
    import sounddevice as sd

    try:
        info = sd.query_devices(device, kind=None) if device is not None else None
        if info is None:
            return False
        return "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]
    except Exception:
        return False


def find_device(name: str | None, kind: str):
    """Resolve a device by index or case-insensitive name fragment, preferring WASAPI."""
    import sounddevice as sd

    if name is None:
        return None
    if kind == "input" and name.lower().startswith("loopback:"):
        return LoopbackCapture(name.split(":", 1)[1])
    if name.isdigit():
        return int(name)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [(i, d) for i, d in enumerate(sd.query_devices())
               if name.lower() in d["name"].lower() and d[key] > 0]
    if not matches:
        raise RuntimeError(f"no {kind} device matching '{name}'. Run: voiceonly devices")
    for i, d in matches:
        if "WASAPI" in sd.query_hostapis(d["hostapi"])["name"]:
            return i
    return matches[0][0]
