import argparse
import json

import pytest

from voiceonly import bench, cli
from voiceonly.live import PRESETS, StreamingProcessor
from voiceonly.separators import create
from voiceonly.synthetic import write_set


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_valid_for_the_live_engine(name):
    p = PRESETS[name]
    proc = StreamingProcessor(create("passthrough"), p.config())
    assert proc.window >= proc.hop + proc.look + proc.fade
    assert p.delay_s == pytest.approx(p.hop_s + p.lookahead_s + p.crossfade_s + p.jitter_s)


def test_presets_get_faster_in_order():
    delays = [p.delay_s for p in PRESETS.values()]
    assert delays == sorted(delays, reverse=True)


def _args(**kw):
    base = dict(model="htdemucs", preset="clean", hop=None, lookahead=None, window=None, crossfade=None,
                jitter=None)
    return argparse.Namespace(**{**base, **kw})


def test_cli_flags_override_the_preset():
    cfg = cli._live_config(_args(preset="fast", lookahead=0.3))
    assert cfg.hop_s == PRESETS["fast"].hop_s and cfg.lookahead_s == 0.3
    assert cli._jitter(_args(preset="ultra")) == PRESETS["ultra"].jitter_s
    assert cli._jitter(_args(preset="ultra", jitter=0.2)) == 0.2


def test_bench_keeps_going_when_a_model_fails(tmp_path):
    write_set(tmp_path / "data", items=1, seconds=2)
    result = bench.run(tmp_path / "data", ["no-such-model", "passthrough"], tmp_path / "run",
                       log=lambda *_: None)
    assert "no-such-model" in result["failed"]
    assert {r["model"] for r in result["rows"]} == {"passthrough"}
    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert "no-such-model" in saved["failed"]


def test_app_config_ignores_stale_timing_and_bad_presets(tmp_path, monkeypatch):
    from voiceonly import app

    monkeypatch.setattr(app, "CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps(
        {"listen": "X", "hop": 0.25, "preset": "custom", "custom": None}))
    cfg = app.load_config()
    assert cfg["listen"] == "X" and "hop" not in cfg
    assert cfg["preset"] == app.DEFAULT_PRESET  # "custom" without values falls back


def test_model_specific_presets_are_valid():
    from voiceonly.live import MODEL_PRESETS
    from voiceonly.separators.base import ModelInfo, SpeechSeparator

    class Fake(SpeechSeparator):
        info = ModelInfo(key="fake", name="f", version="1", sample_rate=44100, channels=2,
                         chunk_seconds=1, overlap_seconds=0, live_window_seconds=1.5)

        def separate(self, chunk):
            return chunk

    for presets in MODEL_PRESETS.values():
        for p in presets.values():
            proc = StreamingProcessor(Fake(), p.config())
            assert proc.window >= proc.hop + proc.look + proc.fade


def test_cli_rejects_preset_missing_for_model():
    with pytest.raises(RuntimeError):
        cli._preset(_args(model="bs-roformer-live", preset="ultra"))
