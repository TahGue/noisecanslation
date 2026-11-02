import json

import numpy as np
import pytest

from voiceonly import audio_io, bench, metrics, quality
from voiceonly.chunking import chunk_starts, process_chunked
from voiceonly.pipeline import extract, run_file
from voiceonly.separators import create
from voiceonly.synthetic import write_set


def run_ola(x, process, chunk, overlap):
    blocks = list(process_chunked(lambda s, n: x[s:s + n], x.shape[0], process, chunk, overlap))
    return np.concatenate(blocks), blocks


@pytest.mark.parametrize("total", [1, 99, 100, 101, 1000, 1037, 4321])
@pytest.mark.parametrize("chunk,overlap", [(100, 0), (100, 10), (100, 50), (256, 255)])
def test_identity_reconstruction_is_exact(total, chunk, overlap):
    x = np.random.default_rng(total).standard_normal((total, 2)).astype(np.float32)
    y, _ = run_ola(x, lambda c: c, chunk, overlap)
    assert y.shape == x.shape  # no missing or duplicated frames
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_constant_gain_survives_crossfades():
    x = np.ones((5000, 1), np.float32)
    y, _ = run_ola(x, lambda c: 0.5 * c, 512, 128)
    np.testing.assert_allclose(y, 0.5, atol=1e-6)


def test_output_is_streamed_with_bounded_buffers():
    x = np.zeros((100_000, 1), np.float32)
    _, blocks = run_ola(x, lambda c: c, 1000, 100)
    assert len(blocks) == len(chunk_starts(100_000, 1000, 100))
    assert max(b.shape[0] for b in blocks) <= 1000


def test_processor_must_keep_length():
    x = np.zeros((500, 1), np.float32)
    with pytest.raises(RuntimeError):
        run_ola(x, lambda c: c[:-1], 100, 10)


def test_resample_round_trip_keeps_duration():
    x = np.random.default_rng(0).standard_normal((44_101, 2)).astype(np.float32)
    up = audio_io.resample(x, 44_100, 48_000)
    assert up.shape[0] == 48_001
    assert audio_io.resample(up, 48_000, 44_100).shape == x.shape


def test_si_sdr_and_decomposition():
    rng = np.random.default_rng(1)
    s, b = rng.standard_normal(48_000), rng.standard_normal(48_000)
    assert metrics.si_sdr(s, s) > 90
    d = metrics.decompose(s + 0.1 * b, s, b)
    assert abs(d["speech_gain_db"]) < 0.01
    assert d["leakage_db"] == pytest.approx(-20, abs=0.1)
    d = metrics.decompose(0.5 * s, s, b)
    assert d["speech_gain_db"] == pytest.approx(-6.02, abs=0.01)
    assert d["leakage_db"] < -60


def test_quality_flags_problems():
    sr = 1000
    ref = np.full((5 * sr, 1), 0.1, np.float32)
    assert quality.check(ref.copy(), ref.shape[0], sr, ref)["ok"]
    out = ref.copy()
    out[1000:4000] = 0
    issues = quality.check(out, ref.shape[0], sr, ref)["issues"]
    assert any("silence" in i for i in issues)
    assert not quality.check(ref[:-1], ref.shape[0], sr, ref)["ok"]


def test_passthrough_preserves_rate_channels_and_timeline(tmp_path):
    # Band-limited content: resampling legitimately removes energy near Nyquist.
    t = np.arange(44_100 * 3 + 7) / 44_100
    x = sum(np.sin(2 * np.pi * f * t) for f in (110, 1234, 5000, 15000)) / 5
    x = x[:, None].astype(np.float32)
    src = tmp_path / "in.wav"
    audio_io.save(src, x, 44_100)
    run_file(src, tmp_path / "out", create("passthrough"))

    out, info = audio_io.load(tmp_path / "out" / "speech.wav")
    assert info.sample_rate == 44_100 and out.shape == x.shape
    # Passthrough resamples 44.1k -> 48k -> 44.1k, so only band-edge error remains.
    assert metrics.si_sdr(out, x) > 25
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["quality"]["ok"] and meta["model"]["key"] == "passthrough"
    assert (tmp_path / "out" / "speech.flac").exists()
    assert "rtf" in json.loads((tmp_path / "out" / "benchmark.json").read_text())


def test_passthrough_at_native_rate_is_bit_exact():
    x = np.random.default_rng(3).uniform(-1, 1, (48_000 * 25, 2)).astype(np.float32)
    y, _ = extract(x, 48_000, create("passthrough"))
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_bench_on_synthetic_set(tmp_path):
    write_set(tmp_path / "data", items=2, seconds=3)
    result = bench.run(tmp_path / "data", ["passthrough"], tmp_path / "run", log=lambda *_: None)
    row = result["rows"][0]
    # Passthrough output is the mix, so it gains nothing and leaks the whole background.
    assert row["si_sdr_i"] == pytest.approx(0, abs=0.01)
    assert row["speech_gain_db"] == pytest.approx(0, abs=0.1)
    assert row["leakage_db"] == pytest.approx(row["si_sdr_mix"] * -1, abs=1.0)
    key = json.loads((tmp_path / "run" / "key.json").read_text())
    assert set(key) == {"item_000", "item_001"}
    assert (tmp_path / "run" / "listening" / "item_000" / "A.wav").exists()
