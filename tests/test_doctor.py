import json

from voiceonly import doctor


def test_report_lists_system_and_every_app_model():
    text = doctor.report()
    assert "System" in text and "Models" in text
    for key in doctor.MODEL_NOTES:
        assert key in text


def test_model_checks_use_benchmark_numbers(tmp_path, monkeypatch):
    bench = {"meta": {"gpu": "Test GPU"}, "models": {"tiger": {
        "params_m": 4.22, "offline": {"voice": 15.9},
        "live": {"balanced": {"voice": 15.4, "step_ms": 68.0, "hop_ms": 150.0,
                              "headroom": 2.1, "vram_mb": 900, "engine_delay_s": 0.4}}}}}
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(bench))
    monkeypatch.setattr(doctor, "BENCHMARKS", path)
    rows = {r["key"]: r for r in doctor.model_checks("Test GPU")}
    assert rows["tiger"]["live"]["voice"] == 15.4 and rows["tiger"]["measured_here"]
    assert rows["htdemucs"]["live"] is None  # not benchmarked: no numbers, no crash


def test_missing_benchmark_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "BENCHMARKS", tmp_path / "nope.json")
    assert doctor.load_benchmarks() == {"meta": {}, "models": {}}
