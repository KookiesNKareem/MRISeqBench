from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pypulseq import Opts, Sequence, make_adc, make_block_pulse, make_delay

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate, preflight
from mriseqbench.evaluation.sar import peak_window, rf_intervals, sar_checks
from mriseqbench.models import Hardware

ROOT = Path(__file__).resolve().parents[2]


def case_for(hardware="standard"):
    catalog = Catalog(ROOT)
    return catalog.resolve(
        catalog.get("tasks", "pulseq_smoke"),
        catalog.get("physics", "ideal"),
        hardware_ref=hardware,
    )


def high_duty_submission(tmp_path):
    system = Opts(rf_dead_time=100e-6, rf_ringdown_time=30e-6, adc_dead_time=10e-6)
    seq = Sequence(system)
    # 2 uT is well below peak B1, but high duty cycle exceeds global SAR at 3 T.
    rf = make_block_pulse(
        2 * np.pi * system.gamma * 2e-6 * 0.1,
        duration=0.1,
        delay=100e-6,
        system=system,
        use="saturation",
    )
    for _ in range(5):
        seq.add_block(rf)
    seq.add_block(make_adc(1, dwell=10e-6, delay=10e-6, system=system))
    path = tmp_path / "high-duty.seq"
    seq.write(str(path))
    return path, seq


def test_field_strength_and_analytic_constant_rf(tmp_path):
    path, seq = high_duty_submission(tmp_path)
    values = {}
    for hardware, passes in [
        ("standard", False),
        ("standard_1_5t", True),
        ("low_field", True),
    ]:
        case = case_for(hardware)
        result = evaluate(path, case, ROOT, tmp_path / hardware)
        check = next(c for c in result["checks"] if c["name"] == "sar_360s")
        values[hardware] = check["detail"]["peak_W_per_kg"]
        assert check["passed"] is passes
        assert result["status"] == ("smoke_passed" if passes else "failed")
        assert next(c for c in result["checks"] if c["name"] == "peak_b1")["passed"]
    assert values["standard"] / values["standard_1_5t"] == pytest.approx(4)
    assert values["low_field"] / values["standard"] == pytest.approx((0.55 / 3) ** 2)
    load = case_for()["hardware"]["sar"]
    coefficient = (
        load["conductivity_S_per_m"]
        * (2 * np.pi * seq.system.gamma * 3) ** 2
        * load["radius_m"] ** 2
        / (5 * load["density_kg_per_m3"])
    )
    expected = coefficient * (2e-6) ** 2 * 0.5 / seq.duration()[0]
    assert values["standard"] == pytest.approx(expected, rel=1e-6)


def test_sar_never_runs_or_leaks_in_feedback(tmp_path, monkeypatch):
    path, _ = high_duty_submission(tmp_path)
    case = case_for()
    assert all(c["passed"] for c in preflight(path, case))

    def forbidden(*args):
        raise AssertionError("Feedback must not compute SAR")

    import importlib

    monkeypatch.setattr(
        importlib.import_module("mriseqbench.evaluation.evaluate"),
        "evaluate_sar",
        forbidden,
    )
    result = evaluate(path, case, ROOT, tmp_path / "feedback", feedback=True)
    assert result["status"] == "smoke_passed"
    assert all(not c["name"].startswith("sar") for c in result["checks"])


def test_real_feedback_worker_excludes_sar_and_submit_shows_no_verdict(tmp_path):
    from mriseqbench.harness.feedback import FeedbackServer
    from mriseqbench.models import FeedbackConfig

    path, _ = high_duty_submission(tmp_path)
    config = FeedbackConfig(mode="evaluation", max_checks=1, timeout_s=30)
    with FeedbackServer(
        case_for(), config, ROOT, tmp_path / "control", max_submissions=1
    ) as server:
        for kind in ("lint", "check"):
            code, result = server.handle(kind, server.token, path.read_bytes())
            assert code == 200
            assert result["status"] in ("lint_passed", "preflight_passed")
            assert all(not c["name"].startswith("sar") for c in result["checks"])
        code, receipt = server.handle("submit", server.token, path.read_bytes())
        assert code == 200
        assert receipt["status"] == "submitted"
        assert "checks" not in receipt
        assert "metrics" not in receipt
    # The identical accepted bytes then fail final evaluation.
    result = evaluate(server.submission_path, case_for(), ROOT, tmp_path / "judge")
    assert result["status"] == "failed"
    assert any(c["name"] == "sar_360s" and not c["passed"] for c in result["checks"])


def test_sar_failure_precedes_missing_physical_backend(tmp_path):
    path, _ = high_duty_submission(tmp_path)
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), catalog.get("physics", "ideal")
    )
    result = evaluate(path, case, ROOT, tmp_path / "judge")
    assert result["status"] == "failed"
    assert any(c["name"] == "sar_360s" and not c["passed"] for c in result["checks"])


def test_sar_failure_feedback_only_after_blind_submission(submission, tmp_path):
    from mriseqbench.harness.feedback import FeedbackServer
    from mriseqbench.models import FeedbackConfig

    path, _ = high_duty_submission(tmp_path)
    with FeedbackServer(
        case_for(), FeedbackConfig(timeout_s=30), ROOT, tmp_path / "control"
    ) as server:
        code, basic = server.handle("check", server.token, path.read_bytes())
        assert code == 200
        assert basic["status"] == "preflight_passed"
        assert all(not c["name"].startswith("sar") for c in basic["checks"])
        assert basic["metrics"] == {}
        assert server.attempts == []
        result = server.handle("submit", server.token, path.read_bytes())[1]
        assert result["status"] == "retry_required"
        assert any(
            c["name"] == "sar_360s" and not c["passed"]
            for c in result["evaluation"]["checks"]
        )
        assert server.attempts[0]["blind"]
        assert not server.submitted.is_set()
        assert (
            server.handle("submit", server.token, submission.read_bytes())[1]["status"]
            == "submitted"
        )
        assert [a["evaluation"]["status"] for a in server.attempts] == [
            "failed",
            "smoke_passed",
        ]


def test_burst_fails_ten_seconds_despite_low_scan_average():
    seq = Sequence()
    rf = make_block_pulse(
        2 * np.pi * seq.system.gamma * 2e-6, duration=1, use="saturation"
    )
    seq.add_block(make_delay(11.37))
    for _ in range(10):
        seq.add_block(rf)
    seq.add_block(make_delay(400))
    checks = sar_checks(seq, case_for()["hardware"])
    assert checks[0]["passed"]
    assert not checks[1]["passed"]
    assert checks[1]["detail"]["window_start_s"] == pytest.approx(11.37)
    assert checks[1]["detail"]["b1_rms_uT"] == pytest.approx(2)


def test_exact_windows_and_short_acquisitions():
    # The maximum crosses multiple event boundaries and begins off a 1 s grid.
    edges = np.array([0, 0.37, 3.37, 7.37, 10.37, 20])
    power = np.array([0, 2, 3, 2, 0])
    assert peak_window(edges, power, 10) == pytest.approx((2.4, 0.37, 10))
    assert peak_window(np.array([0, 1, 2]), np.array([4, 0]), 360) == (2, 0, 2)
    # A long-window violation can occur even when the 10 s limit passes.
    seq = Sequence()
    seq.add_block(
        make_block_pulse(
            2 * np.pi * seq.system.gamma * 1.4e-6 * 2, duration=2, use="saturation"
        )
    )
    checks = sar_checks(seq, case_for()["hardware"])
    assert not checks[0]["passed"]
    assert checks[1]["passed"]


def test_shaped_rf_energy_phase_and_delay():
    gamma = 42.576e6
    rf = SimpleNamespace(
        signal=np.array([1, 2j, -2, -1j]) * gamma * 1e-6,
        t=np.array([0.5, 1.5, 2.5, 3.5]) * 1e-6,
        shape_dur=4e-6,
    )
    edges, power = rf_intervals(rf, 1e-6, gamma)
    assert np.sum(np.diff(edges) * power) == pytest.approx(10e-6)
    rf.signal *= np.exp(0.43j)
    shifted_edges, shifted_power = rf_intervals(rf, 1e-6, gamma)
    assert np.sum(np.diff(shifted_edges) * shifted_power) == pytest.approx(10e-6)
    seq = Sequence()
    seq.add_block(
        make_block_pulse(
            2 * np.pi * gamma * 1e-6, duration=1, delay=1, use="saturation"
        )
    )
    check = sar_checks(seq, case_for()["hardware"])[0]
    assert check["detail"]["b1_rms_uT"] == pytest.approx(1 / np.sqrt(2))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rf_fails_closed(value):
    rf = SimpleNamespace(signal=np.array([value]), t=np.array([0.5e-6]), shape_dur=1e-6)
    with pytest.raises(ValueError):
        rf_intervals(rf, 1e-6, 42.576e6)


def test_unsupported_explicit_rf_time_shape_fails_closed():
    rf = SimpleNamespace(signal=np.array([1, 2]), t=np.array([0, 2e-6]), shape_dur=2e-6)
    with pytest.raises(ValueError, match="regular-raster"):
        rf_intervals(rf, 1e-6, 42.576e6)


def test_missing_sar_contract_fails_catalog_and_evaluation(submission, tmp_path):
    case = case_for()
    case["hardware"].pop("sar")
    with pytest.raises(ValueError):
        Hardware.model_validate(case["hardware"])
    result = evaluate(submission, case, ROOT, tmp_path)
    assert result["status"] == "failed"
    assert any(c["name"] == "sar_model" and not c["passed"] for c in result["checks"])
