import json
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import normalize_verdict, preflight
from mriseqbench.evaluation.sar import evaluate_sar

ROOT = Path(__file__).resolve().parents[1]
CASE = "gre_t1w-v2--static_b0-v1--hardware-high_performance-v1"


def test_normalize_verdict_fails_unresolved_adaptive_verdicts():
    case = {"case_id": "c", "evaluation": {"id": "s1"}}
    result = normalize_verdict({"status": "numerically_unresolved"}, case)
    assert result["status"] == "failed"
    assert result["failure_kind"] == "unresolved_at_sample_cap"
    smoke = {"case_id": "c", "evaluation": {"id": "pulseq_preflight"}}
    assert normalize_verdict({"status": "passed"}, smoke) == {"status": "passed"}


def test_cli_evaluate_reports_normalized_verdict(tmp_path, monkeypatch):
    from mriseqbench import cli

    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda *_a, **_k: {"case_id": CASE, "status": "numerically_unresolved"},
    )
    submission = tmp_path / "sequence.seq"
    submission.write_text("x")
    out = tmp_path / "out"
    argv = ["--root", str(ROOT), "evaluate", str(submission), "--case", CASE]
    assert cli.main([*argv, "--out", str(out)]) == 1
    assert json.loads((out / "result.json").read_text())["status"] == "failed"


def test_malformed_sequences_fail_but_infrastructure_errors_propagate(
    tmp_path, monkeypatch, submission
):
    import pypulseq

    case = Catalog(ROOT).find_case(CASE)
    garbage = tmp_path / "garbage.seq"
    garbage.write_bytes(b"\xff\x00 not a pulseq file")
    assert not all(check["passed"] for check in preflight(garbage, case))
    assert not all(c["passed"] for c in evaluate_sar(garbage, case["hardware"]))

    def unavailable(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(pypulseq.Sequence, "read", unavailable)
    with pytest.raises(OSError):
        preflight(submission, case)
    with pytest.raises(OSError):
        evaluate_sar(submission, case["hardware"])
