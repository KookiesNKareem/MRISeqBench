from mriseqbench.harness.feedback import disclosed


def test_private_artifacts_stay_in_trusted_verdict_not_agent_feedback():
    from copy import deepcopy

    artifacts = {"images": "/private/control/submissions/physical/images.npz"}
    result = {
        "status": "failed",
        "metrics": {"case_magnitude_nrmse": 0.12},
        "artifacts": artifacts,
        "frames": [{"status": "failed", "artifacts": artifacts}],
        "diagnostic": {"frame_artifacts": [artifacts]},
        "physio": {
            "original_path": "/private/ecg.json",
            "resolved_path": "/private/ecg.json",
            "valid": True,
        },
        "encoded_replay_cache": {"directory": "/private/cache"},
    }
    original = deepcopy(result)
    assert disclosed(result) == {
        "status": "failed",
        "metrics": {"case_magnitude_nrmse": 0.12},
        "frames": [{"status": "failed"}],
        "diagnostic": {},
        "physio": {"valid": True},
    }
    assert result == original


def test_submission_feedback_reports_sar_without_the_model():
    sar = {"peak_W_per_kg": 2.4, "limit_W_per_kg": 2.0, "coefficient": 0.7}
    result = disclosed(
        {
            "status": "failed",
            "checks": [
                {"name": "timing", "passed": True, "detail": "ok"},
                {"name": "sar_360s", "passed": False, "detail": sar},
                {"name": "sar_10s", "passed": True, "detail": sar},
            ],
        }
    )
    assert result["checks"] == [
        {"name": "timing", "passed": True, "detail": "ok"},
        {
            "name": "sar",
            "passed": False,
            "detail": "scanner SAR limit exceeded; the scan was refused",
        },
    ]
