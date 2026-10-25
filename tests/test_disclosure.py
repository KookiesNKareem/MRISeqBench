from mriseqbench.harness.feedback import disclosed


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
