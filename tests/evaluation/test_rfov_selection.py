"""RF-selection decisions must not depend on common post-RF water decay."""

from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s3


def assay(monkeypatch, times, mutation=None, *, fam="rfov_epi", k=None, late_rf=False):
    rf_end = 0.085
    acq = SimpleNamespace(
        adc=rf_end + np.asarray(times),
        adc_phase=np.zeros(3),
        k=np.zeros((3, 3)) if k is None else np.asarray(k, float),
        pulses=[SimpleNamespace(end=rf_end, refocusing=False)]
        # RF after the final ADC must not move the T2 reference time.
        + ([SimpleNamespace(end=rf_end + 0.5, refocusing=False)] if late_rf else []),
        excitation=[0, 0, 0],
        measurements={"reduced_fov_fraction": 0.5, "centre_sample_indices": [0, 1, 2]},
    )
    case = {
        "task": {"id": fam, "sequence": {"fov_mm": [200, 200], "matrix": [128, 128]}},
        "physics": {"effects": []},
    }

    def outputs(a, spins, _effects, **kwargs):
        assert kwargs == {"individual": True}
        x, y = spins.positions[:, :2].T
        selected = (abs(x) <= 0.08) & (abs(y) <= 0.05)
        if mutation == "no_selection":
            selected[:] = True
        elif mutation == "one_dimensional":
            selected = abs(y) <= 0.05
        elif mutation == "shifted":
            selected = (abs(x) <= 0.08) & (abs(y - 0.03) <= 0.05)
        strength = 0.05 if mutation == "tiny_flip" else 0.4
        exterior = 1.0 if mutation in {"crushed", "crushed24", "coherent"} else 0.05
        magnitude = np.where(selected, strength, strength * exterior)
        signal = magnitude[None, :] * np.exp(-(a.adc - rf_end)[:, None] / 0.08)
        # Nominal encoding phase of the refocused pathway at each sample.
        signal = signal * np.exp(-2j * np.pi * (a.k @ spins.positions.T))
        if mutation == "crushed":
            # Unrefocused exterior pathway: 4 extra crusher cycles per x voxel.
            signal = (
                signal * np.where(selected, 1, np.exp(2j * np.pi * 2560 * x))[None, :]
            )
        if mutation == "crushed24":
            # Exactly 24 crusher cycles per 1.5625 mm voxel: aliases a 24-grid.
            cycles = 24 / 0.0015625
            signal = (
                signal * np.where(selected, 1, np.exp(2j * np.pi * cycles * x))[None, :]
            )
        # Spatial phase, including static B0/encoding, cannot change magnitude.
        return signal * np.exp(17j * a.adc[:, None] * x)

    monkeypatch.setattr(s3, "simulate", outputs)
    response = np.array([[0.3 + 0.2j]])
    monkeypatch.setattr(s3, "canonical_response", lambda *_args, **_kwargs: response)
    checks, metrics, actual_response = s3.mechanism_probes(
        acq, case, [], np.empty((0, 3))
    )
    assert actual_response is response
    return {c["name"]: c for c in checks}, metrics


def test_rfov_selection_is_invariant_to_train_duration_and_delay(monkeypatch):
    decisions = []
    for times in ([0.001, 0.013, 0.025], [0.001, 0.054, 0.106], [0.05, 0.10, 0.16]):
        checks, metrics = assay(monkeypatch, times)
        assert all(c["passed"] for c in checks.values())
        np.testing.assert_allclose(metrics["selection_leakage_ratio"], 0.05, rtol=1e-14)
        np.testing.assert_allclose(
            checks["rfov_inside_signal"]["detail"]["min_signal"], 0.4, rtol=1e-5
        )
        decisions.append(metrics)
    for metrics in decisions[1:]:
        np.testing.assert_allclose(
            list(metrics.values()), list(decisions[0].values()), rtol=1e-5
        )


def test_same_echo_leakage_is_invariant_without_amplitude_normalization(monkeypatch):
    for times in ([0.001, 0.013, 0.025], [0.001, 0.054, 0.106]):
        checks, metrics = assay(monkeypatch, times, fam="reduced_fov")
        assert checks["spatial_selection"]["passed"]
        np.testing.assert_allclose(metrics["selection_leakage_ratio"], 0.05, rtol=1e-14)


@pytest.mark.parametrize(
    "mutation", ["no_selection", "one_dimensional", "shifted", "tiny_flip"]
)
def test_rfov_selection_rejects_wrong_spatial_profiles_and_weak_excitation(
    monkeypatch, mutation
):
    checks, _metrics = assay(monkeypatch, [0.001, 0.054, 0.106], mutation)
    if mutation == "tiny_flip":
        assert not checks["rfov_inside_signal"]["passed"]
    else:
        assert not checks["spatial_selection"]["passed"]


def test_reduced_fov_voxel_probes_credit_crusher_dephasing(monkeypatch):
    """Inner-volume designs: crushed unrefocused exterior signal cancels per voxel."""
    k = [[0, -320, 0], [0, 0, 0], [0, 315, 0]]  # first, central, last ky lines
    times = [0.001, 0.054, 0.106]
    checks, metrics = assay(monkeypatch, times, "crushed", fam="reduced_fov", k=k)
    assert checks["spatial_selection"]["passed"]
    assert metrics["selection_leakage_ratio"] < 1e-3
    np.testing.assert_allclose(
        metrics["diagnostic_signal_fraction"],
        0.4 * np.mean(np.exp(-np.asarray(times) / 0.08)),
        rtol=1e-5,
    )
    checks, metrics = assay(monkeypatch, times, "coherent", fam="reduced_fov", k=k)
    assert not checks["spatial_selection"]["passed"]
    np.testing.assert_allclose(metrics["selection_leakage_ratio"], 1.0, rtol=1e-9)


def test_selection_quadrature_does_not_alias_crusher_moments(monkeypatch):
    """24 cycles/voxel aliases a 24-point grid; the coprime pair must refine."""
    k = [[0, -320, 0], [0, 0, 0], [0, 315, 0]]
    times = [0.001, 0.054, 0.106]
    checks, metrics = assay(monkeypatch, times, "crushed24", fam="reduced_fov", k=k)
    detail = checks["spatial_selection"]["detail"]
    assert detail["quadrature_converged"]
    assert len(detail["quadrature"]) >= 2  # the first pair disagreed
    first = detail["quadrature"][0]["scores"]
    assert first[0]["leakage"] > 0.9 and first[1]["leakage"] < 0.1
    assert checks["spatial_selection"]["passed"]
    assert metrics["selection_leakage_ratio"] < 0.05


def test_rfov_reference_time_ignores_rf_after_the_shot(monkeypatch):
    times = [0.001, 0.054, 0.106]
    base, base_metrics = assay(monkeypatch, times)
    late, late_metrics = assay(monkeypatch, times, late_rf=True)
    assert late["rfov_inside_signal"]["passed"]
    np.testing.assert_allclose(
        late["rfov_inside_signal"]["detail"]["min_signal"],
        base["rfov_inside_signal"]["detail"]["min_signal"],
    )
    np.testing.assert_allclose(
        late_metrics["selection_leakage_ratio"], base_metrics["selection_leakage_ratio"]
    )
