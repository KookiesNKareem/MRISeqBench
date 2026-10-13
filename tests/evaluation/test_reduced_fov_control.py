"""Independent exported-RF and hardware checks for the reduced-FOV repair."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.preflight import preflight
from mriseqbench.evaluation.s3_sequence import inspect_sequence
from mriseqbench.evaluation.sar import evaluate_sar
from tests.fixtures.calibrate_cardiac import options_for
from tests.fixtures.s3_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]
REPAIR_OPTIONS = {
    "selective_rf_duration": 0.0096,
    "reduced_refocusing": "composite",
    "reduced_refocusing_duration": 0.0008,
    "reduced_flip_deg": 90,
    "preparations": 1,
    "repetition_time": 0.34,
    "readout_flat": 0.00032,
    "reduced_fat_saturation_duration": 0.008,
}


def test_qualified_cardiac_control_is_automatic_without_changing_legacy_author():
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "reduced_fov"))
    options = options_for(case, None)
    assert options == {
        **REPAIR_OPTIONS,
        "repetition_time": 0.352,
        "reduced_fat_saturation": True,
        "reduced_fat_saturation_pulses": 2,
    }
    case["evaluation"]["intrinsic_tissue_response"] = False
    assert options_for(case, None) == {"readout_flat": 0.00032}


def refocusing_matrix(pulse, frequency):
    result = np.eye(3)
    for value, duration in zip(pulse.signal, np.diff(pulse.edges), strict=True):
        field = 2 * np.pi * np.array([value.real, value.imag, frequency])
        result = Rotation.from_rotvec(-field * duration).as_matrix() @ result
    return result


@pytest.mark.parametrize("fat_saturation_pulses", [0, 1, 2])
@pytest.mark.parametrize("hardware", ["low_field", "standard_1_5t", "high_performance"])
def test_standard_composite_control_is_exported_robust_and_hardware_legal(
    tmp_path, hardware, fat_saturation_pulses
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "reduced_fov"), hardware_ref=hardware)
    path = tmp_path / "sequence.seq"
    options = dict(REPAIR_OPTIONS)
    if fat_saturation_pulses:
        options.update(
            reduced_fat_saturation=True,
            reduced_fat_saturation_pulses=fat_saturation_pulses,
            repetition_time=0.352,
        )
    make_sequence(case, **options).write(str(path))
    acq = inspect_sequence(path, case)
    checks = preflight(path, case) + evaluate_sar(path, case["hardware"]) + acq.checks
    assert all(item["passed"] for item in checks), checks
    refocusing = next(pulse for pulse in acq.pulses if pulse.refocusing)
    assert refocusing.flip == pytest.approx(180, abs=1e-5)
    assert len(refocusing.signal) == 3
    fat = case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"]
    extent = 70 * case["hardware"]["field_strength_T"] / 3
    frequencies = np.r_[
        np.linspace(-extent, extent, 17), fat + np.linspace(-extent, extent, 17)
    ]
    if fat_saturation_pulses:
        saturation = next(
            pulse for pulse in acq.pulses if abs(pulse.frequency - fat) < 0.001
        )
        # CHESS must leave negligible water coherence outside the selective slab.
        # This catches the earlier short Gaussian's exterior-water FID.
        for water in np.linspace(-extent, extent, 17):
            response = refocusing_matrix(saturation, water - saturation.frequency)
            assert np.linalg.norm(response[:2, 2]) < 0.02
    for frequency in frequencies:
        rotation = refocusing_matrix(refocusing, frequency)
        conjugate = (
            rotation[0, 0] - rotation[1, 1] + 1j * (rotation[1, 0] + rotation[0, 1])
        ) / 2
        # Independent spin-echo coherence assay, rather than nominal flip alone.
        assert abs(conjugate) >= 0.9
