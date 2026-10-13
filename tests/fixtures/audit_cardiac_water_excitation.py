"""Independent piecewise-exact affine Bloch check of exported binomial water RF."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pypulseq as pp
from scipy.linalg import expm


def run(root):
    output = []
    for path in root.glob("*/case.json"):
        case = json.loads(path.read_text())
        sequence_path = path.parent / "sequence.seq"
        sequence = pp.Sequence()
        sequence.read(str(sequence_path))
        rf = sequence.get_block(1).rf
        signal = rf.signal
        cuts = np.r_[0, np.flatnonzero(np.diff(signal) != 0) + 1, len(signal)]
        raster = sequence.system.rf_raster_time

        def evolve(
            tissue, frequency, case=case, cuts=cuts, signal=signal, raster=raster
        ):
            properties = case["object"]["tissue_properties"][tissue]
            density = properties["proton_density"]
            t1, t2 = properties["T1_s"], properties["T2_s"]
            state = np.array([0.0, 0.0, density, 1.0])
            for begin, end in pairwise(cuts):
                wx, wy = 2 * np.pi * signal[begin].real, 2 * np.pi * signal[begin].imag
                wz = 2 * np.pi * frequency
                generator = np.array(
                    [
                        [-1 / t2, wz, -wy, 0],
                        [-wz, -1 / t2, wx, 0],
                        [wy, -wx, -1 / t1, density / t1],
                        [0, 0, 0, 0],
                    ]
                )
                state = expm(generator * ((end - begin) * raster)) @ state
            return abs(state[0] + 1j * state[1])

        fat = case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"]
        bound = 67.5 * case["hardware"]["field_strength_T"] / 3
        offsets = np.linspace(-bound, bound, 101)
        fat_zero, water_zero = evolve("fat", 0), evolve("myocardium", 0)
        output.append(
            {
                "case_id": case["case_id"],
                "case_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sequence_sha256": hashlib.sha256(
                    sequence_path.read_bytes()
                ).hexdigest(),
                "rf_duration_ms": len(signal) * raster * 1000,
                "fat_nominal_fraction": evolve("fat", fat) / fat_zero,
                "worst_fat_fraction_over_declared_B0_bound": max(
                    evolve("fat", fat + shift) / fat_zero for shift in offsets
                ),
                "min_myocardium_fraction_over_declared_B0_bound": min(
                    evolve("myocardium", shift) / water_zero for shift in offsets
                ),
            }
        )
    report = {
        "integrator": "Independent exact affine 4x4 matrix exponential for each constant RF segment; T1/T2/recovery/chemical shift retained",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": output,
    }
    destination = root / "independent-rf-notch-audit-reproducible.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
