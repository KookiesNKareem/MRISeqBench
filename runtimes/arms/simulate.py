#!/usr/bin/env python3
"""Fixed simulator ABI; accepts Pulseq data and public phantom inputs only."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ENGINE = os.environ["MRISEQBENCH_SIMULATOR"]


def simulate(source, output):
    source, output = Path(source), Path(output)
    case = json.loads((source / "case.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("requires_field_updates"):
        raise ValueError(
            "this adapter does not support time-varying fields or motion through nonuniform fields"
        )
    fields = source / "fields.h5"
    if fields.exists():
        with h5py.File(fields) as f:
            if not np.allclose(f["b1_scale_t0"][()], 1):
                raise ValueError("this adapter does not support nonuniform B1")
    if ENGINE == "koma":
        subprocess.run(
            [
                "julia",
                "--startup-file=no",
                "--project=/opt/simulator",
                "/opt/mriseqbench/simulate_koma.jl",
                str(source),
                str(output),
            ],
            check=True,
        )
        return
    if ENGINE == "komamripy":
        import komamripy as km

        obj = km.files.read_phantom(str(source / "phantom.phantom"))
        seq = km.files.read_seq(str(source / "sequence.seq"))
        sim_params = km.core.default_sim_params()
        sim_params["return_type"] = "mat"
        sim_params["gpu"] = False
        signal = np.asarray(
            km.simulate(
                obj,
                seq,
                km.Scanner(
                    limits=km.HardwareLimits(B0=case["hardware"]["field_strength_T"])
                ),
                sim_params=sim_params,
            )
        )
    elif ENGINE == "mrzero":
        import MRzeroCore as mr0

        if any(
            e["kind"] in {"motion", "time_variation"}
            for e in case["physics"]["effects"]
        ):
            raise ValueError(
                "MRzero adapter does not support this motion/time-variation profile"
            )
        with h5py.File(source / "phantom.phantom") as f:
            positions = np.stack(
                [f["position"][k][()] for k in ("x", "y", "z")], axis=1
            ).astype(np.float32)
            c = f["contrast"]
            # MRzero's imported sequence and coil tensors use float32/complex64.
            # Native Koma phantom files store float64; mixing these breaks the
            # signal matrix product inside execute_graph.
            obj = mr0.CustomVoxelPhantom(
                pos=positions,
                PD=c["ρ"][()].astype(np.float32),
                T1=c["T1"][()].astype(np.float32),
                T2=c["T2"][()].astype(np.float32),
                T2dash=1e6,
                D=0,
                B0=c["Δw"][()].astype(np.float32) / (2 * np.pi),
                B1=1,
                voxel_size=1e-9,
                voxel_shape="box",
            ).build()
        seq = mr0.Sequence.import_file(str(source / "sequence.seq"))
        graph = mr0.compute_graph(seq, obj, 200, 1e-5)
        signal = mr0.execute_graph(graph, seq, obj).detach().cpu().numpy()
    else:
        raise ValueError("unsupported engine: " + ENGINE)
    with h5py.File(output / "signal.h5", "w") as f:
        f["signal_real"] = np.real(signal)
        f["signal_imag"] = np.imag(signal)
    (output / "simulation.json").write_text(
        json.dumps(
            {
                "engine": ENGINE,
                "diagnostic": True,
                "phantom": "public materialized spin centres",
                "signal_shape": list(signal.shape),
            }
        )
    )


def probe():
    import pypulseq as pp

    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "input", Path(tmp) / "output"
        source.mkdir()
        output.mkdir()
        system = pp.Opts()
        seq = pp.Sequence(system)
        seq.add_block(
            pp.make_block_pulse(
                flip_angle=np.pi / 6, duration=1e-3, system=system, use="excitation"
            )
        )
        seq.add_block(pp.make_adc(num_samples=4, dwell=1e-5, system=system))
        seq.write(str(source / "sequence.seq"))
        with h5py.File(source / "phantom.phantom", "w") as f:
            f.attrs.update(Version="0.9.9", Name="arm-runtime-self-test", Ns=1, Dims=0)
            for k in ("x", "y", "z"):
                f["position/" + k] = [0.0]
            for k, v in {"ρ": 1.0, "T1": 1.0, "T2": 0.1, "T2s": 1e6, "Δw": 0.0}.items():
                f["contrast/" + k] = [v]
        (source / "case.json").write_text(
            json.dumps(
                {"hardware": {"field_strength_T": 3.0}, "physics": {"effects": []}}
            )
        )
        (source / "manifest.json").write_text(
            json.dumps({"requires_field_updates": False})
        )
        simulate(source, output)
        with h5py.File(output / "signal.h5") as f:
            signal = f["signal_real"][()] + 1j * f["signal_imag"][()]
            if (
                signal.size != 4
                or not np.all(np.isfinite(signal))
                or not np.any(np.abs(signal) > 0.01)
            ):
                raise ValueError("Pulseq-to-signal self-test failed")
    print(json.dumps({"engine": ENGINE, "pulseq_input": True, "self_test": True}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        probe()
    elif len(sys.argv) == 3:
        simulate(sys.argv[1], sys.argv[2])
    else:
        raise ValueError(
            "usage: simulate --probe | simulate INPUT_DIRECTORY OUTPUT_DIRECTORY"
        )
