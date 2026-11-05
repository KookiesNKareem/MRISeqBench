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


def _mrzero_sequence_path(source, output, case):
    import pypulseq as pp

    seq_path = source / "sequence.seq"
    rr_ms = (
        case.get("task", {})
        .get("sequence", {})
        .get("temporal", {})
        .get("rr_interval_ms")
    )
    raw = pp.Sequence(pp.Opts(B0=case["hardware"]["field_strength_T"]))
    raw.read(str(seq_path))
    has_adc = any(raw.get_block(b).adc is not None for b in raw.block_events)
    if rr_ms is None and has_adc:
        return seq_path
    resolved = pp.Sequence(raw.system)
    rr_s = float(rr_ms) / 1000.0 if rr_ms is not None else None
    clock = 0.0
    last_r_wave = -np.inf
    raster = raw.block_duration_raster
    for number in raw.block_events:
        block = raw.get_block(number)
        if rr_s is not None:
            for event in (getattr(block, "trigger", None) or {}).values():
                if event.type == "trigger" and event.channel == "physio1":
                    t_arm = clock + float(event.delay)
                    r_wave = max(
                        np.ceil((t_arm - 1e-12) / rr_s) * rr_s,
                        last_r_wave + rr_s,
                    )
                    wait = round((r_wave - t_arm) / raster) * raster
                    if wait > 0:
                        resolved.add_block(pp.make_delay(wait))
                        clock += wait
                    last_r_wave = r_wave
        events = [
            e
            for k in ("rf", "gx", "gy", "gz", "adc")
            if (e := getattr(block, k, None)) is not None
        ]
        duration = raw.block_durations[number]
        if events:
            resolved.add_block(*events, pp.make_delay(duration))
        else:
            resolved.add_block(pp.make_delay(duration))
        clock += duration
    if not has_adc:
        resolved.add_block(
            pp.make_adc(
                num_samples=1,
                dwell=raw.system.adc_raster_time,
                delay=raw.system.adc_dead_time,
                system=raw.system,
            )
        )
    out_path = output / ".mrzero-resolved.seq"
    resolved.write(str(out_path))
    return out_path


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
                "--heap-size-hint=1G",
                "--compiled-modules=existing",
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
        from juliacall import Main as jl

        device = str(
            jl.seval("""begin
            using CUDA, KomaMRICore, SimWarm
            SimWarm.WARM || error("Simulator image was not prewarmed on a GPU")
            CUDA.functional() || error("CUDA is required; no CPU fallback")
            CUDA.allowscalar(false)
            KomaMRICore.get_backend(true) isa KomaMRICore.KA.GPU ||
                error("KomaMRI simulation cannot fall back to CPU")
            CUDA.name(CUDA.device())
        end""")
        )

        obj = km.files.read_phantom(str(source / "phantom.phantom"))
        seq = km.files.read_seq(str(source / "sequence.seq"))
        rr_ms = (
            case.get("task", {})
            .get("sequence", {})
            .get("temporal", {})
            .get("rr_interval_ms")
        )
        if rr_ms is not None:
            seq = jl.KomaMRIBase.resolve_triggers(
                seq,
                jl.KomaMRIBase.CardiacSignal(
                    heart_rate=1000.0 / float(rr_ms), first_peak=0.0
                ),
            )
        has_adc = bool(
            int(jl.seval("seq -> sum(adc.N for adc in seq.ADC; init=0)")(seq)) > 0
        )
        sim_params = km.core.default_sim_params()
        sim_params["return_type"] = "mat" if has_adc else "state"
        sim_params["gpu"] = True
        raw_out = km.simulate(
            obj,
            seq,
            km.Scanner(
                limits=km.HardwareLimits(B0=case["hardware"]["field_strength_T"])
            ),
            sim_params=sim_params,
        )
        signal = np.asarray(raw_out if has_adc else raw_out.xy)
    elif ENGINE == "mrzero":
        import MRzeroCore as mr0
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for MRzero simulation; no CPU fallback"
            )
        device = torch.cuda.get_device_name()

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
            phantom = mr0.CustomVoxelPhantom(
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
            )
            obj = phantom.build()
        resolved_path = _mrzero_sequence_path(source, output, case)
        seq = mr0.Sequence.import_file(str(resolved_path))
        if resolved_path != source / "sequence.seq":
            resolved_path.unlink(missing_ok=True)
        graph = mr0.compute_graph(seq, obj, 200, 1e-5)
        cuda_obj = obj.cuda()
        # MRzeroCore 1.1.1's SimData.cuda() retains the CPU dephasing closure.
        # Rebuild the same box-voxel function with its size on the CUDA device.
        from MRzeroCore.phantom.custom_voxel_phantom import build_dephasing_func

        cuda_obj.dephasing_func = build_dephasing_func(
            phantom.voxel_shape, phantom.voxel_size.to(cuda_obj.PD.device)
        )
        result = mr0.execute_graph(graph, seq.cuda(), cuda_obj)
        if result.device.type != "cuda":
            raise RuntimeError("MRzero simulation returned a CPU signal")
        signal = result.detach().cpu().numpy()
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
                "backend": "cuda",
                "device": device,
                "cpu_fallback": False,
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
            if ENGINE in {"koma", "komamripy"}:
                # Independent closed form for one resonant rectangular RF
                # pulse, with T1/T2 relaxation; no CPU simulator or fitted gain.
                omega, duration, r1, r2 = (np.pi / 6) / 1e-3, 1e-3, 1.0, 10.0
                frequency = np.sqrt(omega**2 - ((r1 - r2) / 2) ** 2)
                y_eq = omega * r1 / (omega**2 + r1 * r2)
                z_eq = r1 * r2 / (omega**2 + r1 * r2)
                y = y_eq + np.exp(-(r1 + r2) * duration / 2) * (
                    -y_eq * np.cos(frequency * duration)
                    + np.sin(frequency * duration)
                    / frequency
                    * ((r1 - r2) / 2 * -y_eq + omega * (1 - z_eq))
                )
                expected = y * np.exp(-(np.arange(4) + 0.5) * 1e-5 * r2)
                relative_error = float(
                    np.linalg.norm(np.abs(signal).ravel() - expected)
                    / np.linalg.norm(expected)
                )
                if relative_error > 1e-3:
                    raise ValueError(
                        f"CUDA RF self-test disagrees with closed form: {relative_error}"
                    )
        execution = json.loads((output / "simulation.json").read_text())
    print(
        json.dumps(
            {
                "engine": ENGINE,
                "pulseq_input": True,
                "self_test": True,
                "backend": "cuda",
                "device": execution["device"],
                "cpu_fallback": False,
                **(
                    {"RF_closed_form_magnitude_relative_error": relative_error}
                    if ENGINE in {"koma", "komamripy"}
                    else {}
                ),
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        probe()
    elif len(sys.argv) == 3:
        simulate(sys.argv[1], sys.argv[2])
    else:
        raise ValueError(
            "usage: simulate --probe | simulate INPUT_DIRECTORY OUTPUT_DIRECTORY"
        )
