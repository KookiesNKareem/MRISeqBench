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
DEVICE = os.environ.get("MRISEQBENCH_SIMULATOR_DEVICE", "cuda")
if DEVICE not in {"cuda", "metal", "cpu"}:
    raise ValueError("MRISEQBENCH_SIMULATOR_DEVICE must be 'cuda', 'metal', or 'cpu'")
SIMULATOR_PROJECT = os.environ.get("MRISEQBENCH_SIMULATOR_PROJECT", "/opt/simulator")


def reconstruct_mrzero(signal, seq, source, output):
    import MRzeroCore as mr0
    import pypulseq as pp
    import torch

    metadata = {"api": "MRzeroCore.reco_adjoint", "method": "adjoint", "diagnostic": True}
    try:
        public_seq = pp.Sequence()
        public_seq.read(str(source / "sequence.seq"))
        definitions = public_seq.definitions
        if not all(key in definitions for key in ("Nx", "Ny", "FOV")):
            raise ValueError("Add Nx/Ny/Nz and FOV to the .seq file's [DEFINITIONS] section")
        matrix = tuple(int(definitions.get(key, 1)) for key in ("Nx", "Ny", "Nz"))
        fov = tuple(float(value) for value in np.atleast_1d(definitions["FOV"]))
        if len(fov) != 3 or min(matrix) < 1 or min(fov) <= 0:
            raise ValueError("Expected positive Nx/Ny/Nz and three positive FOV values")
        names = ("SLC", "SET", "REP", "AVG", "PHS", "ECO")
        labels = np.stack([seq.get_adc_labels(name.lower()).cpu().numpy() for name in names], axis=1)
        trajectory = seq.get_kspace().to(signal.device)
        images = []
        # The native adjoint forms a sample-by-voxel matrix. Accumulate its
        # linear contributions in bounded chunks instead of allocating it whole.
        chunk_size = max(1, min(512, 2**20 // int(np.prod(matrix))))
        with h5py.File(output / "reconstruction.h5", "w") as file:
            for index, values in enumerate(np.unique(labels, axis=0), 1):
                indices = np.flatnonzero(np.all(labels == values, axis=1))
                image = None
                for start in range(0, len(indices), chunk_size):
                    selected = torch.as_tensor(indices[start:start + chunk_size], device=signal.device)
                    part = mr0.reco_adjoint(
                        signal[selected], trajectory[selected], resolution=matrix,
                        FOV=fov, return_multicoil=True,
                    )
                    image = part if image is None else image + part
                data = np.moveaxis(image.detach().cpu().numpy(), 0, -1)[:, :, :, None, :, None]
                name = f"image_{index:03d}"
                group = file.create_group(name)
                group.create_dataset("image_real", data=data.real, compression="gzip")
                group.create_dataset("image_imag", data=data.imag, compression="gzip")
                acquisition = dict(zip(names, (int(value) for value in values)))
                images.append({
                    "group": name, "shape": list(data.shape),
                    "axes": ["x", "y", "z", "echo", "coil", "repetition"],
                    "labels": {"ROLE": "imaging", **acquisition},
                    "source": {key: [value] for key, value in acquisition.items()},
                })
        metadata.update(status="completed", images=images, matrix=list(matrix), fov_m=list(fov))
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    (output / "reconstruction.json").write_text(json.dumps(metadata, indent=2))


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
                f"--project={SIMULATOR_PROJECT}",
                str(Path(__file__).with_name("simulate_koma.jl")),
                str(source),
                str(output),
            ],
            check=True,
        )
        return
    if ENGINE == "komamripy":
        os.environ["MRISEQBENCH_SIMULATOR_DEVICE"] = DEVICE
        import komamripy as km
        from juliacall import Main as jl

        backend_info = str(
            jl.seval("""begin
            using KomaMRICore
            device = ENV["MRISEQBENCH_SIMULATOR_DEVICE"]
            if device == "cuda"
                @eval using CUDA
                @eval using SimWarm
                SimWarm.WARM || error("Simulator image was not prewarmed on a GPU")
                CUDA.functional() || error("CUDA is required; no CPU fallback")
                CUDA.allowscalar(false)
            elseif device == "metal"
                Base.find_package("Metal") === nothing &&
                    error("Metal backend requested, but Metal.jl is not installed in the selected Julia project")
                @eval using Metal
                Metal.functional() || error("A functional Metal GPU is required; no CPU fallback")
            end
            backend = KomaMRICore.get_backend(device != "cpu"; verbose=false)
            expected = device == "cpu" ? "CPU" : device == "cuda" ? "CUDA" : "Metal"
            KomaMRICore.name(backend) == expected ||
                error("Requested $device simulator, but KomaMRI selected $(KomaMRICore.name(backend))")
            if device != "cpu"
                backend isa KomaMRICore.KA.GPU ||
                    error("Requested $device simulator, but KomaMRI selected $(typeof(backend))")
                matches = device == "cuda" ? occursin("CUDA", string(typeof(backend))) :
                    occursin("Metal", string(typeof(backend)))
                matches || error("Requested $device simulator, but KomaMRI selected $(typeof(backend))")
            end
            device_name = if device == "cpu"
                "CPU"
            elseif device == "cuda"
                CUDA.name(CUDA.device())
            else
                KomaMRICore.device_name(backend)
            end
            string(KomaMRICore.name(backend), "|", device_name)
        end""")
        )
        _, device = backend_info.split("|", 1)
        backend = DEVICE

        obj = km.files.read_phantom(str(source / "phantom.phantom"))
        seq = km.files.read_seq(str(source / "sequence.seq"))
        sim_params = km.core.default_sim_params()
        sim_params["return_type"] = "mat"
        sim_params["gpu"] = DEVICE != "cpu"
        scanner = km.Scanner(
            limits=km.HardwareLimits(B0=case["hardware"]["field_strength_T"])
        )
        jl.include(str(Path(__file__).with_name("reconstruct_koma.jl")))
        rr_ms = (case["task"]["sequence"].get("temporal") or {}).get("rr_interval_ms")
        rr_interval_s = 1.0 if rr_ms is None else float(rr_ms) / 1000
        seq, physio = jl.resolve_simulation_physio(seq, rr_interval_s)
        physio = json.loads(str(jl.JSON.json(physio)))
        julia_signal = km.simulate(obj, seq, scanner, sim_params=sim_params)
        signal = np.asarray(julia_signal)
        jl.reconstruct_files(
            julia_signal, seq, scanner, str(output), reconstruct=km.reconstruct_with_labels
        )
        after = str(
            jl.seval("""begin
            device = ENV["MRISEQBENCH_SIMULATOR_DEVICE"]
            backend = KomaMRICore.get_backend(device != "cpu"; verbose=false)
            device_name = if device == "cpu"
                "CPU"
            elseif device == "cuda"
                CUDA.name(CUDA.device())
            else
                KomaMRICore.device_name(backend)
            end
            string(KomaMRICore.name(backend), "|", device_name)
        end""")
        )
        if after != backend_info:
            raise RuntimeError("KomaMRI simulation changed backend during execution")
    elif ENGINE == "mrzero":
        if DEVICE not in {"cuda", "cpu"}:
            raise ValueError(
                "MRzero simulation supports CUDA or CPU; set MRISEQBENCH_SIMULATOR_DEVICE=cuda or cpu"
            )
        import MRzeroCore as mr0
        import torch

        if DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for MRzero but is unavailable; no CPU fallback")
        device = torch.cuda.get_device_name() if DEVICE == "cuda" else "CPU"
        backend = DEVICE

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
        seq = mr0.Sequence.import_file(str(source / "sequence.seq"))
        graph = mr0.compute_graph(seq, obj, 200, 1e-5)
        sim_seq, sim_obj = seq, obj
        if DEVICE == "cuda":
            sim_seq, sim_obj = seq.cuda(), obj.cuda()
            # MRzeroCore 1.1.1's SimData.cuda() retains the CPU dephasing closure.
            # Rebuild the same box-voxel function on the requested device.
            from MRzeroCore.phantom.custom_voxel_phantom import build_dephasing_func

            sim_obj.dephasing_func = build_dephasing_func(
                phantom.voxel_shape, phantom.voxel_size.to(sim_obj.PD.device)
            )
        result = mr0.execute_graph(graph, sim_seq, sim_obj)
        if result.device.type != DEVICE:
            raise RuntimeError(f"MRzero returned a {result.device.type} signal; expected {DEVICE}")
        signal = result.detach().cpu().numpy()
        reconstruct_mrzero(result, seq, source, output)
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
                "backend": backend,
                "device": device,
                "cpu_fallback": False,
                **({"physio": physio} if ENGINE == "komamripy" else {}),
                **(
                    {"reconstruction": json.loads((output / "reconstruction.json").read_text())}
                    if ENGINE in {"komamripy", "mrzero"} else {}
                ),
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
                        f"RF self-test disagrees with closed form: {relative_error}"
                    )
        execution = json.loads((output / "simulation.json").read_text())
    print(
        json.dumps(
            {
                "engine": ENGINE,
                "pulseq_input": True,
                "self_test": True,
                "backend": execution["backend"],
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
