"""Reuse only identical native zero-effect inputs across S3 physics controls.

No physical signal, reference, score, interval or verdict is transferred. The
frozen waveform, hardware, object, geometry, actual quadrature nodes, materials,
weights, properties and integration steps must describe the same native call.
"""

import argparse
import fcntl
import hashlib
import inspect
import io
import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from migrate_s3_replays import implementation, key, read, sha

from mriseqbench.evaluation.s1_sampling import sampled_phantom


def inputs(campaign, source, destination, samples, replicate, steps):
    a, b = read(source / "case.json"), read(destination / "case.json")
    assert all(a[k] == b[k] for k in ("task", "hardware", "object"))
    sequence = sha(source / "sequence.seq")
    assert sequence == sha(destination / "sequence.seq")
    nodes = []
    for case in (a, b):
        spins = sampled_phantom(case, campaign / "source", samples, replicate)
        nodes.append(
            replace(
                spins, weight=np.broadcast_to(spins.weight, (len(spins.positions),))
            )
        )
    signature = implementation(campaign / "source") + sequence
    assert key(signature, nodes[0], [], steps) == key(signature, nodes[1], [], steps)
    # A retained native sum can be a cache leaf only when it was one batch.
    for case in (a, b):
        limit = max(
            64,
            int(
                min(case["evaluation"]["spatial_max_samples"], 32768)
                * min(1, steps["rf_step_s"] / 20e-6)
            ),
        )
        assert len(nodes[0].positions) <= limit
    identity = (
        implementation(campaign / "source")
        + sequence
        + json.dumps({"case": b, "encoded_integrator": "bloch"}, sort_keys=True)
    )
    return key(signature, nodes[0], [], steps), key(identity, nodes[1], [], steps)


def verify(campaign, receipt):
    assert sha(campaign / "cross_control_ideal_reuse_tool.py") == receipt["tool_sha256"]
    hashes = read(campaign / "source-hashes.json")
    assert (
        sha(Path(inspect.getsourcefile(sampled_phantom)))
        == hashes["src/mriseqbench/evaluation/s1_sampling.py"]
    )
    assert all(
        sha(campaign / "source" / rel) == digest for rel, digest in hashes.items()
    )
    assert receipt["implementation_sha256"] == implementation(campaign / "source")
    plan = {r["directory"]: r for r in read(campaign / "plan.json")}
    total = 0
    for row in receipt["records"]:
        source, destination = (
            Path(row["source_control"]),
            Path(row["destination_control"]),
        )
        assert plan[str(source)]["role"] == "ideal_positive"
        assert plan[str(destination)]["role"] == "physics_characterization"
        for directory in (source, destination):
            assert sha(directory / "case.json") == plan[str(directory)]["case_sha256"]
            assert (
                sha(directory / "sequence.seq")
                == plan[str(directory)]["sequence_sha256"]
            )
        record = read(source / "record.json")
        assert record["case_sha256"] == plan[str(source)]["case_sha256"]
        assert record["sequence_sha256"] == plan[str(source)]["sequence_sha256"]
        result_path = Path(record["result"])
        assert sha(result_path) == record["result_sha256"]
        result = read(result_path)
        image_path, provenance_path = (
            Path(result["artifacts"]["images"]),
            Path(result["artifacts"]["provenance"]),
        )
        assert (
            sha(image_path)
            == record["artifact_sha256"]["images"]
            == row["source_images_sha256"]
        )
        assert sha(provenance_path) == record["artifact_sha256"]["provenance"]
        provenance = read(provenance_path)
        assert all(
            version(name) == value for name, value in provenance["versions"].items()
        )
        assert row["samples"] == provenance["spatial_samples"]
        assert row["steps"] == {
            "step_s": provenance["dynamic_step_s"][0],
            "rf_step_s": provenance["rf_step_s"][0],
        }
        assert read(source / "case.json")["physics"]["effects"] == []
        for directory, digest in [
            (source, row["source_case_sha256"]),
            (destination, row["destination_case_sha256"]),
        ]:
            assert sha(directory / "case.json") == digest
        assert sorted(signal["replicate"] for signal in row["signals"]) == list(
            range(provenance["spatial_replicates"])
        )
        with np.load(image_path, allow_pickle=False) as images:
            for signal in row["signals"]:
                assert (
                    Path(signal["destination"]).parent
                    == destination / "evaluation/physical/s3/replays"
                )
                native, target = inputs(
                    campaign,
                    source,
                    destination,
                    row["samples"],
                    signal["replicate"],
                    row["steps"],
                )
                assert (
                    native == signal["native_input_key"]
                    and target == Path(signal["destination"]).name
                )
                value = images["koma_ideal_raw_replicates"][signal["replicate"]]
                assert np.isfinite(value).all()
                destination_file = Path(signal["destination"])
                if destination_file.exists():
                    assert sha(destination_file) == signal["destination_sha256"]
                stream = io.BytesIO()
                np.save(stream, value, allow_pickle=False)
                assert (
                    hashlib.sha256(stream.getvalue()).hexdigest()
                    == signal["destination_sha256"]
                )
                total += 1
            completed = destination / "record.json"
            if completed.exists():
                destination_record = read(completed)
                destination_result = read(Path(destination_record["result"]))
                destination_provenance = read(
                    Path(destination_result["artifacts"]["provenance"])
                )
                destination_steps = {
                    "step_s": destination_provenance["dynamic_step_s"][0],
                    "rf_step_s": destination_provenance["rf_step_s"][0],
                }
                if (
                    destination_provenance["spatial_samples"] == row["samples"]
                    and destination_steps == row["steps"]
                ):
                    destination_images = Path(destination_result["artifacts"]["images"])
                    assert (
                        sha(destination_images)
                        == destination_record["artifact_sha256"]["images"]
                    )
                    with np.load(destination_images, allow_pickle=False) as actual:
                        np.testing.assert_array_equal(
                            actual["koma_ideal_raw_replicates"],
                            images["koma_ideal_raw_replicates"],
                        )
    return {
        "controls": len(receipt["records"]),
        "native_zero_effect_signals": total,
        "input_equivalence_verified": True,
    }


def run(campaign):
    campaign = campaign.resolve()
    path = campaign / "cross_control_ideal_reuse.json"
    assert not path.exists() and not (campaign / "complete.json").exists()
    hashes = read(campaign / "source-hashes.json")
    assert all(
        sha(campaign / "source" / rel) == digest for rel, digest in hashes.items()
    )
    assert (
        sha(Path(inspect.getsourcefile(sampled_phantom)))
        == hashes["src/mriseqbench/evaluation/s1_sampling.py"]
    )
    plan = read(campaign / "plan.json")
    positives = [
        r
        for r in plan
        if r["role"] == "ideal_positive"
        and (Path(r["directory"]) / "record.json").exists()
    ]
    receipt = {
        "method": __doc__,
        "implementation_sha256": implementation(campaign / "source"),
        "tool_sha256": sha(Path(__file__)),
        "records": [],
    }
    for row in plan:
        destination = Path(row["directory"])
        if (
            row["role"] != "physics_characterization"
            or (destination / "record.json").exists()
        ):
            continue
        with (destination / ".evaluation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (destination / "record.json").exists():
                continue
            assert sha(destination / "case.json") == row["case_sha256"]
            assert sha(destination / "sequence.seq") == row["sequence_sha256"]
            case = read(destination / "case.json")
            for prior in positives:
                source = Path(prior["directory"])
                before = read(source / "case.json")
                if (
                    any(before[k] != case[k] for k in ("task", "hardware", "object"))
                    or prior["sequence_sha256"] != row["sequence_sha256"]
                ):
                    continue
                assert sha(source / "case.json") == prior["case_sha256"]
                assert sha(source / "sequence.seq") == prior["sequence_sha256"]
                record = read(source / "record.json")
                result_path = Path(record["result"])
                assert sha(result_path) == record["result_sha256"]
                result = read(result_path)
                assert all(
                    sha(Path(path)) == record["artifact_sha256"][name]
                    for name, path in result.get("artifacts", {}).items()
                )
                provenance = read(Path(result["artifacts"]["provenance"]))
                assert all(
                    version(name) == value
                    for name, value in provenance["versions"].items()
                )
                if (
                    provenance["spatial_samples"]
                    != case["evaluation"]["spatial_samples"]
                ):
                    continue
                steps = {
                    "step_s": provenance["dynamic_step_s"][0],
                    "rf_step_s": provenance["rf_step_s"][0],
                }
                image_path = Path(result["artifacts"]["images"])
                signals = []
                cache = destination / "evaluation/physical/s3/replays"
                cache.mkdir(parents=True, exist_ok=True)
                with np.load(image_path, allow_pickle=False) as images:
                    for replicate, value in enumerate(
                        images["koma_ideal_raw_replicates"]
                    ):
                        native, target = inputs(
                            campaign,
                            source,
                            destination,
                            provenance["spatial_samples"],
                            replicate,
                            steps,
                        )
                        output = cache / target
                        if output.exists():
                            np.testing.assert_array_equal(
                                np.load(output, allow_pickle=False), value
                            )
                        else:
                            temporary = output.with_suffix(".tmp")
                            with temporary.open("wb") as stream:
                                np.save(stream, value, allow_pickle=False)
                            temporary.replace(output)
                        signals.append(
                            {
                                "replicate": replicate,
                                "native_input_key": native,
                                "destination": str(output),
                                "destination_sha256": sha(output),
                            }
                        )
                receipt["records"].append(
                    {
                        "source_control": str(source),
                        "destination_control": str(destination),
                        "source_case_sha256": sha(source / "case.json"),
                        "destination_case_sha256": sha(destination / "case.json"),
                        "source_images_sha256": sha(image_path),
                        "samples": provenance["spatial_samples"],
                        "steps": steps,
                        "signals": signals,
                    }
                )
                break
    (campaign / "cross_control_ideal_reuse_tool.py").write_bytes(
        Path(__file__).read_bytes()
    )
    checked = verify(campaign, receipt)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(checked), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    run(parser.parse_args().campaign)
