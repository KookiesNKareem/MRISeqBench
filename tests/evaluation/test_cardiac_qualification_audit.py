"""Coverage receipts must not turn partial or changed evidence into calibration."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from tests.fixtures.audit_cardiac_qualification import (
    REQUIRED_SCIENTIFIC_SOURCES,
    ROOT,
    audit,
    contract,
    explicit_failure,
    qualified,
    sealed_native_artifacts,
    sealed_s4_artifacts,
    selected_s4_negatives,
    sha,
    source_manifest,
    verified,
    verify_receiver_negative,
)


def frozen_sources(campaign, name="source_manifest.json"):
    source = campaign / "source"
    for relative in REQUIRED_SCIENTIFIC_SOURCES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen scientific source")
    declared = {
        str(path.relative_to(source)): sha(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    (campaign / name).write_text(json.dumps(declared))
    return declared


@pytest.fixture
def native_artifacts(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    paths, hashes = {}, {}
    for name in ("sequence", "images", "provenance", "policy"):
        path = directory / name
        path.write_text(f"synthetic sealed {name}")
        paths[name], hashes[name] = str(path), sha(path)
    return directory, {"artifact_sha256": hashes}, {"artifacts": paths}


def test_native_positive_requires_every_artifact_hash(native_artifacts):
    directory, row, report = native_artifacts
    sealed_native_artifacts(directory, row, report)
    for missing in ("sequence", "images", "provenance", "policy"):
        altered = deepcopy(row)
        altered["artifact_sha256"].pop(missing)
        with pytest.raises(ValueError, match="artifact hash coverage"):
            sealed_native_artifacts(directory, altered, report)
    with pytest.raises(ValueError, match="artifact hash coverage"):
        sealed_native_artifacts(directory, {"artifact_sha256": {}}, report)


@pytest.mark.parametrize("corruption", ["changed", "outside", "symlink"])
def test_native_positive_rejects_changed_or_unrelated_artifact(
    native_artifacts, corruption
):
    directory, row, report = native_artifacts
    if corruption == "changed":
        Path(report["artifacts"]["images"]).write_text("changed")
    else:
        outside = directory.parent / "unrelated-image"
        outside.write_text("synthetic image with deliberately matching receipt")
        row["artifact_sha256"]["images"] = sha(outside)
        if corruption == "symlink":
            link = directory / "image-link"
            link.symlink_to(outside)
            report["artifacts"]["images"] = str(link)
        else:
            report["artifacts"]["images"] = str(outside)
    with pytest.raises(ValueError):
        sealed_native_artifacts(directory, row, report)


@pytest.mark.parametrize("corruption", [None, "traversal", "absolute", "symlink"])
def test_source_manifest_is_contained_and_has_no_aliases(tmp_path, corruption):
    campaign = tmp_path / "campaign"
    source = campaign / "source"
    declared = frozen_sources(campaign, "source-hashes.json")
    original = source / "source.py"
    original.write_text("synthetic frozen source")
    relative, digest = "source.py", sha(original)
    if corruption in {"traversal", "absolute"}:
        outside = campaign / "outside.py"
        outside.write_text("synthetic external source")
        relative = "../outside.py" if corruption == "traversal" else str(outside)
        digest = sha(outside)
    elif corruption == "symlink":
        (source / "alias.py").symlink_to(original)
        relative = "alias.py"
    declared[relative] = digest
    (campaign / "source-hashes.json").write_text(json.dumps(declared))
    if corruption is None:
        assert source_manifest(campaign)["sha256"] == sha(
            campaign / "source-hashes.json"
        )
    else:
        with pytest.raises(ValueError):
            source_manifest(campaign)


def test_incomplete_failed_s4_payload_cannot_count_as_a_negative(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "adiabatic_inversion"))
    campaign = tmp_path / "campaign"
    directory = (
        campaign / "cases/adiabatic_inversion--high_performance--missing_trigger"
    )
    directory.mkdir(parents=True)
    frozen_sources(campaign)
    (directory / "case.json").write_text(json.dumps(case))
    (directory / "sequence.seq").write_text("synthetic exported sequence")
    report = {
        "status": "failed",
        "sequence_valid": False,
        "checks": [],
        "image_checks": [],
    }
    (directory / "result.json").write_text(json.dumps(report))
    row = {
        "name": directory.name,
        "mutation": "missing_trigger",
        "status": "failed",
        "case_sha256": sha(directory / "case.json"),
        "sequence_sha256": sha(directory / "sequence.seq"),
        "result_sha256": sha(directory / "result.json"),
    }
    (campaign / "results.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="no explicit failed check"):
        audit([campaign], [])


def test_complete_audit_requires_receiver_campaign_bart_summary(tmp_path):
    # Progress can inspect an in-flight campaign. Publication must wait for its
    # final BART identity receipt even if the last per-case record exists.
    assert audit([], [tmp_path])["complete"] is False
    with pytest.raises(ValueError, match="Receiver campaign summary is not sealed"):
        audit([], [tmp_path], complete=True)


def test_native_negative_failure_can_be_in_a_later_frame_or_preparation():
    assert not explicit_failure({"status": "failed"})
    assert explicit_failure({"application_checks": [{"passed": False}]})
    assert explicit_failure({"frames": [{"image_checks": [{"passed": False}]}]})


def test_qualification_requires_every_check_and_resolved_native_evidence():
    report = {
        "status": "uncalibrated",
        "sequence_valid": True,
        "numerically_converged": True,
        "checks": [{"passed": True}],
        "image_checks": [{"passed": True}],
    }
    assert qualified(report)
    for key, value in (
        ("status", "numerically_unresolved"),
        ("numerically_converged", False),
        ("sequence_valid", False),
        ("image_checks", []),
        ("checks", []),
        ("checks", [{"passed": False}]),
        ("image_checks", [{"passed": None}]),
        ("frames", [{"status": "uncalibrated"}]),
    ):
        altered = {**report, key: value}
        assert not qualified(altered)


def test_contract_accepts_independent_seed_but_not_budget_or_threshold_changes():
    catalog = Catalog(ROOT)
    current = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    altered = deepcopy(current)
    altered["evaluation"]["spatial_seed"] += 1
    contract(altered, current)
    altered["evaluation"]["spatial_replicates"] //= 2
    with pytest.raises(ValueError, match="spatial_replicates"):
        contract(altered, current)
    altered = deepcopy(current)
    altered["evaluation"]["case_image_criteria"]["static"]["phase_rms_rad_max"] += 0.01
    with pytest.raises(ValueError, match="case_image_criteria"):
        contract(altered, current)


@pytest.mark.parametrize("level_task", ["gre_t1w", "adiabatic_inversion"])
def test_small_pilot_budget_cannot_certify_current_contract(level_task):
    catalog = Catalog(ROOT)
    current = catalog.resolve(catalog.get("tasks", level_task))
    for key, value in (("spatial_samples", 4096), ("spatial_replicates", 4)):
        pilot = deepcopy(current)
        pilot["evaluation"][key] = value
        with pytest.raises(ValueError, match=key):
            contract(pilot, current)


def test_only_failed_reduced_fov_metadata_substitution_is_retained():
    catalog = Catalog(ROOT)
    current = catalog.resolve(catalog.get("tasks", "reduced_fov"))
    old = deepcopy(current)
    capabilities = old["task"]["sequence"]["capabilities"]
    capabilities[capabilities.index("spin_echo")] = "spoiled_gre"
    report = {
        "status": "uncalibrated",
        "sequence_valid": True,
        "numerically_converged": True,
        "checks": [{"passed": True}],
        "image_checks": [{"passed": False}],
    }
    assert contract(old, current, report=report) == [
        "failed_reduced_fov_capability_metadata_only"
    ]
    for altered_report in (
        {**report, "image_checks": [{"passed": True}]},
        {**report, "status": "numerically_unresolved"},
    ):
        with pytest.raises(ValueError, match="task"):
            contract(old, current, report=altered_report)
    with pytest.raises(ValueError, match="task"):
        contract(old, current, report=report, mutation="wrong_fov")
    old["task"]["sequence"]["duration_s"] += 1
    with pytest.raises(ValueError, match="task"):
        contract(old, current, report=report)


def test_incomplete_suite_cannot_be_certified():
    result = audit([], [])
    assert result["qualified_positive_count"] == 0
    assert result["expected_positive_count"] == 75
    assert len(result["missing_positive_cases"]) == 75
    assert len(result["missing_receiver_negative_cases"]) == 63
    assert len(result["missing_s4_negative_cases"]) == 124
    assert not result["complete"]
    with pytest.raises(ValueError, match="Incomplete qualification"):
        audit([], [], complete=True)


def test_changed_bytes_invalidate_receipt(tmp_path):
    path = tmp_path / "native-result.json"
    path.write_text('{"status":"failed"}')
    digest = sha(path)
    verified(path, digest)
    path.write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="Changed evidence"):
        verified(path, digest)


def test_every_s4_frame_must_independently_qualify():
    frame = {
        "status": "uncalibrated",
        "sequence_valid": True,
        "numerically_converged": True,
        "checks": [{"passed": True}],
        "image_checks": [{"passed": True}],
    }
    report = {
        **frame,
        "frames": [deepcopy(frame) for _ in range(3)],
        "application_checks": [{"passed": True}],
    }
    assert qualified(report, frames_required=True)
    report["frames"][1]["image_checks"][0]["passed"] = False
    assert not qualified(report, frames_required=True)
    assert not qualified(frame, frames_required=True)


def test_s4_negatives_cannot_certify_a_different_repaired_control():
    selected = {"case": {"sequence_sha256": "repaired"}}
    candidates = [("mutation", {"case_id": "case", "base_sequence_sha256": "old"})]
    assert selected_s4_negatives(candidates, selected) == {}
    # A source-only optimization with byte-identical unmutated control remains paired.
    candidates.append(
        ("mutation", {"case_id": "case", "base_sequence_sha256": "repaired"})
    )
    assert (
        selected_s4_negatives(candidates, selected)["mutation"]["base_sequence_sha256"]
        == "repaired"
    )


@pytest.fixture
def sealed_frames(tmp_path):
    campaign = tmp_path / "campaign"
    directory = campaign / "cases/control"
    directory.mkdir(parents=True)
    row = {"name": "control"}
    record = {"frames": []}
    for filename, name, key in (
        ("case.json", "case", "case_sha256"),
        ("sequence.seq", "sequence", "sequence_sha256"),
        ("result.json", "result", "result_sha256"),
    ):
        path = directory / filename
        path.write_text("synthetic protocol fixture")
        row[key] = sha(path)
        record[name] = {"path": str(path.relative_to(campaign)), "sha256": sha(path)}
    report = {"frames": []}
    for index in range(3):
        paths, receipts = {}, {}
        for key in ("images", "provenance", "sequence"):
            path = directory / f"frame-{index}-{key}"
            path.write_text(f"synthetic frame artifact {index}/{key}")
            paths[key] = str(path)
            receipts[key] = {
                "path": str(path.relative_to(campaign)),
                "sha256": sha(path),
            }
        report["frames"].append({"artifacts": paths})
        record["frames"].append({"index": index, "artifacts": receipts})
    seal = campaign / "evidence_manifest.json"
    seal.write_text(json.dumps({"cases": {"control": record}}))
    return campaign, directory, row, report, {"task": {"id": "t1_mapping"}}, seal


def test_all_sealed_frame_artifacts_are_bound_to_the_selected_report(sealed_frames):
    campaign, directory, row, report, case, seal = sealed_frames
    assert sealed_s4_artifacts(campaign, directory, row, report, case) == sha(seal)


@pytest.mark.parametrize(
    "corruption",
    ["missing_frame", "unrelated_seal", "unrelated_report", "changed_artifact"],
)
def test_missing_or_unrelated_sealed_frames_cannot_certify(sealed_frames, corruption):
    campaign, directory, row, report, case, seal = sealed_frames
    if corruption == "missing_frame":
        report["frames"].pop()
    elif corruption == "changed_artifact":
        from pathlib import Path

        Path(report["frames"][0]["artifacts"]["images"]).write_text("changed")
    else:
        unrelated = campaign / "unrelated-image"
        unrelated.write_text("unrelated native evidence")
        if corruption == "unrelated_report":
            report["frames"][0]["artifacts"]["images"] = str(unrelated)
        else:
            data = json.loads(seal.read_text())
            data["cases"]["control"]["frames"][0]["artifacts"]["images"] = {
                "path": unrelated.name,
                "sha256": sha(unrelated),
            }
            seal.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        sealed_s4_artifacts(campaign, directory, row, report, case)


@pytest.fixture
def receiver_fixture(tmp_path):
    positive = tmp_path / "positive.seq"
    text = "[RF]\n1 1 2 3\n[ADC]\n1 8 100 0 0 0 0 0 0\n[SIGNATURE]\noriginal\n"
    positive.write_text(text)
    negative = tmp_path / "receiver-rotation.seq"
    negative.write_text(
        text.replace("0 0 0\n[SIGNATURE]", "0 1.5707963267948966 0\n[SIGNATURE]")
    )
    images = tmp_path / "negative-images.npz"
    images.write_bytes(b"synthetic image receipt")
    row = {"sequence_sha256": sha(negative), "artifact_sha256": sha(images)}
    return tmp_path, row, {"sequence": str(positive)}


def test_receiver_negative_export_is_verified_against_selected_positive(
    receiver_fixture,
):
    directory, row, positive = receiver_fixture
    verify_receiver_negative(directory, row, positive)


def test_receiver_negative_cannot_change_unrelated_rf_bytes(receiver_fixture):
    directory, row, positive = receiver_fixture
    path = directory / "receiver-rotation.seq"
    path.write_text(path.read_text().replace("1 1 2 3", "1 9 2 3"))
    row["sequence_sha256"] = sha(path)
    with pytest.raises(ValueError, match="RF, gradient or timing"):
        verify_receiver_negative(directory, row, positive)


def test_failed_fresh_native_crosscheck_cannot_be_ignored(receiver_fixture):
    directory, row, positive = receiver_fixture
    row["fresh_native_crosscheck"] = {"passed": False, "raw_relative_error": 0.01}
    with pytest.raises(ValueError, match="Failed fresh native"):
        verify_receiver_negative(directory, row, positive)


@pytest.mark.parametrize("corruption", ["empty", "omitted", "removed_core"])
def test_source_manifest_requires_complete_scientific_closure(tmp_path, corruption):
    declared = frozen_sources(tmp_path)
    if corruption == "empty":
        declared = {}
    elif corruption == "omitted":
        (tmp_path / "source/unlisted.py").write_text("unlisted scientific code")
    else:
        relative = "src/mriseqbench/evaluation/s1_koma.jl"
        declared.pop(relative)
        (tmp_path / "source" / relative).unlink()
    (tmp_path / "source_manifest.json").write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="Incomplete scientific source manifest"):
        source_manifest(tmp_path)
