"""Render the human S3 qualification record from the sealed machine evidence."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    data = json.loads(args.record.read_text())
    assert data["qualified"]
    rows = data["records"]
    groups = defaultdict(list)
    for row in rows:
        if row["role"] == "ideal_positive":
            groups[row["case"]["task"]["id"].removesuffix("_research")].append(row)
    text = [
        "# S3 calibration and qualification\n",
        (
            "The `s3@1` sampling judge is calibrated for its documented benchmark image policy. "
            "All nine families and both prompt modes have nominal-grid qualification. Completed "
            "submissions return `passed` or `failed`; interrupted computations preserve progress "
            "without assigning an image verdict.\n"
        ),
        (
            "The quality cutoffs were fixed before qualification: ideal complex NRMSE <=0.30, "
            "relative signal norm >=0.50 and outside-object energy <=0.25. P2/P3 use the "
            "case-image criteria in [S3 evaluation](S3_EVALUATION.md). Cutoffs were not relaxed "
            "to make disturbed acquisitions pass. These are benchmark quality tolerances.\n"
        ),
        "## Qualification controls\n",
        "| Role | Controls | Accepted | Rejected |",
        "|---|---:|---:|---:|",
    ]
    for role, counts in data["roles"].items():
        text.append(
            f"| {role.replace('_', ' ')} | {counts['controls']} | {counts['passed']} | {counts['failed']} |"
        )
    text += [
        (
            "\nThe 77 ideal positives cover all 25 compatible family/object pairs in both prompt "
            "modes, plus every family on the other three hardware profiles. The 75 physics "
            "characterizations cover every compatible registered profile for every family. "
            "Nineteen structural negatives and nine actual receiver-phase-corrupted submissions "
            "test rejection. All acquisition matrices are 32 by 32 or 32 by 32 by 32.\n"
        ),
        (
            "Qualification uses spatial seed 106458 and reversed view ordering, after design "
            "controls with seed 1729 and forward ordering. The anatomy and control author remain "
            "shared; this is independent numerical replication and a changed acquisition order, "
            "not a claim of unseen anatomy or a separate acquisition-author population. "
            "This is a 180-control qualification, not an exhaustive run of all 800 S3 suite cases "
            "or an agent-performance experiment.\n"
        ),
        "## Ideal-control numerical margins\n",
        "| Family | Controls | Maximum complex NRMSE | Largest upper endpoint | Minimum signal lower endpoint | Maximum outside-energy upper endpoint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam, values in sorted(groups.items()):
        metric = max(v["metrics"]["ideal_complex_image_nrmse"] for v in values)
        upper = max(
            v["numerical_intervals"]["ideal"]["complex_image_nrmse"][1] for v in values
        )
        signal = min(
            v["numerical_intervals"]["ideal"]["signal_fraction"][0] for v in values
        )
        outside = max(
            v["numerical_intervals"]["ideal"]["outside_object_energy_fraction"][1]
            for v in values
        )
        text.append(
            f"| {fam} | {len(values)} | {metric:.5f} | {upper:.5f} | {signal:.5f} | {outside:.5f} |"
        )
    text += [
        (
            "\nScores and approximate 99% numerical intervals use independently scrambled, "
            "primitive-stratified Sobol integration and unbiased complex cross-products. A "
            "negative squared-error estimate may display as zero NRMSE; its untruncated squared "
            "statistic and uncertainty remain in the records. A verdict can resolve by an "
            "interval wholly inside/outside a quality bound before the tighter score-precision "
            "target is reached. This does not make the displayed noisy mean image an exact "
            "integral.\n"
        ),
        "## Physics characterization\n",
        "| Family | Profile | Accepted |",
        "|---|---|---|",
    ]
    for row in sorted(
        (r for r in rows if r["role"] == "physics_characterization"),
        key=lambda r: (r["case"]["task"]["id"], r["case"]["physics"]["id"]),
    ):
        text.append(
            f"| {row['case']['task']['id']} | {row['case']['physics']['id']} | {'yes' if row['candidate_pass'] else 'no'} |"
        )
    text += [
        (
            "\nA waveform can be legal and still fail imaging under B0 or motion. These "
            "failures remain image-quality failures, with resolved scores; they are not "
            "discarded calibration inputs or infrastructure errors.\n"
        ),
        "## Derived image stress assays\n",
        "| Corruption | Passing sources | Rejected | Accepted | Unresolved |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in sorted(data["stress_controls"].items()):
        text.append(
            f"| {name} | {counts['controls']} | {counts['rejected']} | {counts['accepted']} | {counts['unresolved']} |"
        )
    reuse = data.get("native_replay_reuse")
    if reuse:
        migrations = reuse.get("migrations", [reuse])
        reused_rows = [r for m in migrations for r in m["records"]]
        text.append(
            "\nThe corrected qualification reuses "
            f"{sum(len(r['signals']) for r in reused_rows)} SHA-verified native signals "
            f"from {len({r['control'] for r in reused_rows})} earlier controls with identical submitted inputs "
            "and unchanged dynamics for those effects. Verdicts, reconstructed images, "
            "references and uncertainty intervals are recomputed. The native helper also "
            "now retains discontinuous motion keyframe knots; physical rigid-keyframe "
            "signals computed before that correction are excluded from reuse. The "
            "machine record retains every source/destination signal hash and the dependency "
            "equivalence checks.\n"
        )
    cross_reuse = data.get("cross_control_ideal_reuse")
    if cross_reuse:
        text.append(
            f"\nAn additional {cross_reuse['native_zero_effect_signals']} zero-effect native diagnostic signals "
            f"are shared across {cross_reuse['controls']} physics controls with identical waveform bytes, hardware, "
            "object, geometry, actual quadrature nodes/materials/weights/properties and integration steps. "
            "The audit recomputes every native-input/cache-key equivalence and raw-array hash. "
            "The optional [reuse tool](../tests/fixtures/reuse_s3_ideal_replays.py) records the frozen writer and all receipts. "
            "Disturbed-physics signals and all scores, references, intervals and verdicts are evaluated for each control.\n"
        )
    text += [
        (
            "\nDerived controls use SHA-verified retained complex replicates. Receiver rotation "
            "and blanking are also linear raw-signal transformations; displacement and ROI "
            "dropout are image-domain stress controls. Contrast erasure is applied separately "
            "to the full-ensemble and leave-one-out mean images, with exact equal ROI magnitude "
            "means asserted before jackknife scoring. Only native-passing sources contribute "
            "to rejection totals. These assays are distinct from the nine freshly encoded "
            "receiver-phase-corrupted Pulseq submissions.\n"
        ),
        "## Design corrections and independent checks\n",
        (
            "A reconstructed-image timestep assay found a 0.97% raw error amplified "
            "to 1.21% relative image error in reduced-FOV tissue controls. S3 now requires "
            "both raw convergence and a reconstructed timestep difference <=1% of the "
            "independent target norm. A nominal-grid regression reproduces the coarse "
            "failure and verifies finer RF integration. Earlier raw-only verdicts remain "
            "in the superseded qualification-v3 records.\n"
        ),
        (
            "Hardware qualification exposed a circular-mean branch-cut bug: tiny gradient "
            "round-off selected the negative half-sample representative on high-performance "
            "hardware, shifting the integer readout origin and falsely rejecting all nine "
            "families. The correction chooses the positive representative within the published "
            "lattice tolerance while retaining physical coordinates. All 36 family/hardware "
            "waveform controls pass the regression. The superseded qualification-v2 receipts "
            "remain unchanged; every case is judged again from corrected frozen source.\n"
        ),
        (
            "The initial reduced-FOV control passed small-grid smoke tests but failed nominal "
            "imaging (complex NRMSE 0.54650). Symmetric refocusing crushers and changing "
            "two-axis spoilers removed long-T2 ghosts but left a 0.3876 error. A sharper "
            "90-degree selective pulse alone left 0.3608. The final control uses a "
            "20-degree selective excitation, initial saturation, the crushers/spoilers and "
            "a sharply bounded slab. Its corrected design estimate is 0.07904, with upper "
            "endpoint 0.21182. Earlier controls and source snapshots remain retained.\n"
        ),
        (
            "The reduced-FOV ideal target now integrates the registered primitives clipped "
            "to the prescribed inner slab, on a complete independent reduced-FOV Fourier "
            "grid, then embeds that image at the prescribed spatial resolution. Cropping a "
            "full-FOV band-limited image was not the same operation. Cross-section integration "
            "is analytic in x and Gauss-Legendre in y; volume, independent Sobol Fourier "
            "checks and quadrature refinement verify its accuracy. The reference does not "
            "copy measured RF selectivity or candidate signals.\n"
        ),
        (
            "Receive channels share streamed native per-spin BlochDict ADC blocks. Separate "
            "coil-density native Bloch replays agree across B0, motion, combined and dynamic-PD "
            "tests. A nominal 3D CAIPIRINHA comparison measured relative raw/image differences "
            "of 6.6e-14/7.6e-14. Uniform-receive replay retains native summed Bloch. "
            "The fixed receive model and linear reconstruction policy are documented in "
            "[S3 evaluation](S3_EVALUATION.md); this calibration does not claim GRAPPA kernel "
            "training, nonlinear compressed-sensing reconstruction or a scanner coil model.\n"
        ),
        (
            "The audit regenerated all submitted sequence bytes from the frozen author and "
            "independently read their Pulseq ADC trajectories. Every sequence, case, result "
            "and result artifact hash was verified, including finite raw/image arrays and "
            "the exported coordinates/timestamps. Source snapshots and package manifests are "
            "verified. Completed per-chunk checkpoints are pruned only after record sealing; "
            "actual coil signals, image replicates, temporal differences and provenance remain.\n"
        ),
        "## Reproduction and evidence\n",
        (
            "The [machine-readable record](S3_CALIBRATION.json) includes every case, outcome, "
            "score, interval, source/sequence/result/artifact hash and independent export check. "
            "Initial qualification sampling is 4096 per replicate with eight scrambles, using "
            "the same precision and acceptance rules as the 32768 release default. Adaptive "
            "refinement continues until a numerical verdict resolves.\n"
        ),
        (
            "```sh\nuv run python tests/fixtures/calibrate_s3.py runs/s3-design --phase pilot --seed 1729 --initial-samples 4096 --workers 2 --batch-size 5\n"
            "uv run python tests/fixtures/calibrate_s3.py runs/s3-qualification --phase qualification --ordering reverse --seed 106458 --initial-samples 4096 --workers 2 --batch-size 12\n"
            "uv run python tests/fixtures/score_s3_calibration.py runs/s3-qualification --output runs/s3-corruptions.json\n"
            "uv run python tests/fixtures/audit_s3_calibration.py runs/s3-qualification --corruptions runs/s3-corruptions.json --output docs/S3_CALIBRATION.json\n"
            "uv run python tests/fixtures/report_s3_calibration.py docs/S3_CALIBRATION.json --output docs/S3_CALIBRATION.md\n```\n"
        ),
        (
            "Use `--resume` with the same destination to verify completed receipts and "
            "continue missing work from the frozen source. A fresh qualification must use a "
            "new directory. The recorded qualification directory is "
            f"`{data['qualification_directory']}`. Design snapshots are under "
            "`runs/s3-calibration-design-v1-20261012` through `v4-20261012`; the prepared "
            "qualification-v1 plan was superseded before simulation.\n"
        ),
        (
            "The recorded corrected run was prepared with `--prepare-only`, then "
            "`tests/fixtures/migrate_s3_replays.py` verified and imported eligible native "
            "signals from qualification-v2 before `--resume`. A fresh reproduction does "
            "not require those caches. The migration rejects changed dynamics/input "
            "dependencies and never reuses a submitted verdict.\n"
        ),
    ]
    args.output.write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
