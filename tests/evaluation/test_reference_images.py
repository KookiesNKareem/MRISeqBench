"""Figure selection follows the final verdict, rather than refinement history."""

import json

import numpy as np
import pytest

from scripts.reference_images import collect


def stage(directory, name, values):
    frames = []
    for i, value in enumerate(values):
        path = directory / "evaluation" / name / f"frame-{i}" / "images.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, physical=np.full((2, 2), value), case_reference=np.ones((2, 2)))
        frames.append({"artifacts": {"images": str(path)}})
    return frames


@pytest.mark.parametrize("recovered", [False, True])
def test_only_three_final_frames_are_rendered(tmp_path, recovered):
    initial = stage(tmp_path, "initial", [1, 2, 3])
    final = stage(tmp_path, "mapping-stage-1", [10, 11, 12])
    (tmp_path / "verdict.json").write_text(
        json.dumps({"frames": initial if recovered else final})
    )
    if recovered:
        (tmp_path / "recovered-verdict.json").write_text(json.dumps({"frames": final}))
    images = collect(tmp_path)
    assert len(images) == 3
    assert [actual[0, 0] for actual, _ in images] == [10, 11, 12]


def test_finished_verdict_without_images_does_not_render_stale_stage(tmp_path):
    stage(tmp_path, "initial", [1])
    (tmp_path / "verdict.json").write_text(json.dumps({"status": "failed"}))
    assert collect(tmp_path) == []
