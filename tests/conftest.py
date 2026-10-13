import json
import subprocess
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def submission(tmp_path_factory):
    root = tmp_path_factory.mktemp("submission")
    catalog = Catalog(ROOT)
    # This shared backend-protocol fixture must fit every active scanner and
    # the stricter hardware used by isolated analytic regression controls.
    case = catalog.resolve(
        catalog.get("tasks", "pulseq_smoke"), hardware_ref="low_field"
    )
    (root / "case.json").write_text(json.dumps(case))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/fixtures/make_smoke_submission.py"),
            "--case",
            str(root / "case.json"),
            "--output",
            str(root / "sequence.seq"),
        ],
        check=True,
    )
    return root / "sequence.seq"
