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
    case = catalog.cases(catalog.get("suites", "smoke"))[0]
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
