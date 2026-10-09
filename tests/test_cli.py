from pathlib import Path

from mriseqbench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli(tmp_path, capsys):
    assert main(["--root", str(ROOT), "validate"]) == 0
    assert main(["--root", str(ROOT), "describe", "gre_t1w", "--physics", "b0_b1"]) == 0
    assert "gradient_hz_per_m" in capsys.readouterr().out
    assert main(["schema", "--out", str(tmp_path / "schemas")]) == 0
    assert (tmp_path / "schemas/tasks.schema.json").exists()
