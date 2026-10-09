from pathlib import Path

from mriseqbench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli(tmp_path, capsys):
    assert main(["--root", str(ROOT), "validate"]) == 0
    assert main(["--root", str(ROOT), "describe", "gre_t1w", "--physics", "b0_smooth"]) == 0
    assert "gradient_hz_per_m" in capsys.readouterr().out
    assert main(["schema", "--out", str(tmp_path / "schemas")]) == 0
    assert (tmp_path / "schemas/tasks.schema.json").exists()


def test_describe_hardware_override(capsys):
    assert (
        main(["--root", str(ROOT), "describe", "gre_t1w", "--hardware", "low_field"])
        == 0
    )
    text = capsys.readouterr().out
    assert "field_strength_T: 0.55" in text
    assert "max_gradient_mT_per_m: 26" in text
