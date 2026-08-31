from __future__ import annotations

from typer.testing import CliRunner

from stock_analysis.cli import app

runner = CliRunner()


def test_doctor_runs_without_optional_dependencies(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    (tmp_path / "stock-analysis.toml").write_text("[risk]\ncash_floor=0.05\n", encoding="utf-8")
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "SQLite" in result.stdout
