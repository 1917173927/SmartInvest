from __future__ import annotations

from pathlib import Path

import pytest

from stock_analysis.files import atomic_write_text


def test_atomic_write_text_replaces_complete_file(tmp_path) -> None:
    target = tmp_path / "reports" / "latest.md"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")

    result = atomic_write_text(target, "完整的新报告\n")

    assert result == target
    assert target.read_text(encoding="utf-8") == "完整的新报告\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_text_preserves_old_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    target = tmp_path / "latest.md"
    target.write_text("old", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("stock_analysis.files.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
