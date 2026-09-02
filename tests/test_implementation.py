from market_jepa.implementation import (
    changed_implementation_files,
    implementation_manifest,
    manifest_sha256,
)


def test_implementation_manifest_has_frozen_scope() -> None:
    manifest = implementation_manifest()
    files = set(manifest["files"])
    assert "market_jepa/model/encoders.py" in files
    assert "train_market_jepa.py" in files
    assert "benchmark_market_jepa.py" in files
    assert "configs/market_jepa_v0.yaml" in files
    assert "pyproject.toml" in files
    assert "uv.lock" in files
    assert "README.md" not in files
    assert not any(name.startswith("tests/") for name in files)
    assert len(manifest_sha256(manifest)) == 64


def test_changed_implementation_files_reports_added_removed_and_modified() -> None:
    expected = {"files": {"same.py": "a", "modified.py": "old", "removed.py": "x"}}
    current = {"files": {"same.py": "a", "modified.py": "new", "added.py": "y"}}
    assert changed_implementation_files(expected, current) == [
        "added.py",
        "modified.py",
        "removed.py",
    ]
