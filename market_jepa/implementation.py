from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "train_market_jepa.py",
    "export_market_latents.py",
    "eval_market_jepa.py",
    "benchmark_market_jepa.py",
    "pyproject.toml",
    "uv.lock",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_paths(root: Path = ROOT) -> list[Path]:
    paths = [*root.glob("market_jepa/**/*.py"), *root.glob("configs/*.yaml")]
    paths.extend(root / name for name in ENTRYPOINTS)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"implementation files missing: {[str(path) for path in missing]}")
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def implementation_manifest(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "files": {
            path.relative_to(root).as_posix(): _file_sha256(path)
            for path in implementation_paths(root)
        },
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def changed_implementation_files(
    expected: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    expected_files = expected.get("files", {})
    current_files = current.get("files", {})
    names = set(expected_files) | set(current_files)
    return sorted(name for name in names if expected_files.get(name) != current_files.get(name))
