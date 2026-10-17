"""Build a small, deterministic Kaggle source bundle without reading market data.

This exporter only visits the explicitly allowed source directories. It never
walks the repository root, data caches, saved research, or local environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "astra-kaggle.zip"
NOTEBOOK_NAME = "Astra_Research.ipynb"
MANIFEST_NAME = "EXPORT-MANIFEST.json"
MANIFEST_SCHEMA = "astra.kaggle.export.v1"
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024
ROOT_FILES = frozenset({"README.md", "AGENTS.md", "pyproject.toml", ".gitignore"})
KAGGLE_FILES = frozenset(
    {
        f"kaggle/{NOTEBOOK_NAME}",
        "kaggle/requirements.txt",
        "kaggle/provenance/audit.json",
        "kaggle/provenance/audit.md",
    }
)
REQUIRED_FILES = (
    ROOT_FILES
    | KAGGLE_FILES
    | {
        "astra/__init__.py",
        "scripts/kaggle_run.py",
        "scripts/export_kaggle.py",
        "docs/kaggle.md",
    }
)
WEB_DIRECTORIES = frozenset({"sandbox_web", "training_arena_web", "progress_desk_web"})
WEB_SUFFIXES = frozenset({".html", ".css", ".js", ".svg", ".png", ".ico", ".woff2"})
FIXTURE_SUFFIXES = frozenset({".json", ".csv", ".txt", ".yaml", ".yml"})
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "data",
        "datasets",
        "runs",
        "results",
        "output",
        "exports",
        "cache",
        "hf-cache",
        "kaggle-output",
    }
)


def _forbidden_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith(".env")
        or lowered in {"kaggle.json", "id_rsa", "id_ed25519"}
        or "credential" in lowered
        or "secret" in lowered
        or Path(lowered).suffix in {".pem", ".key", ".p12", ".pfx"}
    )


def allowed_path(name: str) -> bool:
    """Validate a portable archive name against the complete file allowlist."""
    if not name or "\\" in name or ":" in name:
        return False
    if any(part in {"", ".", ".."} for part in name.split("/")):
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or any(_forbidden_name(part) for part in path.parts):
        return False
    if any(part.lower() in SKIP_DIRECTORIES for part in path.parts):
        return False
    if name in ROOT_FILES or name in KAGGLE_FILES:
        return True
    if path.parts[0] == "astra":
        return path.suffix == ".py" or (
            len(path.parts) == 3
            and path.parts[1] in WEB_DIRECTORIES
            and path.suffix in WEB_SUFFIXES
        )
    if path.parts[0] == "scripts":
        return len(path.parts) == 2 and path.suffix == ".py"
    if path.parts[0] == "docs":
        return len(path.parts) == 2 and path.suffix == ".md"
    if path.parts[0] == "tests":
        return path.suffix == ".py" or (
            len(path.parts) >= 3
            and path.parts[1] == "fixtures"
            and path.suffix in FIXTURE_SUFFIXES
        )
    return False


def _reject_link(path: Path) -> os.stat_result:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or (
        getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"Refusing symlink or reparse point: {path}")
    return details


def _checked_file(root: Path, relative: str) -> Path:
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        _reject_link(path)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Not a regular project file: {relative}")
    return path


def collect_files(project_root: Path) -> dict[str, Path]:
    """Collect only source paths; excluded root directories are never listed."""
    _reject_link(project_root)
    root = project_root.resolve()
    files: dict[str, Path] = {}
    for name in sorted(ROOT_FILES | KAGGLE_FILES):
        files[name] = _checked_file(root, name)

    def visit(directory: Path, *, recursive: bool) -> None:
        _reject_link(directory)
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            details = _reject_link(path)
            if stat.S_ISDIR(details.st_mode):
                if (
                    recursive
                    and entry.name.lower() not in SKIP_DIRECTORIES
                    and not entry.name.startswith(".")
                    and not _forbidden_name(entry.name)
                ):
                    visit(path, recursive=True)
                continue
            name = path.relative_to(root).as_posix()
            if allowed_path(name):
                files[name] = _checked_file(root, name)

    for name in ("astra", "scripts", "tests", "docs"):
        visit(root / name, recursive=name in {"astra", "tests"})
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"Missing required export files: {', '.join(sorted(missing))}")
    return dict(sorted(files.items()))


def validate_notebook(payload: bytes) -> None:
    notebook = json.loads(payload)
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Kaggle notebook must be a valid nbformat 4 notebook")
    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            raise TypeError(f"Invalid notebook cell {index}")
        if cell.get("cell_type") == "code":
            code_cells += 1
            if cell.get("execution_count") is not None or cell.get("outputs") != []:
                raise ValueError(f"Notebook cell {index} contains execution history")
    if not code_cells:
        raise ValueError("Kaggle notebook must contain code cells")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def verify_archive(archive: Path) -> dict:
    """Read back every small member and verify the exact manifest inventory."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or MANIFEST_NAME not in names:
            raise ValueError("Export has duplicate names or no manifest")
        if sum(member.file_size for member in members) > MAX_TOTAL_BYTES:
            raise ValueError("Export exceeds the total source size limit")
        for member in members:
            if member.filename != MANIFEST_NAME and not allowed_path(member.filename):
                raise ValueError(f"Forbidden archive path: {member.filename}")
            if (
                member.is_dir()
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > MAX_FILE_BYTES
            ):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        manifest = json.loads(bundle.read(MANIFEST_NAME))
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("Unrecognized export manifest schema")
        rows = manifest.get("files", [])
        inventory = {row["path"]: row for row in rows}
        if len(inventory) != len(rows) or set(inventory) != set(names) - {
            MANIFEST_NAME
        }:
            raise ValueError("Archive inventory differs from manifest")
        if REQUIRED_FILES - inventory.keys():
            raise ValueError("Archive is missing required project files")
        for name, row in inventory.items():
            payload = bundle.read(name)
            if len(payload) != row["size_bytes"] or _sha256(payload) != row["sha256"]:
                raise ValueError(f"Export hash or size mismatch: {name}")
        validate_notebook(bundle.read(f"kaggle/{NOTEBOOK_NAME}"))
    return manifest


def _prepare_output(directory: Path) -> Path:
    # Refuse existing symlink ancestors before creating any output directory.
    for ancestor in (directory, *directory.parents):
        if ancestor.exists() or ancestor.is_symlink():
            _reject_link(ancestor)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def _temporary_path(directory: Path, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(
        prefix=".astra-export-", suffix=suffix, dir=directory
    )
    os.close(handle)
    return Path(name)


def build_export(
    project_root: Path = PROJECT_ROOT, output_dir: Path | None = None
) -> dict:
    files = collect_files(project_root)
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    for name, path in files.items():
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Source file exceeds the small export limit: {name}")
        payload = path.read_bytes()
        total_bytes += len(payload)
        if len(payload) > MAX_FILE_BYTES or total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("Source bundle exceeds the small export limit")
        payloads[name] = payload
    notebook_payload = payloads[f"kaggle/{NOTEBOOK_NAME}"]
    validate_notebook(notebook_payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "Source-only Kaggle research; no market cache or local results included",
        "files": [
            {"path": name, "size_bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(payloads.items())
        ],
    }
    payloads[MANIFEST_NAME] = _json_bytes(manifest)
    output = _prepare_output(output_dir or project_root / "exports")
    archive = output / ARCHIVE_NAME
    notebook = output / NOTEBOOK_NAME
    for target in (archive, notebook):
        if target.exists() or target.is_symlink():
            _reject_link(target)
    temporary_zip = _temporary_path(output, ".zip")
    temporary_notebook = _temporary_path(output, ".ipynb")
    try:
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for name, payload in sorted(payloads.items()):
                bundle.writestr(_zip_info(name), payload, compresslevel=9)
        verify_archive(temporary_zip)
        temporary_notebook.write_bytes(notebook_payload)
        validate_notebook(temporary_notebook.read_bytes())
        os.replace(temporary_zip, archive)
        os.replace(temporary_notebook, notebook)
    finally:
        temporary_zip.unlink(missing_ok=True)
        temporary_notebook.unlink(missing_ok=True)
    return {
        "archive": str(archive),
        "notebook": str(notebook),
        "file_count": len(files),
        "source_bytes": total_bytes,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for the ZIP and standalone notebook (default: project exports/)",
    )
    args = parser.parse_args()
    print(json.dumps(build_export(output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
