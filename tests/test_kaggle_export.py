"""Small synthetic source bundles; no project data or cloud execution is used."""

import json
import zipfile
from pathlib import Path

import pytest

from scripts import export_kaggle as exporter


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "project"
    for name in exporter.REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source fixture\n", encoding="utf-8")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('synthetic')\n"],
            }
        ],
    }
    (root / "kaggle" / exporter.NOTEBOOK_NAME).write_text(json.dumps(notebook))
    (root / "tests").mkdir()
    return root


def test_export_is_complete_deterministic_and_leaves_data_unvisited(
    source, tmp_path, monkeypatch
):
    web = source / "astra/sandbox_web/index.html"
    web.parent.mkdir()
    web.write_text("<html>synthetic</html>")
    (source / "tests/test_synthetic.py").write_text("assert True\n")
    for relative in (
        ".env",
        ".env.example",
        "kaggle.json",
        "local.sqlite3",
        "data/large.arrow",
        ".venv/hidden.py",
        "scripts/credentials.py",
        "tests/saved.csv",
        "astra/__pycache__/old.pyc",
    ):
        path = source / relative
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("EXCLUDED SENTINEL")
    visited = []
    original_scandir = exporter.os.scandir

    def record_scan(directory):
        visited.append(Path(directory).relative_to(source).as_posix())
        return original_scandir(directory)

    monkeypatch.setattr(exporter.os, "scandir", record_scan)
    first = exporter.build_export(source, tmp_path / "out-one")
    second = exporter.build_export(source, tmp_path / "out-two")
    assert "data" not in visited and ".venv" not in visited
    assert Path(first["archive"]).read_bytes() == Path(second["archive"]).read_bytes()
    with zipfile.ZipFile(first["archive"]) as bundle:
        assert "astra/sandbox_web/index.html" in bundle.namelist()
        assert "tests/test_synthetic.py" in bundle.namelist()
        assert all(
            b"EXCLUDED SENTINEL" not in bundle.read(name) for name in bundle.namelist()
        )
        assert (
            bundle.read(f"kaggle/{exporter.NOTEBOOK_NAME}")
            == Path(first["notebook"]).read_bytes()
        )
    manifest = exporter.verify_archive(Path(first["archive"]))
    assert len(manifest["files"]) == first["file_count"]


@pytest.mark.parametrize(
    "name",
    [
        "../README.md",
        "/README.md",
        "C:/README.md",
        "astra\\risk.py",
        "astra/../data/test.py",
        "scripts/.env.py",
        "scripts/kaggle.json",
        "astra/secrets.py",
        "data/source.csv",
        "kaggle/provenance/other.json",
    ],
)
def test_archive_paths_are_allowlisted(name):
    assert not exporter.allowed_path(name)


@pytest.mark.parametrize(
    "history",
    [
        {"execution_count": 1},
        {"outputs": [{"output_type": "stream", "text": "saved output"}]},
    ],
)
def test_executed_notebook_is_rejected_before_replacing_exports(
    source, tmp_path, history
):
    output = tmp_path / "output"
    output.mkdir()
    previous = output / exporter.ARCHIVE_NAME
    previous.write_bytes(b"existing export")
    path = source / "kaggle" / exporter.NOTEBOOK_NAME
    notebook = json.loads(path.read_text())
    notebook["cells"][0].update(history)
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="execution history"):
        exporter.build_export(source, output)
    assert previous.read_bytes() == b"existing export"


def test_symlink_inside_source_is_rejected(source, tmp_path):
    external = tmp_path / "outside.py"
    external.write_text("sensitive = True\n")
    link = source / "astra/external.py"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("The operating system does not allow test symlink creation")
    with pytest.raises(ValueError, match="symlink"):
        exporter.build_export(source, tmp_path / "output")


def test_hash_verification_rejects_tampered_member(source, tmp_path):
    result = exporter.build_export(source, tmp_path / "output")
    tampered = tmp_path / "tampered.zip"
    with (
        zipfile.ZipFile(result["archive"]) as original,
        zipfile.ZipFile(tampered, "w") as rewritten,
    ):
        for member in original.infolist():
            contents = original.read(member.filename)
            if member.filename == "README.md":
                contents = b"tampered"
            rewritten.writestr(member, contents)
    with pytest.raises(ValueError, match="hash or size mismatch"):
        exporter.verify_archive(tampered)


def test_oversized_source_is_rejected(source, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "MAX_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="small export limit"):
        exporter.build_export(source, tmp_path / "output")
