from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.export_baseline import COMMIT_MESSAGE, export_baseline, record_baseline


def write_run(run_dir: Path, weights: bytes) -> None:
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/best.pt").write_bytes(weights)
    (run_dir / "metadata.yaml").write_text("model_backend: yolo\nimage_size: 640\n")


def test_export_replaces_read_only_baseline_files(tmp_path: Path) -> None:
    baseline = tmp_path / "models/current_best"
    baseline.mkdir(parents=True)
    for name in ("best.pt", "metadata.yaml"):
        (baseline / name).write_text("old")
        (baseline / name).chmod(0o444)
    write_run(tmp_path / "run", b"new-weights")

    export_baseline(tmp_path / "run", baseline)

    assert (baseline / "best.pt").read_bytes() == b"new-weights"
    assert (baseline / "metadata.yaml").read_text() == "model_backend: yolo\nimage_size: 640\n"


def test_unfinished_run_changes_nothing(tmp_path: Path) -> None:
    baseline = tmp_path / "models/current_best"
    baseline.mkdir(parents=True)
    (baseline / "best.pt").write_text("old")
    write_run(tmp_path / "run", b"")

    with pytest.raises(FileNotFoundError, match="finished training run"):
        export_baseline(tmp_path / "run", baseline)
    assert (baseline / "best.pt").read_text() == "old"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_baseline_is_tracked_and_committed_without_other_changes(tmp_path: Path, monkeypatch) -> None:
    git("init", "-q", cwd=tmp_path)
    git("config", "user.email", "test@example.invalid", cwd=tmp_path)
    git("config", "user.name", "Test", cwd=tmp_path)
    subprocess.run([sys.executable, "-m", "dvc", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "params.yaml").write_text("seed: 42\n")
    (tmp_path / "dvc.lock").write_text("schema: 2.0\n")
    git("add", ".", cwd=tmp_path)
    git("commit", "-qm", "Project", cwd=tmp_path)
    (tmp_path / "unrelated.txt").write_text("staged, but not part of the baseline")
    git("add", "unrelated.txt", cwd=tmp_path)
    (tmp_path / "params.yaml").write_text("seed: 7\n")  # A pending edit, not part of the baseline.
    (tmp_path / "dvc.lock").write_text("schema: 2.0\nstages: {}\n")
    git("add", "dvc.lock", cwd=tmp_path)
    write_run(tmp_path / "run", b"weights")
    monkeypatch.chdir(tmp_path)

    export_baseline(Path("run"))
    assert record_baseline()

    assert git("log", "-1", "--format=%s", cwd=tmp_path).strip() == COMMIT_MESSAGE
    committed = git("show", "--name-only", "--format=", "HEAD", cwd=tmp_path).split()
    assert committed == ["models/current_best/best.pt.dvc", "models/current_best/metadata.yaml"]
    assert git("diff", "--cached", "--name-only", cwd=tmp_path).split() == ["dvc.lock", "unrelated.txt"]
    assert not record_baseline()  # Exporting the same run again changes nothing.
    assert git("diff", "--name-only", cwd=tmp_path).split() == ["params.yaml"]


@pytest.mark.parametrize("metadata", ["not: a model\n", "[yolo, 640]\n", "model_backend: [\n"])
def test_invalid_metadata_keeps_the_old_baseline(tmp_path: Path, metadata: str) -> None:
    baseline = tmp_path / "models/current_best"
    baseline.mkdir(parents=True)
    (baseline / "best.pt").write_text("old")
    write_run(tmp_path / "run", b"new-weights")
    (tmp_path / "run/metadata.yaml").write_text(metadata)

    with pytest.raises((ValueError, yaml.YAMLError)):
        export_baseline(tmp_path / "run", baseline)
    assert (baseline / "best.pt").read_text() == "old"
