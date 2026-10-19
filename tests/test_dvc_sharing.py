"""Two independent project copies sharing data, a baseline and DVC experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.heavy


def test_two_copies_share_and_restore_experiments(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    for name in ("dvc.yaml", ".dvcignore", ".gitignore", "pyproject.toml", "poetry.lock"):
        shutil.copy2(project / name, first / name)
    shutil.copytree(project / "tools", first / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    for folder in ("models/current_best", "models/finetune"):
        (first / folder).mkdir(parents=True)
        (first / folder / ".gitkeep").touch()

    for split, count in (("train", 4), ("val", 1), ("test", 1)):
        for index in range(count):
            source = first / "raw_data" / split / "camera"
            (source / "images").mkdir(parents=True, exist_ok=True)
            (source / "labels").mkdir(parents=True, exist_ok=True)
            image = np.random.default_rng(index + len(split) * 100).integers(
                0, 256, (128, 128, 3), dtype=np.uint8,
            )
            assert cv2.imwrite(str(source / "images" / f"{index}.jpg"), image)
            (source / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    params = yaml.safe_load((project / "params.yaml").read_text())
    params["data"].update(dataset_name="sharing", custom_classes=["waste"])
    params["train"].update(model="yolov8n", image_size=128, epochs=1)
    params["models"]["yolov8n"] = {
        "backend": "yolo", "asset_id": "yolov8n.pt", "batch_size": 1,
        "effective_batch_size": 2, "extra_train_kwargs": {"workers": 0, "optimizer": "auto"},
    }
    (first / "params.yaml").write_text(yaml.safe_dump(params))
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["DVC_NO_ANALYTICS"] = "1"

    def run(where: Path, *args: str) -> str:
        result = subprocess.run(args, cwd=where, env=env, text=True, capture_output=True, timeout=300, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    def dvc(where: Path, *args: str) -> str:
        return run(where, sys.executable, "-m", "dvc", *args)

    def identify(where: Path) -> None:
        run(where, "git", "config", "user.name", "Sharing test")
        run(where, "git", "config", "user.email", "sharing@example.invalid")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def labels(where: Path) -> dict[str, str]:
        return {
            str(path.relative_to(where / "datasets/sharing")): path.read_text()
            for path in (where / "datasets/sharing").glob("*/labels/*.txt")
        }

    def assert_hard_links(where: Path) -> None:
        prepared = next((where / "datasets/sharing/train/images").glob("0__scene_camera.jpg"))
        source = where / "raw_data/train/camera/images/0.jpg"
        assert os.path.samefile(source, prepared)

    origin = tmp_path / "origin.git"
    run(tmp_path, "git", "init", "--bare", "-q", str(origin))
    run(first, "git", "init", "-q")
    identify(first)
    dvc(first, "init", "-q")
    dvc(first, "remote", "add", "-d", "sharing", str(tmp_path / "storage"))
    run(first, "git", "remote", "add", "origin", str(origin))
    dvc(first, "add", "raw_data")
    run(first, "git", "add", ".")
    run(first, "git", "commit", "-qm", "Record project and first data version")
    branch = run(first, "git", "branch", "--show-current").strip()

    dvc(first, "exp", "run", "-n", "data-v1")
    first_labels = labels(first)
    first_weights = digest(first / "runs/current/train/weights/best.pt")
    assert_hard_links(first)
    run(first, sys.executable, "tools/export_baseline.py")
    # The exporter leaves run settings/results for the separate README recording step.
    run(first, "git", "add", "params.yaml", "dvc.lock")
    run(first, "git", "commit", "-qm", "Record training settings and results")
    run(first, "git", "push", "-qu", "origin", branch)
    dvc(first, "push")
    dvc(first, "exp", "push", "origin", "data-v1")

    run(tmp_path, "git", "clone", "-q", str(origin), str(second))
    identify(second)
    assert not (second / "raw_data").exists()
    assert not (second / ".dvc/cache").exists()
    assert not (second / "models/current_best/best.pt").exists()
    # README join workflow, followed by a real training change on the second machine.
    dvc(second, "pull", "raw_data.dvc")
    dvc(second, "pull", "models/current_best/best.pt.dvc")
    assert digest(second / "models/current_best/best.pt") == first_weights
    assert dvc(first, "cache", "dir").strip() != dvc(second, "cache", "dir").strip()
    dvc(second, "exp", "run", "-n", "colleague-run", "-S", "train.epochs=2")
    result = json.loads((second / "runs/current/eval/results.json").read_text())
    assert list(result) == ["baseline", "current"]
    assert labels(second) == first_labels
    assert_hard_links(second)

    # The first copy changes one annotation and shares an experiment for that data version.
    (first / "raw_data/train/camera/labels/0.txt").write_text("0 0.4 0.5 0.5 0.5\n")
    dvc(first, "add", "raw_data")
    run(first, "git", "add", "raw_data.dvc")
    run(first, "git", "commit", "-qm", "Record second data version")
    dvc(first, "exp", "run", "-n", "data-v2")
    second_labels = labels(first)
    second_weights = digest(first / "runs/current/train/weights/best.pt")
    assert first_labels != second_labels
    dvc(first, "exp", "push", "origin", "data-v2")
    dvc(second, "exp", "pull", "origin", "data-v1")
    dvc(second, "exp", "pull", "origin", "data-v2")

    # Both copies restore models/data, then rebuild the uncached prepared dataset.
    for where in (first, second):
        for name, expected_labels, expected_weights, raw_label in (
            ("data-v1", first_labels, first_weights, "0 0.5 0.5 0.5 0.5\n"),
            ("data-v2", second_labels, second_weights, "0 0.4 0.5 0.5 0.5\n"),
            ("data-v1", first_labels, first_weights, "0 0.5 0.5 0.5 0.5\n"),
        ):
            dvc(where, "exp", "apply", name)
            assert (where / "raw_data/train/camera/labels/0.txt").read_text() == raw_label
            weights = where / "runs/current/train/weights/best.pt"
            assert digest(weights) == expected_weights
            trained_at = weights.stat().st_mtime_ns
            dvc(where, "repro")
            assert labels(where) == expected_labels
            assert digest(weights) == expected_weights
            assert weights.stat().st_mtime_ns == trained_at
            assert_hard_links(where)

    # Interrupt real training after it has written an epoch, then restart normally.
    log_path = tmp_path / "interrupted.log"
    results_path = second / "runs/current/train/results.csv"
    old_mtime = results_path.stat().st_mtime_ns
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "dvc", "exp", "run", "-n", "interrupted", "-S", "train.epochs=10000"],
            cwd=second, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and process.poll() is None:
                args_path = second / "runs/current/train/args.yaml"
                if results_path.exists() and args_path.exists():
                    args = yaml.safe_load(args_path.read_text()) or {}
                    if (args.get("epochs") == 10000 and results_path.stat().st_mtime_ns != old_mtime
                            and len(results_path.read_text().splitlines()) >= 2):
                        break
                time.sleep(0.2)
            else:
                pytest.fail("Training did not reach an epoch:\n" + log_path.read_text())
            assert process.poll() is None
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=30)
            assert process.returncode != 0
            interruption = log_path.read_text()
            assert "RuntimeError:" not in interruption
            assert "KeyboardInterrupt" in interruption or "exited with -2" in interruption
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
    dvc(second, "exp", "run", "-n", "restarted", "-S", "train.epochs=1")
    metadata = yaml.safe_load((second / "runs/current/train/metadata.yaml").read_text())
    assert metadata["epochs"] == 1
    assert (second / "runs/current/train/weights/best.pt").stat().st_size > 0
    assert list(json.loads((second / "runs/current/eval/results.json").read_text())) == ["baseline", "current"]
    assert_hard_links(second)
