"""Real DVC workflow: fixed CVAT subsets, reload, baseline lifecycle, fine-tuning and caching."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.smoke_data import create_sources

pytestmark = pytest.mark.heavy

MODELS = {
    "yolo": {
        "backend": "yolo", "asset_id": "yolo26n.pt", "batch_size": 1, "effective_batch_size": 2,
        "extra_train_kwargs": {"warmup_epochs": 0, "workers": 0},
    },
    "rfdetr": {
        "backend": "rfdetr", "variant": "nano", "asset_id": "rf-detr-nano.pth",
        "resolution": 128, "batch_size": 1, "effective_batch_size": 2,
        "extra_train_kwargs": {"num_workers": 0},
    },
}


@pytest.mark.parametrize("backend", MODELS)
def test_dvc_pipeline(tmp_path: Path, backend: str) -> None:
    project = Path(__file__).resolve().parents[1]
    for name in ("dvc.yaml", ".dvcignore", ".gitignore", "pyproject.toml", "poetry.lock"):
        shutil.copy2(project / name, tmp_path / name)
    shutil.copytree(project / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    for folder in ("models/current_best", "models/finetune"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / ".gitkeep").touch()
    create_sources(tmp_path)
    source = tmp_path / "raw_data/camera"
    params = yaml.safe_load((project / "params.yaml").read_text())
    params["data"].update(custom_classes=["person", "object"], class_mapping={})
    params["prepare"] = {"val_split": 0, "test_split": 0}
    params["train"].update(model="smoke", image_size=128, epochs=1)
    params["models"]["smoke"] = MODELS[backend]
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    dataset_name = params["data"]["dataset_name"]
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["DVC_NO_ANALYTICS"] = "1"

    def run(*args):
        result = subprocess.run(args, cwd=tmp_path, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    def dvc(*args):
        return run(sys.executable, "-m", "dvc", *args)

    def stale_stages() -> set[str]:
        return set(json.loads(dvc("status", "--json"))) - {"raw_data.dvc"}

    def membership():
        with (tmp_path / "datasets" / dataset_name / "manifest.csv").open() as handle:
            return sorted((row["source_image"], row["split"]) for row in csv.DictReader(handle))

    run("git", "init", "-q")
    run("git", "config", "user.email", "template-test@example.invalid")
    run("git", "config", "user.name", "Template test")
    dvc("init")
    dvc("add", "raw_data")
    run("git", "add", ".")
    run("git", "commit", "-qm", "Test project")
    dvc("exp", "run", "-n", "smoke")

    weights = tmp_path / "runs/current/train/weights/best.pt"
    trained_at = weights.stat().st_mtime_ns
    prepared = tmp_path / "datasets" / dataset_name
    assert [len(list((prepared / split / "images").iterdir())) for split in ("train", "val", "test")] == [5, 2, 3]
    assert (prepared / "train/labels/background__scene_negatives.txt").read_text() == ""
    assert (prepared / "test/labels/background__scene_camera.txt").read_text() == ""
    assert set(json.loads((tmp_path / "runs/current/metrics.json").read_text())["per_class"]) == {"person", "object"}
    original_membership = membership()
    assert run("git", "status", "--porcelain").splitlines() == ["?? dvc.lock"]
    assert "no baseline at models/current_best/best.pt yet" in (tmp_path / "runs/current/eval/report.txt").read_text()
    assert stale_stages() == {"evaluate_model"}

    dvc("exp", "run", "-n", "unchanged")
    assert weights.stat().st_mtime_ns == trained_at
    run(sys.executable, "tools/export_baseline.py")
    assert stale_stages() == {"evaluate_model"}
    dvc("repro")
    assert weights.stat().st_mtime_ns == trained_at
    results = json.loads((tmp_path / "runs/current/eval/results.json").read_text())
    assert list(results) == ["baseline", "current"]

    (tmp_path / "models/current_best/best.pt").unlink()
    dvc("repro")
    assert "not downloaded" in (tmp_path / "runs/current/eval/report.txt").read_text()
    dvc("checkout", "models/current_best/best.pt.dvc")
    dvc("repro")
    assert list(json.loads((tmp_path / "runs/current/eval/results.json").read_text())) == ["baseline", "current"]
    assert weights.stat().st_mtime_ns == trained_at

    shutil.copy2(weights, tmp_path / "models/finetune/best.pt")
    finetune = {**MODELS[backend], "init_weights": "models/finetune/best.pt", "epochs": 1}
    if backend == "yolo":
        finetune.update(
            freeze_backbone=True,
            extra_train_kwargs={"optimizer": "AdamW", "lr0": 0.0002, "warmup_epochs": 0, "workers": 0},
        )
    else:
        finetune.update(lr=0.00001, extra_train_kwargs={"num_workers": 0, "lr_encoder": 0.000015})
    params["models"]["finetuned"] = finetune
    params["train"]["model"] = "finetuned"
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    dvc("exp", "run", "-n", "finetuned")
    metadata = yaml.safe_load((tmp_path / "runs/current/train/metadata.yaml").read_text())
    assert metadata["class_names"] == {0: "person", 1: "object"}
    assert metadata["model_options"]["init_weights"] == "models/finetune/best.pt"
    if backend == "yolo":
        native = yaml.safe_load((tmp_path / "runs/current/train/args.yaml").read_text())
        assert (native["optimizer"], native["lr0"]) == ("AdamW", 0.0002)
    else:
        native = json.loads((tmp_path / "runs/current/train/training_config.json").read_text())["train_config"]
        assert (native["lr"], native["lr_encoder"]) == (0.00001, 0.000015)
    trained_at = weights.stat().st_mtime_ns

    label = source / "labels/Train/000000000030.txt"
    rows = label.read_text().splitlines()
    rows[0] = "0 0.4 0.5 0.2 0.2"
    label.write_text("\n".join(rows) + "\n")
    assert "prepare_data" in stale_stages()
    dvc("repro", "prepare_data")
    updated = prepared / "train/labels/000000000030__scene_camera.txt"
    assert updated.read_text().splitlines()[0] == "1 0.400000 0.500000 0.200000 0.200000"
    assert len(updated.read_text().splitlines()) == len(rows)
    assert membership() == original_membership
    assert weights.stat().st_mtime_ns == trained_at
