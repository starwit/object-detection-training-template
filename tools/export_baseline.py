"""Make a trained run the baseline that every new run is compared against.

    python tools/export_baseline.py                              # runs/current/train
    python tools/export_baseline.py --run-dir runs/my-run/train

Copies the run's weights/best.pt and metadata.yaml to models/current_best/,
tracks the weights with DVC and commits the baseline files in Git.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

BASELINE_DIR = Path("models/current_best")
COMMIT_MESSAGE = "Select new baseline"


def export_baseline(run_dir: Path, baseline_dir: Path = BASELINE_DIR) -> None:
    weights = run_dir / "weights" / "best.pt"
    metadata = run_dir / "metadata.yaml"
    for path in (weights, metadata):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{path} is missing or empty. Is {run_dir} a finished training run?")
    content = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not {"model_backend", "image_size"} <= content.keys():
        raise ValueError(f"{metadata} is not a valid model metadata.yaml; the baseline was not changed.")

    baseline_dir.mkdir(parents=True, exist_ok=True)
    for source in (weights, metadata):
        target = baseline_dir / source.name
        # DVC may have checked the old file out as a read-only link: replace it, do not write into it.
        target.unlink(missing_ok=True)
        shutil.copy2(source, target)


def record_baseline(baseline_dir: Path = BASELINE_DIR) -> bool:
    """Track the weights with DVC and commit the baseline files. Returns False if nothing changed."""
    run(sys.executable, "-m", "dvc", "add", str(baseline_dir / "best.pt"))
    files = [str(baseline_dir / "best.pt.dvc"), str(baseline_dir / "metadata.yaml")]
    run("git", "add", *files)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *files]).returncode == 0:
        return False
    # Commit only these files, even if other changes are staged.
    run("git", "commit", "-q", "-m", COMMIT_MESSAGE, "--", *files)
    return True


def run(*command: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stdout}{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a trained run the comparison baseline.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/current/train"),
                        help="Training run folder with weights/best.pt and metadata.yaml.")
    args = parser.parse_args()
    export_baseline(args.run_dir)
    if record_baseline():
        print(f'{args.run_dir} is the new baseline, committed as "{COMMIT_MESSAGE}".')
    else:
        print(f"{args.run_dir} is already the baseline; nothing to commit.")
    print("If shared storage is configured, upload it with: poetry run dvc push")


if __name__ == "__main__":
    main()
