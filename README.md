# Object Detection Training Template

Template repository for object detection projects built on the external
`object-detector-trainer` package:

- https://github.com/starwit/object-detector-trainer

This repo contains project wiring only:

- `train.py`: project CLI wrapper used by DVC
- `dvc.yaml`: bootstrap, prepare, train, and evaluate stages
- `params.yaml`: dataset, model, training, and evaluation parameters
- `tools/export_baseline.py`: promote a trained run as the comparison baseline
- `tests/`: project-level contract tests

Do not commit raw images, local run directories, downloaded weights, or trained
weights directly to Git. Use DVC for shared datasets and promoted baselines.

## Create A Project

1. Create a new repository from this template.
2. Update `pyproject.toml`:
   - `tool.poetry.name`
   - `tool.poetry.description`
   - `object-detector-trainer` tag, if the project should use another trainer release
3. Update `params.yaml`:
   - `data.dataset_name`
   - `data.custom_classes` and `data.class_mapping`, or enable COCO classes
   - `train.model`, `train.image_size`, `train.epochs`, and `train.batch_size`
   - optional per-backend defaults under `models_defaults`
4. Install dependencies:

   ```bash
   poetry install
   ```

5. Create `raw_data/` folders and add images/labels there. See
   [Data Layout](#data-layout).
6. Run an experiment:

   ```bash
   poetry run dvc exp run -n "my-experiment"
   ```

If you initialize a repository manually instead of using a hosted template,
commit the template files before the first `dvc exp run`. DVC experiments require
a Git repository with at least one commit.

## Data Layout

Each new project creates `raw_data/` locally. The directory is Git-ignored so
DVC can track it through `raw_data.dvc`:

```bash
mkdir -p raw_data/train/my-source/images raw_data/train/my-source/labels
# Optional explicit test set:
mkdir -p raw_data/test/my-test-source/images raw_data/test/my-test-source/labels
```

Use one or more source folders:

```text
raw_data/train/<source>/images/*
raw_data/train/<source>/labels/*.txt
raw_data/test/<source>/images/*
raw_data/test/<source>/labels/*.txt
```

`raw_data/test/` is optional. If it contains images, it becomes the test set and
`prepare.test_split` is ignored. If it is absent, set `prepare.test_split > 0`
for an end-to-end run.

Labels must use YOLO rows:

```text
class_id x_center y_center width height
```

Missing or empty label files are treated as background images. A source-level
`data.yaml` with a `names` section is recommended when source label IDs may not
match the class order in `params.yaml`.

## Local Workflow

Run the full DVC pipeline:

```bash
poetry run dvc exp run -n "my-experiment"
```

Override parameters for one run:

```bash
poetry run dvc exp run -n "small-batch" -S train.batch_size=4
```

Run without a promoted baseline:

```bash
poetry run dvc exp run -n "local" \
  -S evaluation.baseline_weights_path=models/local_only/best.pt
```

Review and apply experiments:

```bash
poetry run dvc exp show --sort-by metrics/fitness --sort-order desc
poetry run dvc exp apply <experiment-name>
```

Some backends download pretrained assets into `models/pretrained/`.

## Shared DVC Workflow

A DVC remote is optional for local-only work. Add one when datasets, promoted
baselines, or run artifacts need to be shared across machines.

```bash
poetry run dvc remote add -d <remote-name> <remote-url>
poetry run dvc remote modify <remote-name> verify true
```

Add remote credentials only if your remote needs them, for example:

```bash
poetry run dvc remote modify <remote-name> user <remote-user>
```

Keep `.dvc/config` empty in the template until the real project remote is known.
Do not commit placeholder remote URLs.

Track and publish data:

```bash
poetry run dvc add raw_data
git add raw_data.dvc
git commit -m "Update training data"
poetry run dvc push
git push
```

Reproduce a shared project:

```bash
poetry install
poetry run dvc pull raw_data
poetry run dvc pull models/current_best/best.pt
poetry run dvc exp run
```

`verify true` makes DVC recalculate hashes during fetch/pull so remote object
corruption is caught against the hashes in DVC metadata. It is not SSH host-key
verification.

## Baseline Promotion

Promote a trained run to the project baseline:

```bash
poetry run python tools/export_baseline.py --run-dir runs/<experiment-name>
poetry run dvc add models/current_best/best.pt
poetry run dvc push
git add models/current_best/best.pt.dvc models/current_best/metadata.yaml
```

For RTMDet baselines, also commit the copied config:

```bash
git add models/current_best/model_config.py
```

## Naming

- `tool.poetry.name`: Python/Poetry project name
- `data.dataset_name`: prepared dataset path under `datasets/<dataset_name>/`
- `dvc exp run -n <name>`: DVC experiment name used in experiment history and run folders

These names are independent. DVC remote names and URLs only configure storage.

## Testing

Fast project contract tests:

```bash
poetry run pytest
```

DVC wiring tests with lightweight backend stubs:

```bash
poetry run pytest --heavy
```

Real backend one-epoch tests live in the `object-detector-trainer` repository.

## RTMDet/OpenMMLab

This template enables the trainer's RTMDet extras. RTMDet requires full `mmcv`
ops (`mmcv`, not `mmcv-lite`). Recommended Python: 3.11.

```bash
poetry env use 3.11
poetry install
poetry run mim install mmcv==2.1.0
poetry run python -c "import mmcv._ext"
```
