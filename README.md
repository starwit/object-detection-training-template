# Object Detection Training Template

This repository is a starting point for an object detection training project built on the external `object-detector-trainer` package.

It contains project-level wiring only:

- `train.py`: thin CLI wrapper used by DVC stages
- `dvc.yaml`: split bootstrap, prepare, train, and evaluate stages
- `params.yaml`: example dataset, model, and evaluation configuration
- `tools/export_baseline.py`: helper for promoting a trained run as the project baseline
- `tests/`: contract tests for the project wiring

The reusable trainer implementation lives in:

- https://github.com/starwit/object-detector-trainer

## What To Customize

For a new project, update these files first:

- `pyproject.toml`: project name, description, and trainer dependency tag if needed
- `params.yaml`: dataset/project slug, output classes, class mapping, model selection, training defaults, and folder subsets
- `.dvc/config`: add your real project DVC remote, with `verify = true`, if you want to share datasets, baselines, and run artifacts
- `README.md`: replace this template guide with project-specific reproduction notes once the project exists

Do not commit local model weights, run directories, or raw images directly to Git. Use DVC for shared data and promoted baselines.

## Project Naming

There are three separate names:

- `pyproject.toml` `tool.poetry.name`: the Python/Poetry project name. Change it for the new repository, but DVC stages do not use it to find data.
- `params.yaml` `data.dataset_name`: the training dataset/project slug. This is required by the pipeline and controls the prepared dataset path under `datasets/<dataset_name>/`.
- `dvc exp run -n <name>`: the DVC experiment/run name. Use it to make run folders and `dvc exp show` entries readable. It is independent from `data.dataset_name`.

The DVC remote name and URL are not project names. They only configure shared storage for DVC-tracked files.

## Remote Requirement

A DVC remote is not required for local-only development. You can install dependencies, put your own data under `raw_data/`, run tests, and run local DVC experiments without configuring shared storage.

A DVC remote is required when the project should share versioned datasets, promoted baseline weights, or run artifacts across machines. Set one up when other machines should be able to run `dvc pull`, when you want to publish `raw_data.dvc` or promoted baseline weights, or when large files should be recoverable without committing them to Git. Commands such as `dvc pull`, `dvc push`, and reproducing someone else's DVC-tracked data require a configured remote.

## Creating A Project From This Template

No bootstrap script is required for the minimal setup. Create a new repository from the template, then make the first project-specific edits before your first project-specific commit:

1. Update `pyproject.toml`:
   - `tool.poetry.name`
   - `tool.poetry.description`
2. Update `params.yaml`:
   - `data.dataset_name`
   - `data.custom_classes` and `data.class_mapping`, or switch to COCO mode
   - `train.model`, image size, epochs, and batch size
3. Install dependencies before running the DVC commands below:

   ```bash
   poetry install
   ```

4. Add a DVC remote only if the project should share data/artifacts. Skip this step for a purely local project.

   `dvc remote add -d` writes the remote into `.dvc/config` and makes it the default remote used by `dvc pull` and `dvc push`:

   ```bash
   poetry run dvc remote add -d <remote-name> <remote-url>
   poetry run dvc remote modify <remote-name> user <remote-user>  # only if needed
   poetry run dvc remote modify <remote-name> verify true
   ```

   For a Hetzner Storage Box SSH remote in the same style as the waste-detection project:

   ```bash
   poetry run dvc remote add -d dvc-hetzner ssh://<storagebox-host>/<project-slug>
   poetry run dvc remote modify dvc-hetzner user <storagebox-user>
   poetry run dvc remote modify dvc-hetzner verify true
   ```

   Example shape:

   ```bash
   poetry run dvc remote add -d dvc-hetzner ssh://u123456-sub1.your-storagebox.de/my-detection-project
   poetry run dvc remote modify dvc-hetzner user u123456-sub1
   poetry run dvc remote modify dvc-hetzner verify true
   ```

   The committed project config should then look like this, with real project values and no placeholders:

   ```ini
   [core]
   remote = dvc-hetzner

   ['remote "dvc-hetzner"']
   url = ssh://u123456-sub1.your-storagebox.de/my-detection-project
   user = u123456-sub1
   verify = true
   ```

   `verify true` makes DVC recalculate file hashes during `dvc pull` and `dvc fetch`. Use it for shared project remotes so corrupted or unexpectedly changed remote objects are caught against the hashes stored in `.dvc` files and `dvc.lock`. It can slow pulls/fetches, but that is usually the right tradeoff for shared datasets and promoted model weights. It is not SSH host-key verification.

   Keep the template `.dvc/config` empty until the real remote URL is known. Do not commit a placeholder default remote: DVC treats it as a real remote and commands such as `dvc pull`, `dvc push`, and `dvc fetch` will use it.

   Review the generated `.dvc/config` before committing it.

5. Put data under `raw_data/`, run the pipeline, and review generated artifacts.
6. Commit only source/config files plus DVC pointer files, not raw data or weights.

If you initialize a repository manually instead of creating it from a hosted template, commit the template files before the first `dvc exp run`. DVC experiments require a Git repository with at least one commit.

## Setup

```bash
poetry install
```

`dvc exp run` creates experiment commits. If Git complains about missing user name/email, configure them globally or locally:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

## Data Layout

Put raw data under `raw_data/`:

- `raw_data/train/<source>/images/*`
- `raw_data/train/<source>/labels/*`
- `raw_data/test/<source>/images/*`
- `raw_data/test/<source>/labels/*`

`params.yaml` documents the supported raw source layouts and how label class IDs are resolved. In short: a source-level `data.yaml` lets the trainer remap labels by class name; without it, label IDs must already match the configured class order.

If `raw_data/test/` contains images, it is used as the test set and `prepare.test_split` is ignored. If `raw_data/test/` is absent, set `prepare.test_split > 0` for an end-to-end run.

## Running Locally

For a local-only experiment without a promoted baseline:

```bash
poetry run dvc exp run -n local \
  -S evaluation.baseline_weights_path=models/local_only/best.pt \
  -S prepare.test_split=0.2
```

Some backends may download upstream pretrained assets into `models/pretrained/`.

## Shared DVC Workflow

After adding a DVC remote for your project, track datasets like this:

```bash
poetry run dvc add raw_data
git add raw_data.dvc
git commit -m "Update training data"
poetry run dvc push
git push
```

To reproduce a shared project:

```bash
poetry install
poetry run dvc pull raw_data
poetry run dvc pull models/current_best/best.pt
poetry run dvc exp run
```

## Training Workflow

Project parameters live in `params.yaml`.

Useful configuration rules:

- `train.image_size`, `train.epochs`, and `train.batch_size` are shared defaults.
- `train.finetune.weights` is an optional fine-tune input path and defaults to `models/finetune/best.pt`.
- `models_defaults.<backend>` applies per-backend defaults to every model of that backend.
- `models.<key>` overrides both shared and backend defaults.
- `models.<key>.asset_id` identifies the pretrained asset to bootstrap for the selected backend.

Run the full DVC experiment pipeline:

```bash
poetry run dvc exp run -n "my-experiment"
```

Override parameters for a one-off experiment:

```bash
poetry run dvc exp run -n "test-smaller-batch" -S train.batch_size=4
```

Review experiment results:

```bash
poetry run dvc exp show --sort-by metrics/fitness --sort-order desc
```

Promote an experiment into your workspace:

```bash
poetry run dvc exp apply <experiment-name>
```

## Promoting A Baseline

To make a trained run the project comparison baseline:

```bash
poetry run python tools/export_baseline.py --run-dir runs/<experiment_name>
```

Then track the promoted artifacts:

```bash
poetry run dvc add models/current_best/best.pt
poetry run dvc push
git add models/current_best/best.pt.dvc
git add models/current_best/metadata.yaml
```

If the exported run is RTMDet-based, also commit the copied config:

```bash
git add models/current_best/model_config.py
```

## Testing

Project-level tests in this repo:

- `poetry run pytest`
  Fast contract tests for config resolution, wrapper behavior, and offline pipeline contracts.
- `poetry run pytest --heavy`
  Slower DVC experiment wiring tests that use lightweight backend stubs.

Real backend-specific one-epoch tests live in the `object-detector-trainer` repository.

## RTMDet/OpenMMLab Note

This template installs `object-detector-trainer` with RTMDet extras enabled.

RTMDet requires full `mmcv` ops (`mmcv`, not `mmcv-lite`). Two supported install methods:

```bash
poetry run mim install mmcv==2.1.0
poetry run python -c "import mmcv._ext"
```

or:

```bash
MMCV_WITH_OPS=1 poetry run pip install "mmcv==2.1.0" --no-binary=mmcv --no-build-isolation --no-cache-dir
poetry run python -c "import mmcv._ext"
```

Recommended Python for RTMDet/OpenMMLab: 3.11 (`poetry env use 3.11`).
