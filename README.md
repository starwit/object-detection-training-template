# Object Detection Training Template

Train and evaluate an object detection model using your own annotated images.
You describe your classes and training settings in `params.yaml`; one command
then prepares the data, trains a model and evaluates it. Models and results
stay inside your project folder.

The guide has three parts:

- **[Train your first model](#train-your-first-model)**: the steps from an
  empty project to a trained, evaluated model.
- **Background and next steps**: [your data](#your-data) in more detail,
  [classes](#classes), and how to [evaluate and improve](#evaluate-and-improve)
  a model, [share a project](#share-a-project) and
  [use the trained model](#use-the-trained-model).
- **[Reference](#reference)**: project folders and troubleshooting.

## Train your first model

### 1. Set up the project

You need Linux, Python 3.11 or newer, Git and
[Poetry](https://python-poetry.org), which installs the project's Python
packages. An NVIDIA GPU is recommended; training on the CPU works but is much
slower.

Create your project repository with GitHub's **Use this template** button
and clone it. Replace `YOUR_PROJECT_URL` with your repository's address;
`my-project` is an example folder name:

```bash
git clone YOUR_PROJECT_URL my-project
```

Install everything (this can take several minutes). Run this and all later
commands from your project folder:

```bash
cd my-project
poetry install
```

This installs the
[trainer](https://github.com/starwit/object-detector-trainer), which does the
actual work, in the version set in `pyproject.toml`. It also installs
**DVC** (Data Version Control). DVC keeps track of large
files such as images and models, which do not belong in Git, and runs the
training steps defined in `dvc.yaml`.

DVC records every training run in Git. If Git does not know your name yet,
set it for this project:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

### 2. Add your data

Training needs images with boxes drawn around the objects, each box labelled
with a class name such as `waste` or `cigarette`. The images are divided into
three sets:

| Set | Purpose |
|---|---|
| Train | Images the model learns from |
| Validation | Checked during training to keep the best version of the model |
| Test | Kept aside for the final evaluation, so the score reflects images the model has never seen |

The simplest way is to keep everything in one **CVAT** project, including
which task belongs to which set, and to give the training project one export
of it:

1. In CVAT, open each task and set its **Subset** to **Train**,
   **Validation** or **Test**. Put different recordings in different sets;
   [Which sets you supply](#which-sets-you-supply) explains why.
2. On the **project** page, choose **Actions → Export dataset**, select
   **Ultralytics YOLO Detection 1.0**, enable **Save images** and download
   the ZIP.
3. Unpack the ZIP into a folder under `raw_data` in your project:

```bash
mkdir -p raw_data
unzip /path/to/my-cvat-project.zip -d raw_data/my-cvat-project
```

Export only when all tasks are fully annotated: a frame without boxes is used
as an example of "nothing to detect here" (see [Get data from
CVAT](#get-data-from-cvat)).

Data from other tools, several CVAT tasks, or sets you want to decide
yourself are described in [Your data](#your-data).

### 3. Configure `params.yaml`

Open `params.yaml` and edit the `data` and `train` sections, for example:

```yaml
data:
  dataset_name: my-dataset
  custom_classes:
    - waste
    - cigarette
  class_mapping: {}
  ignore_classes: []

train:
  model: yolo26n
  image_size: 640
  epochs: 100
  device: auto
```

| Setting | Meaning |
|---|---|
| `dataset_name` | Name of the prepared dataset folder under `datasets/` |
| `custom_classes` | Your class names, as used in the annotations |
| `model` | Which model to train, see below |
| `image_size` | Images are resized to this many pixels for training; larger finds smaller objects but is slower |
| `epochs` | How many times training goes through all training images |
| `device` | `auto` uses the first GPU, otherwise the CPU. Set `cpu` or a specific GPU such as `cuda:1` to choose yourself |

`class_mapping` and `ignore_classes` stay empty for now; [Classes](#classes)
explains when you need them.

`train.model` selects one of the entries in the `models` section further down
in the same file:

| `train.model` | Model |
|---|---|
| `yolo26n` | YOLO26 Nano: small and fast, a good first model |
| `yolo26m` | YOLO26 Medium: larger, often more accurate, slower |
| `rfdetr-nano` | RF-DETR Nano |
| `rfdetr-medium` | RF-DETR Medium |
| `finetune` | Continue from a model you trained before, see [Fine-tune an existing model](#fine-tune-an-existing-model) |

Each entry under `models` has its own `batch_size`: the number of images
processed at once. Lower it if the GPU runs out of memory.

YOLO and RF-DETR are two different families of detection models; see
[Compare models and settings](#compare-models-and-settings) for how to
choose. Leave the other sections of `params.yaml` as they are.

### 4. Record the data version

Let DVC record the exact data you train on, and Git the small file that
points to it:

```bash
poetry run dvc add raw_data
git add raw_data.dvc .gitignore params.yaml
git commit -m "Add training data and class configuration"
```

The images stay out of Git, in DVC's local storage inside the project. You
need no server for this; [Share a project](#share-a-project) explains how to
upload data and models for colleagues later.

### 5. Start training

```bash
poetry run dvc exp run -n name-of-first-model
```

DVC runs three stages and records them as an **experiment** called
`name-of-first-model`:

1. **Preparation** builds the training dataset from `raw_data/` and prints the
   number of images and objects per set and class.
2. **Training** downloads general pretrained weights on first use and adapts
   them to your classes. With the settings above it runs for up to 100 epochs
   and keeps the version that does best on the validation set.
3. **Evaluation** measures that model on the test set.

### 6. Find the model and results

| File | Content |
|---|---|
| `runs/current/train/weights/best.pt` | The trained model |
| `runs/current/train/metadata.yaml` | Model type, classes and settings needed to load it |
| `runs/current/eval/report.txt` | Evaluation report |
| `runs/current/eval/predictions/` | Every test image: your annotations on the left, the model's detections on the right |
| `runs/current/train/` | Training logs and plots |

First compare the image counts printed by preparation with what you expect
in each set. Then open `report.txt` and look through a few images in
`predictions/`: they show what the scores mean in practice.
[Read the evaluation report](#read-the-evaluation-report) explains the
numbers.

The report's first line says that there is no **baseline** yet. A baseline is
a model you choose as the reference that later runs are compared with; see
[Select a baseline](#select-a-baseline).

## Your data

### Data format

The trainer reads labels in the **YOLO** format, which CVAT's Ultralytics
YOLO export produces. A source, for example one CVAT task or one folder of
labelled images, looks like this:

```text
my-source/
  images/frame_001.jpg
  labels/frame_001.txt
  data.yaml
```

Each image has a label file with the same name and path, but under `labels/`
and ending in `.txt`. Subfolders are allowed. Images can be `.jpg`, `.jpeg`,
`.png` or `.bmp`.

Each line in a label file is one box:

```text
0 0.512 0.430 0.120 0.080
```

The numbers are the class ID, the box centre (x, y) and the box width and
height. They are fractions of the image width and height, so they lie
between 0 and 1. Polygons from a segmentation export are turned into the
boxes around them.

`data.yaml` says which class each ID stands for:

```yaml
names:
  0: waste
  1: cigarette
```

With this file, classes are matched to `custom_classes` by name (upper or
lower case does not matter). Without it, the ID is the position in
`custom_classes`: `0` is the first class, `1` the second.

An image with an empty or missing label file is an example of "nothing to
detect here". That is useful for empty scenes, but only if the image really
contains none of your classes.

### Get data from CVAT

Step 2 shows the recommended way: one export of the whole CVAT project, in
**Ultralytics YOLO Detection 1.0** format with **Save images**, unpacked into
a folder under `raw_data/`. A few details:

- **Subsets.** The task's **Subset** field decides the set; the task's name
  does not. `Train`, `Validation`/`Valid`/`Val` and `Test` are recognised in
  upper or lower case.
- **Unfinished tasks.** The export contains every frame of every task. A
  frame nobody has annotated yet has no boxes, so it is used as "nothing to
  detect" and teaches the model to ignore the objects on it. Export when the
  tasks are finished, or keep unfinished work in a separate CVAT project.
- **Older formats.** CVAT's older **YOLO 1.1** format (with `obj_train_data/`)
  and XML or COCO exports are not read; export again as Ultralytics YOLO.

See [CVAT's export documentation](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
for the format itself.

### Assemble the sets in folders

Instead of setting subsets in CVAT, you can decide the sets with folders.
This is useful when combining several CVAT tasks, data from other tools, or
when you want to see results per recording (see
[Read the evaluation report](#read-the-evaluation-report)).

Create a folder per set and put each source into one of them. For CVAT,
export each **task** separately (Ultralytics YOLO Detection 1.0, Save images)
and unpack it into its set:

```bash
mkdir -p raw_data/train raw_data/val raw_data/test
unzip /path/to/recording-1.zip -d raw_data/train/recording-1
unzip /path/to/recording-2.zip -d raw_data/val/recording-2
cp -r /path/to/other-source raw_data/test/
```

```text
raw_data/
  train/
    recording-1/
      images/
      labels/
      data.yaml
  val/
    recording-2/
  test/
    other-source/
```

Here the folder decides the set, even if the task export itself says
`Train`. A set folder can hold several source folders. If a set
holds just one source, its `images/` and `labels/` can also go directly into
the set folder, for example `raw_data/train/images/`.

A whole-project export with several subsets cannot go into a set folder;
it belongs directly under `raw_data/`.

### Combine a CVAT project export with additional sources

You can keep an unpacked whole-project export directly under `raw_data/` and add other
labelled data under `train/`, `val/` or `test/`. For example, extra training
images that are not in the CVAT project:

```text
raw_data/
  my-cvat-project/
  train/
    additional-source/
      images/
      labels/
      data.yaml
```

The images of each set are combined. Do not add exports of tasks that are
already part of the project export.

### Which sets you supply

You can supply all three sets, or let preparation take the missing ones from
the training images:

| Sets you supply | What preparation does |
|---|---|
| Train, validation, test | Uses them as they are |
| Train and test | Keeps test; takes 10% of train for validation |
| Train and validation | Keeps validation; takes 10% of train for test |
| Train only | Takes 10% each for validation and test |

The percentages are `prepare.val_split` and `prepare.test_split` in
`params.yaml`; they only apply to sets you do not supply. This works the same
for CVAT subsets and set folders, and for several sources combined.

The automatic split picks single images at random. For video, neighbouring
frames are almost identical, so the test set would contain near-copies of
training images and the score would look better than the model really is on
a new recording. **Choose the validation and test sets yourself, from
separate recordings, days, locations or cameras**, and include the classes
and conditions that matter to you.

**Keep the test set fixed** once you compare models. An automatic split can
change when you add images, and a changed test set measures something else:
scores from before are then no longer comparable. Adding training data is
fine. If you do change the test set, compare models on the same new version;
the pipeline evaluates the baseline on it too. Make sure the baseline was not
trained on recordings you now use for testing.

### Duplicates

Preparation removes duplicate images before it builds the dataset: exact
copies, and images that look practically identical (for example the same
frame exported twice, or two consecutive frames of a still scene). Of each
group it keeps one image, preferring the test set, then validation, then
train, so a test image never also appears in training. The log shows how many
images were removed from each set; your files in `raw_data/` are not changed.

This only catches near-identical images. Different frames of the same
recording are not detected, which is why the sets should come from separate
recordings.

### Add or update data

After a change in CVAT, export again and replace the old folder:

```bash
rm -rf raw_data/my-cvat-project
unzip /path/to/new-export.zip -d raw_data/my-cvat-project
```

With set folders, replace the changed task folders the same way.

Then record the new data version and train:

```bash
poetry run dvc add raw_data
git add raw_data.dvc
git commit -m "Update annotated dataset"
poetry run dvc exp run -n updated-data
```

## Classes

### Class names and order

`custom_classes` in `params.yaml` lists the classes in your annotations.
Every class that occurs in a source's `data.yaml` must either be listed there
or be ignored (see below); preparation stops otherwise, so that no class is
dropped by accident.

The order matters in two cases: for sources without `data.yaml`, where label
ID `0` means the first class; and for comparisons, where a baseline is only
compared if it has the same classes in the same order.

### Merge classes

Merging trains several annotated classes as one. This is useful when your
application does not need a distinction the annotations make, or when a class
has too few examples to be learned on its own. For example, the annotations
distinguish `car` and `truck` but you only need `vehicle`, or `bottle` has
40 examples and is really a kind of `waste`. The model then learns one class
from all of these examples.

List the original classes in `custom_classes` and map them to the merged
class in `class_mapping`:

```yaml
data:
  custom_classes: [waste, cigarette, bottle]
  class_mapping:
    waste: [waste, bottle]
```

The model learns `waste` and `cigarette`. Your annotations stay unchanged, so
you can separate the classes again later.

### Ignore a class

Ignoring removes a class's boxes before training. Use it for labels the model
should not learn, for example helper labels or a class that is only partly
annotated:

```yaml
data:
  custom_classes: [waste, cigarette]
  ignore_classes: [twig]
```

The images stay in the dataset, so their twigs are now unlabelled and count
as "nothing to detect". That is fine if the model should not react to them;
if they are hard to tell apart from your real classes, remove those images
instead. Ignoring needs the source's `data.yaml`, because the class is
identified by name.

### Change classes later

To add a class, annotate examples and add its name to `custom_classes`. For
sources without `data.yaml`, add it at the end so existing IDs keep their
meaning. Then record the data and run a new experiment.

After changing the classes, the baseline no longer matches. The report says
so and evaluates the new model on its own; select a new baseline once you
have a good model.

## Evaluate and improve

### Read the evaluation report

`runs/current/eval/report.txt` contains these scores, overall and per class:

| Score | Meaning |
|---|---|
| `precision` | Share of the model's detections that were correct |
| `recall` | Share of the annotated objects the model found |
| `map50` | Detection quality averaged over classes, counting a box as correct if it overlaps the annotated box by at least 50% |
| `map` | The same with several overlap requirements up to 95%, so it also rewards precise boxes |
| `fitness` | One combined number for ranking models: `0.1 × map50 + 0.9 × map` |
| `ms_per_frame` | Processing time per image in milliseconds |

Scores range from 0 to 1; higher is better, and lower processing time is
better. Precision and recall count detections with a confidence of at least
0.25, the same detections shown in `predictions/`.

High precision with low recall means few false detections but many missed
objects; the reverse means many false alarms. No score makes a model "good
enough" by itself: look at the errors that matter for your application, and
check the per-class rows, since an overall score can hide a weak class.

The **Scenes** table shows the score per source, meaning per source
folder in your test set. If the test set consists of several sources, for
example one task export per camera, this shows whether the model works
equally well everywhere. A whole-project export is a single source, so the
table then has one row with the overall score. If you need results per
camera or location, supply the test set as one export per location in
`raw_data/test/` (see [Assemble the sets in folders](#assemble-the-sets-in-folders)).

### Compare models and settings

**YOLO** (from Ultralytics) and **RF-DETR** (from Roboflow) are two families
of detection models. Both find objects and draw boxes around them, use the
same data and are evaluated the same way. Which works better depends on your
data, so compare their scores and processing time.

Change one thing per experiment, so you know what made the difference:

| Change | Command |
|---|---|
| Train longer | `poetry run dvc exp run -n longer -S train.epochs=200` |
| Higher resolution | `poetry run dvc exp run -n high-resolution -S train.image_size=1280` |
| Larger YOLO model | `poetry run dvc exp run -n yolo-medium -S train.model=yolo26m` |
| RF-DETR | `poetry run dvc exp run -n rfdetr-medium -S train.model=rfdetr-medium` |
| RF-DETR with less GPU memory | `poetry run dvc exp run -n rfdetr-small-batch -S train.model=rfdetr-medium -S models.rfdetr-medium.batch_size=1` |

`-S` changes `params.yaml` itself, so later runs keep the new value. You can
also edit `params.yaml` directly before running.

Higher resolution helps with small objects but needs more memory. Larger
models need more resources and are not always better.

DVC skips preparation and training when their inputs have not changed.
Evaluation always runs, so it picks up a newly selected baseline.

Show your experiments and their main scores:

```bash
poetry run dvc exp show -n 10 --drop '.*' --keep 'Experiment|Created|train.model|^map50$|^fitness$'
```

`-n 10` covers the experiments of the last ten Git commits; leave out
`--drop` and `--keep` to see every setting and score. `runs/current/` holds
the latest experiment; to bring back an earlier one, with its settings and
results:

```bash
poetry run dvc exp apply EXPERIMENT_NAME
```

To try a model that is not in the list, add an entry to the `models` section
of `params.yaml`:

```yaml
models:
  yolo11m:
    backend: yolo
    asset_id: yolo11m.pt
  rfdetr-small:
    backend: rfdetr
    variant: small
    asset_id: rf-detr-small.pth
```

`asset_id` names the general pretrained weights downloaded on first use; for
RF-DETR, `variant` must match them. Select the entry with, for example,
`-S train.model=yolo11m`.

### Select a baseline

The **baseline** is the model you choose as the reference. Every later run is
evaluated next to it on the same test set, and
`runs/current/eval/comparisons/` shows both models' detections side by side.
It is stored in `models/current_best/` and only changes when you replace it.

Bring back the chosen experiment if it is not the current one, then make its
model the baseline:

```bash
poetry run dvc exp apply EXPERIMENT_NAME
poetry run python tools/export_baseline.py
```

The tool copies the model to `models/current_best/` and commits it. If shared
storage is configured, also run `poetry run dvc push`.

A missing baseline never stops a run. If the weights are not there (for
example not downloaded), belong to other classes or cannot be read, the
report's first line says why and the new model is evaluated on its own.

### Fine-tune an existing model

Normal training starts from general pretrained weights. **Fine-tuning**
starts from a model you already trained for this project and trains it
further with a lower learning rate, so it learns new data without losing what
it already knows. It can help after adding data with the same classes; compare
it with normal training on the same test set.

Fine-tuning trains on the current training set. Keep the earlier training
data in it, otherwise the model can get worse on what it learned before.

| Folder | Content |
|---|---|
| `models/current_best/` | The baseline, the reference for evaluation |
| `models/finetune/` | The starting model for fine-tuning |

The two are separate on purpose: selecting a new baseline does not change
what fine-tuning starts from.

Copy the starting model, for example the baseline, and select the `finetune`
entry:

```bash
cp models/current_best/best.pt models/finetune/best.pt
poetry run dvc exp run -n finetuned -S train.model=finetune
```

The `finetune` entry is for YOLO models with the same classes in the same
order. It trains 30 epochs with a lower learning rate and keeps the first
layers fixed. The starting file is not shared through Git, so copy it again
on each machine.

Use normal training for a first model, after changing classes or when
switching between YOLO and RF-DETR. Fine-tuning does not resume an
interrupted run.

To fine-tune an RF-DETR model, add an entry whose `variant` matches it:

```yaml
models:
  finetune-rfdetr:
    backend: rfdetr
    variant: nano
    asset_id: rf-detr-nano.pth   # Required, but not used when init_weights is set.
    init_weights: models/finetune/best.pt
    epochs: 30
    lr: 0.00001
    extra_train_kwargs:
      lr_encoder: 0.000015
```

Then select it with `-S train.model=finetune-rfdetr`. To return to normal
training, select a normal entry again, for example `-S train.model=yolo26n`.

## Share a project

Git shares the code and settings; DVC shares the large files (data and
models). DVC needs a storage location for that, called a **remote**.

### Set up shared DVC storage

For a Hetzner Storage Box:

```bash
poetry run dvc remote add -d storage ssh://u123456-sub1.your-storagebox.de/my-detection-project
poetry run dvc remote modify storage user u123456-sub1
poetry run dvc remote modify storage verify true
```

Replace `u123456-sub1` with your storage account and `my-detection-project`
with your project's own directory; the account needs SSH access to it.
`verify true` checks downloaded files for damage.

Or use a mounted network drive or local folder:

```bash
poetry run dvc remote add -d storage /mnt/shared/dvc-storage
```

The setting is saved in `.dvc/config`. Commit it, then upload data and
models:

```bash
git add .dvc/config
git commit -m "Configure DVC storage"
poetry run dvc push
git push
```

Run `poetry run dvc push` and `git push` again after recording new data or
selecting a baseline. To share the settings and results of the current run,
commit them first:

```bash
git add params.yaml dvc.lock
git commit -m "Record training settings and results"
git push
```

Experiments are only stored on your computer. To share one, run:

```bash
poetry run dvc exp push origin EXPERIMENT_NAME
```

To download a shared experiment, run:

```bash
poetry run dvc exp pull origin EXPERIMENT_NAME
```

### Join an existing project

Clone and install as in [Set up the project](#1-set-up-the-project), then
download the data and train:

```bash
poetry run dvc pull raw_data.dvc
poetry run dvc exp run -n my-first-run
```

If the project has a selected baseline (`models/current_best/best.pt.dvc`
exists), download it first to get comparisons:
`poetry run dvc pull models/current_best/best.pt.dvc`.

`poetry run dvc pull` without a file name downloads all data and models of the
current version, including the results of the last recorded run. If you have
no access to the storage, copy the data into `raw_data/` yourself. If
`params.yaml` selects a fine-tuning entry, copy its starting model first (see [Fine-tune an existing model](#fine-tune-an-existing-model)).

## Use the trained model

Keep `best.pt` and `metadata.yaml` together when you copy a model: the
metadata tells the trainer which model type and settings to load.

Load a YOLO model directly with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("runs/current/train/weights/best.pt")
results = model.predict("some_image.jpg", conf=0.25)
results[0].save("some_image_with_boxes.jpg")
```

Load YOLO or RF-DETR through the trainer:

```python
from pathlib import Path

import cv2
from object_detector_trainer import load_model

model = load_model(Path("runs/current/train/weights/best.pt"))
detections = model.predict(cv2.imread("some_image.jpg"), conf=0.25, max_det=100)
for box, score, class_id in zip(detections.xyxy, detections.scores, detections.class_ids):
    print(model.class_names[int(class_id)], round(float(score), 2), box)
```

Save either example as a script and run it with
`poetry run python my_script.py`. `conf` is the minimum confidence: a higher
value gives fewer false detections but also misses more objects.

## Reference

### Project folders

| Path | Content |
|---|---|
| `params.yaml` | Your settings |
| `raw_data/` | Your annotated data |
| `datasets/` | The prepared dataset, rebuilt by every preparation |
| `runs/current/` | Model and results of the latest experiment |
| `models/current_best/` | The selected baseline |
| `models/finetune/` | Starting model for fine-tuning |
| `models/pretrained/` | Downloaded general pretrained weights |
| `dvc.yaml` | The pipeline stages |
| `dvc.lock` | Which data, settings and results belong together, written by DVC |
| `tools/export_baseline.py` | Makes a run the baseline |

### Run without DVC

```bash
poetry run object-detector-trainer --stage all
```

This runs the same three stages without recording an experiment. Each run
gets its own folder `runs/<model>-<date-and-time>/`. To make such a run the
baseline, use
`poetry run python tools/export_baseline.py --run-dir runs/<run>/train`.

All settings are described in the
[trainer README](https://github.com/starwit/object-detector-trainer).

### Troubleshooting

| Message or symptom | What to do |
|---|---|
| Poetry is missing | Install it from https://python-poetry.org. |
| `Please tell me who you are` | Set your Git name and email as in step 1. |
| `No such file or directory: '.../raw_data'` | Create `raw_data/` and copy your data into it as in step 2. |
| `is a ZIP file` | Unpack it with the command shown, then delete the ZIP. |
| `Missing image directory or image list` | Export again as Ultralytics YOLO Detection 1.0 with **Save images**, and unpack it so that `data.yaml` is directly in the export's folder. |
| `contains several subsets` | A whole-project export is in a set folder; move it directly under `raw_data/`. |
| `Source class ... is not a configured class` | Add the class to `custom_classes` or `ignore_classes`. |
| `Unknown class id` | A label uses an ID that is not in the source's `data.yaml` or, without one, not in `custom_classes`. |
| `Invalid normalized YOLO box` | A box has its centre outside the image or no size; check or re-export the labels. |
| `Prepare stage produced 0 ... frames` | A set ended up empty; check the data and that train has enough images to take the missing sets from. |
| `CUDA out of memory` | Lower the model's `batch_size` or `train.image_size` in `params.yaml`. |
| `init_weights must point to an existing, non-empty file` | Copy the starting model to `models/finetune/best.pt`. |
| `no remote specified` | Set up shared storage before `dvc push` or `dvc pull`. |
| `No baseline comparison: ...` | Not an error; the message says why there is no comparison. |
| An older experiment is missing from `dvc exp show` | Show more commits, for example `-n 30`. |

To check the installation, run `poetry run pytest`. `poetry run pytest --heavy`
also trains a small model.
