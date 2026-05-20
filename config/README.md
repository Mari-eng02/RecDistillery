# Configuration System

This directory contains the configuration system for the framework RecDistillery.
All new reusable configurations should live here, either as composable base
files or as ready-to-run presets.

## Directory Layout

```text
config/
|-- __init__.py
|-- config_loader.py              # Loading, composition, listing, validation
|-- schemas.py                    # Pydantic schemas for validated configs
|-- datasets/                     # Dataset split paths and parsing options
|   |-- amazon_cd.yaml
|   |-- bookcrossing.yaml
|   `-- citeulike.yaml
|-- models/
|   |-- teacher/                  # Teacher model defaults
|   |   |-- recbole/
|   |   |-- elliot/
|   |   `-- lenskit/
|   `-- student/                  # Student model defaults
|       |-- recbole/
|       |-- elliot/
|       `-- lenskit/
|-- distillers/                   # Distillation strategy defaults
|   |-- de.yaml
|   |-- de_rrd.yaml
|   |-- ftd.yaml
|   |-- htd.yaml
|   |-- rrd.yaml
|   `-- unkd.yaml
|-- experiments/                  # Templates used for on-the-fly composition
|   |-- teacher_template.yaml
|   |-- student_template.yaml
|   |-- recdistill_template_de.yaml
|   |-- recdistill_template_de_rrd.yaml
|   |-- recdistill_template_ftd.yaml
|   |-- recdistill_template_htd.yaml
|   |-- recdistill_template_rrd.yaml
|   `-- recdistill_template_unkd.yaml
`-- presets/
    |-- teacher/                  # Ready-to-run teacher presets
    |   `-- generated/            # Auto-generated teacher presets
    |-- student/                  # Ready-to-run student presets
    |   `-- generated/            # Auto-generated student presets
    `-- recdistill/               # Ready-to-run distillation presets
        `-- generated/            # Auto-generated distillation presets
```

Use `presets/{teacher,student,recdistill}/` for ready-to-run
presets.

When a CLI script composes a config on the fly because no `--config` was
provided, it also saves the generated preset in `generated/` subdirectory, so 
it can be reused later with `--config`:

```text
config/presets/teacher/generated/...
config/presets/student/generated/...
config/presets/recdistill/generated/...
```

CLI examples use paths relative to the project root, so they include the
`config/` prefix. Python `ConfigLoader` examples use paths relative to
`config/`, so they start from `presets/...`.

## Running With Presets

### Teacher Training

Use a teacher preset explicitly with `--config`:

```powershell
python scripts/teacher_training/teacher_training.py `
  --config config/presets/teacher/recbole/ngcf/citeulike/recbole_ngcf_citeulike_200.yaml
```

If `--config` is omitted, `teacher_training.py` composes a native teacher config
from `config/datasets/`, `config/models/teacher/<framework>/`, and
`config/experiments/teacher_template.yaml`:

```powershell
python scripts/teacher_training/teacher_training.py `
  --framework recbole `
  --model NGCF `
  --dataset citeulike
```

### Student Training Without Distillation

Use a student preset explicitly with `--config`:

```powershell
python scripts/student_training/student_training.py `
  --config config/presets/student/recbole/lgcn/citeulike/recbole_lgcn_citeulike_20.yaml `
  --distillation none
```

If `--config` is omitted, `student_training.py` composes a native student config
from `config/datasets/`, `config/models/student/<framework>/`, and
`config/experiments/student_template.yaml`:

```powershell
python scripts/student_training/student_training.py `
  --framework recbole `
  --backbone LGCN `
  --dataset citeulike `
  --distillation none
```

Generated student configs are saved under
`config/presets/student/generated/` and can be reused with `--config`.

### RecDistill Student Training

Run a distilled student from a ready preset:

```powershell
python scripts/recdistill/train_student_from_config.py `
  --config config/presets/recdistill/de/recbole/ngcf/recbole/lgcn/citeulike/de_recbole_ngcf_recbole_lgcn_citeulike.yaml
```

Or let the script compose the config on the fly:

```powershell
python scripts/recdistill/train_student_from_config.py `
  --dataset citeulike `
  --teacher NGCF `
  --teacher-framework recbole `
  --distiller de `
  --student LGCN `
  --student-framework recbole
```

Generated distillation configs are saved under
`config/presets/recdistill/generated/` and can be reused with `--config`.

## Loading Configurations In Python

```python
from config import get_config_loader

loader = get_config_loader()

dataset_cfg = loader.load_dataset_config("citeulike")

teacher_model_cfg = loader.load_model_config(
    "teacher",
    "ngcf",
    framework="recbole",
)

student_model_cfg = loader.load_model_config(
    "student",
    "lgcn",
    framework="recbole",
)

recdistill_cfg = loader.load_recdistill_config(
    "presets/recdistill/de/recbole/ngcf/recbole/lgcn/citeulike/de_recbole_ngcf_recbole_lgcn_citeulike.yaml"
)
```

You can also load wrapped presets directly:

```python
from config import get_config_loader

loader = get_config_loader()

teacher_preset = loader.load_preset(
    "presets/teacher/recbole/ngcf/citeulike/recbole_ngcf_citeulike_200.yaml"
)

student_preset = loader.load_preset(
    "presets/student/recbole/lgcn/citeulike/recbole_lgcn_citeulike_20.yaml"
)

recdistill_cfg = loader.load_recdistill_preset(
    "presets/recdistill/de/recbole/ngcf/recbole/lgcn/citeulike/de_recbole_ngcf_recbole_lgcn_citeulike.yaml"
)
```

## Composing Configurations In Python

```python
from config import get_config_loader

loader = get_config_loader()

teacher_exp = loader.compose_teacher_training(
    dataset_name="citeulike",
    framework="recbole",
    model_name="ngcf",
)

student_exp = loader.compose_student_training(
    dataset_name="citeulike",
    framework="recbole",
    model_name="lgcn",
)

recdistill_exp = loader.compose_recdistill_experiment(
    dataset_name="citeulike",
    teacher_model="nmf",
    teacher_framework="recbole",
    distiller_strategy="de",
    student_backbone="bprmf",
    student_framework="recbole",
)
```


## Listing Available Components

```python
from config import get_config_loader

loader = get_config_loader()

datasets = loader.list_datasets()
all_models = loader.list_models()
teacher_models = loader.list_models("teacher")
student_models = loader.list_models("student")
distillers = loader.list_distillers()
recdistill_presets = loader.list_presets("recdistill")
teacher_presets = loader.list_presets("teacher")
student_presets = loader.list_presets("student")
```

## Data Loading

RecDistill training and evaluation scripts load splits through
`recdistill.data.datarec_loader`. The loader uses DataRec's
`read_transactions_tabular` when `datarec-lib` is installed and falls back to a
compatible pandas reader in lightweight environments.


## Validation

Configurations are validated in two layers.

First, Pydantic schemas from `config/schemas.py` validate structure and field
types. Invalid files raise a validation error with the field that failed.

Second, RecDistill model checks from `recdistill/model_validation.py` validate
framework/model compatibility before training starts. These checks catch:

- model/backbone not provided by the selected framework
- imported models that are torch-compatible but not adapter-backed yet
- known non-torch or non-trainable models
- distillation strategy conflicts

Main schema groups:

- `DataConfig`
- `ModelConfig`
- `OptimizationConfig`
- `EvaluationConfig`
- `EarlyStoppingConfig`
- `RuntimeConfig`
- `DistillerConfig`
- `RecDistillConfig`
- `ConfigPreset`

## Adding New Configurations

### Add A Dataset

Create `config/datasets/new_dataset.yaml`:

```yaml
name: new_dataset
strategy: fixed
train_path: data/new_dataset/train.tsv
validation_path: data/new_dataset/val.tsv
test_path: data/new_dataset/test.tsv
dataloader: DataSetLoader
column_names: ["userId", "itemId", "rating", "timestamp"]
kcore: [10, 10]
```

### Add A Model

Create `config/models/teacher/<framework>/new_model.yaml` or
`config/models/student/<framework>/new_model.yaml`:

```yaml
framework: recbole
backbone: NewModel
embedding_dim: 200
learning_rate: 0.001
l2_reg: 0.0001
```

Teacher model configs use `model`; student model configs use `backbone`:

```yaml
framework: recbole
model: NewModel
embedding_dim: 200
learning_rate: 0.001
l2_reg: 0.0001
```

### Add A Distiller

Create `config/distillers/new_strategy.yaml` and, if needed, a matching
template in `config/experiments/recdistill_template_new_strategy.yaml`:

```yaml
strategy: NewStrategy
temperature: 3.0
lambda_kl: 0.5
```

### Add A Preset

Create the new ready-to-run file under `config/presets/`, for example:

```text
config/presets/student/recbole/lgcn/citeulike/recbole_lgcn_citeulike_20.yaml
config/presets/teacher/recbole/lgcn/citeulike/recbole_lgcn_citeulike_200.yaml
```

Use this when the exact experiment should be reproducible without relying on
on-the-fly composition. 
