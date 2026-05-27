# Configuration System

This directory contains modular configuration files for RecDistillery. 
Complete experiment files can be composed from reusable modules, then stored under `config/experiments/`.

For the canonical shape and editing rules, see:

- `SCHEMA.md`
- `AGENT_GUIDE.md`

Experiment configs use explicit roots:

- `train_teacher` for teacher training
- `train_student` for plain student training
- `distill_student` for student distillation

## Directory Layout

```text
config/
|-- __init__.py
|-- config_loader.py              # Loading, composition, listing, validation
|-- schemas.py                    # Pydantic schemas for validated configs
|-- dataset/                      # Dataset split paths and parsing options
|   |-- amazon_cd.yaml
|   |-- bookcrossing.yaml
|   `-- citeulike.yaml
|-- teacher/                      # Teacher model defaults
|   |-- recbole/
|   |-- elliot/
|   `-- lenskit/
|-- student/                      # Student model defaults
|   |-- recbole/
|   |-- elliot/
|   `-- lenskit/
|-- distillation/                 # Distillation strategy defaults
|-- optimization/                 # Optimization defaults
|-- runtime/                      # Runtime defaults
|-- evaluation/                   # Evaluation defaults
|-- composites/                   # Templates used for composition
`-- experiments/
    |-- teacher/                  # Complete teacher experiment configs
    |-- student/                  # Complete student experiment configs
    `-- recdistill/               # Complete distillation experiment configs
```

## Module Defaults

Any mapping can reference a default module:

```yaml
optimization:
  default: optimization/default.yaml
  epochs: 50
  learning_rate: 0.0005
```

The path may be absolute or relative to `config/`. Sibling keys override the
loaded default recursively.

Use module defaults for teacher, student, distillation, optimization, runtime,
and evaluation blocks. Only write sibling keys when an experiment intentionally
overrides the module default:

```yaml
teacher:
  default: teacher/elliot/lgcn.yaml

student:
  default: student/elliot/lgcn.yaml
  embedding_dim: 32

distillation:
  default: distillation/de_rrd.yaml
  strategy: DE_RRD
```

## Canonical Field Rules

- Teacher training configs use `train_teacher.teacher.model`.
- Plain student training configs use `train_student.student.backbone`.
- RecDistill configs use `distill_student.distillation.strategy`.
- Experiment configs should reference model and distiller modules with `default:`.
- Generic training parameters go under `optimization/`.
- Distiller-specific parameters go under `distillation/<strategy>.yaml`.
- Runtime parameters go under `runtime/`.
- Evaluation parameters go under `evaluation/`.
- Model architecture parameters go under `teacher/` or `student/`.

Do not use legacy fields such as `student.model`, `teacher.adapter`,
`dataloader`, dataset `strategy`, `optimization.optuna`, or
`optimization.grid_search`.

## Running With Experiments

Use complete experiment configs with `--config`:

```powershell
python scripts/teacher_training/teacher_training.py `
  --config config/experiments/teacher/<experiment>.yaml

python scripts/student_training/student_training.py `
  --config config/experiments/student/<experiment>.yaml `
  --distillation none

python scripts/recdistill/train_student_from_config.py `
  --config config/experiments/recdistill/<experiment>.yaml
```

If `--config` is omitted, scripts compose a config from modules and save the
result under `config/experiments/<teacher|student|recdistill>/`.

## Python Usage

```python
from config import get_config_loader

loader = get_config_loader()

dataset_cfg = loader.load_dataset_config("citeulike")
teacher_cfg = loader.load_model_config("teacher", "ngcf", framework="recbole")
student_cfg = loader.load_model_config("student", "lgcn", framework="recbole")
distillers = loader.list_distillers()
experiments = loader.list_experiments("recdistill")
```

Compose complete configs:

```python
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

## Adding Modules

Dataset:

```yaml
name: new_dataset
train_path: data/new_dataset/train.tsv
validation_path: data/new_dataset/val.tsv
test_path: data/new_dataset/test.tsv
column_names: ["userId", "itemId", "rating", "timestamp"]
kcore: [10, 10]
```

Teacher model: `config/teacher/<framework>/<model>.yaml`

Student model: `config/student/<framework>/<model>.yaml`

Distillation strategy: `config/distillation/<strategy>.yaml`

Shared defaults: `config/optimization/`, `config/runtime/`, `config/evaluation/`
