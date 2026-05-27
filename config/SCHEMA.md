# Config Schema

RecDistillery config files are modular. Experiment files should reference
module defaults and override only the fields that differ for a specific run.

## Canonical Roots

Each experiment config must use exactly one top-level root:

- `train_teacher`: train a teacher model.
- `train_student`: train a plain student model without distillation.
- `distill_student`: train a student with knowledge distillation.

Rootless experiment configs are not valid.

## Executable Templates

The canonical templates live in `config/composites/`:

- `teacher_template.yaml`
- `student_template.yaml`
- `recdistill_template.yaml`

Use these files as the executable source of truth for the base shape. This
document describes how to read and fill that shape.

## Shape Summary

```text
train_teacher
|-- dataset
|-- teacher
|   `-- default
|-- optimization
|   `-- default
|-- runtime
|   `-- default
`-- evaluation
    `-- default
```

```text
train_student
|-- dataset
|-- student
|   `-- default
|-- optimization
|   `-- default
|-- runtime
|   `-- default
`-- evaluation
    `-- default
```

```text
distill_student
|-- dataset
|-- teacher
|   `-- default
|-- student
|   `-- default
|-- distillation
|   |-- default
|   `-- strategy
|-- optimization
|   `-- default
|-- runtime
|   `-- default
`-- evaluation
    `-- default
```

Every mapping may also contain override fields next to `default`. For example,
`optimization.batch_size`, `runtime.num_workers`, or `student.embedding_dim`.

## Module Defaults

Any mapping can use a `default` path. Relative paths are resolved from `config/`.

```yaml
student:
  default: student/elliot/lgcn.yaml
```

Sibling keys override loaded defaults recursively:

```yaml
student:
  default: student/elliot/lgcn.yaml
  embedding_dim: 32
```

## Field Ownership

| Field family | Location | Notes |
|---|---|---|
| Dataset paths and parsing | `dataset/` | Dataset files define data locations and parsing options. |
| Teacher architecture | `teacher/<framework>/<model>.yaml` | Teacher modules use `model`. |
| Student architecture | `student/<framework>/<model>.yaml` | Student modules use `backbone`. |
| Distiller parameters | `distillation/<strategy>.yaml` | Includes distiller lambdas, temperatures, RRD, UnKD, topology options. |
| Generic optimization | `optimization/` | Includes epochs, batch size, learning rate, L2 regularization, Bayesian settings. |
| Runtime behavior | `runtime/` | Includes seed, device, workers, output strategy, logging args. |
| Evaluation behavior | `evaluation/` | Includes cutoffs, metrics, evaluation frequency, selection policy. |
| Experiment overrides | `experiments/` | Use sparingly for run-specific deviations from defaults. |

## Teacher Training

Use `train_teacher.teacher.default` to select the teacher module:

```yaml
train_teacher:
  dataset: citeulike

  teacher:
    default: teacher/recbole/sgl.yaml

  optimization:
    default: optimization/default.yaml

  runtime:
    default: runtime/default.yaml

  evaluation:
    default: evaluation/default.yaml
```

Teacher module files contain fields such as:

```yaml
framework: recbole
model: SGL
embedding_dim: 200
```

## Student Training

Use `train_student.student.default` to select the student module:

```yaml
train_student:
  dataset: citeulike

  student:
    default: student/elliot/lgcn.yaml

  optimization:
    default: optimization/default.yaml

  runtime:
    default: runtime/default.yaml

  evaluation:
    default: evaluation/default.yaml
```

Student module files contain fields such as:

```yaml
framework: elliot
backbone: LGCN
embedding_dim: 20
```

## Distillation Training

Use `distill_student.teacher.default`, `distill_student.student.default`, and
`distill_student.distillation.default`:

```yaml
distill_student:
  dataset: citeulike

  teacher:
    default: teacher/elliot/lgcn.yaml

  student:
    default: student/elliot/lgcn.yaml

  distillation:
    default: distillation/de.yaml
    strategy: DE

  optimization:
    default: optimization/default.yaml

  runtime:
    default: runtime/default.yaml

  evaluation:
    default: evaluation/default.yaml
```

`distillation.strategy` is the only distiller selector.

## Overrides

Write experiment-level overrides only when they intentionally differ from the
module default:

```yaml
distill_student:
  student:
    default: student/elliot/lgcn.yaml
    embedding_dim: 32
```

Do not repeat fields that are identical to the referenced default module.

## Deprecated Fields

Do not use:

- `student.model`
- `teacher.adapter`
- `dataloader`
- dataset `strategy`
- `optimization.optuna`
- `optimization.grid_search`
- rootless experiment configs
- `train_student` as the root for RecDistill configs
