# Agent Guide For Config Editing

This guide defines operational rules for agents editing or generating
RecDistillery configs.

## Primary Rule

Prefer composition over duplication.

Use module defaults:

```yaml
default: path/to/module.yaml
```

Only add sibling fields when the experiment intentionally overrides the module
default.

## Canonical Roots

Use exactly one root per experiment config:

- `train_teacher` for teacher training
- `train_student` for plain student training
- `distill_student` for distillation training

Never generate rootless experiment configs.

## Field Ownership

- Dataset paths and parsing options belong in `config/dataset/`.
- Teacher model architecture belongs in `config/teacher/<framework>/<model>.yaml`.
- Student model architecture belongs in `config/student/<framework>/<model>.yaml`.
- Generic training parameters belong in `config/optimization/`.
- Distiller-specific parameters belong in `config/distillation/<strategy>.yaml`.
- Runtime parameters belong in `config/runtime/`.
- Evaluation parameters belong in `config/evaluation/`.
- Experiment-specific overrides belong in `config/experiments/`.

## Module Paths

Use paths relative to `config/`:

```yaml
teacher:
  default: teacher/elliot/lgcn.yaml
```

Do not prefix standard module paths with `config/`.

Use forward slashes in YAML paths.

## Naming Rules

Teacher experiment files:

```text
config/experiments/teacher/<framework>_<model>_<dataset>.yaml
```

Student experiment files:

```text
config/experiments/student/<framework>_<backbone>_<dataset>.yaml
```

RecDistill experiment files:

```text
config/experiments/recdistill/<distiller>_<teacher_framework>_<teacher_model>_to_<student_framework>_<student_backbone>_<dataset>.yaml
```

Shorter names are acceptable only when the experiment directory contains a
single unambiguous setup for that distiller and dataset.

## Teacher Rules

Teacher experiment blocks should usually look like this:

```yaml
teacher:
  default: teacher/recbole/sgl.yaml
```

Teacher module files use:

```yaml
framework: recbole
model: SGL
embedding_dim: 200
```

Do not use `backbone` for teachers.

Do not put `learning_rate` or `l2_reg` in teacher model defaults.

## Student Rules

Student experiment blocks should usually look like this:

```yaml
student:
  default: student/elliot/lgcn.yaml
```

Student module files use:

```yaml
framework: elliot
backbone: LGCN
embedding_dim: 20
```

Do not use `student.model`.

Do not put `learning_rate` or `l2_reg` in student model defaults.

## Distillation Rules

Use:

```yaml
distillation:
  default: distillation/de_rrd.yaml
  strategy: DE_RRD
```

`distillation.strategy` is the only selector for the distiller.

Do not infer the distiller from `student.model`.

Do not combine `HTD` and `FTD` in the same strategy.

## Search Rules

Use:

```yaml
optimization:
  bayesian:
    enabled: true
```

Do not use `optimization.optuna`.

Do not use `optimization.grid_search`.

Generic Bayesian settings belong in `optimization/`.

Distiller-specific search spaces belong in `distillation/<strategy>.yaml`.

## Overrides

Valid override:

```yaml
student:
  default: student/elliot/lgcn.yaml
  embedding_dim: 32
```

Invalid duplication:

```yaml
student:
  default: student/elliot/lgcn.yaml
  framework: elliot
  backbone: LGCN
  embedding_dim: 20
```

Only write `framework`, `model`, `backbone`, or `embedding_dim` in an
experiment when the field intentionally overrides the default module.

## Deprecated Fields

Never introduce:

- `student.model`
- `teacher.adapter`
- `dataloader`
- dataset `strategy`
- `optimization.optuna`
- `optimization.grid_search`
- rootless experiment configs
- RecDistill root `train_student`

## Validation Checklist

After editing configs:

1. Check that each experiment root is exactly one of `train_teacher`, `train_student`, or `distill_student`.
2. Check that `default:` paths are relative to `config/` unless intentionally absolute.
3. Check that YAML paths use forward slashes.
4. Check that teachers use `model` only inside teacher modules.
5. Check that students use `backbone` only inside student modules.
6. Check that RecDistill configs use `distill_student.distillation.strategy`.
7. Check that repeated fields are real overrides, not copies of defaults.
8. Run config validation or the relevant dry-run command.
