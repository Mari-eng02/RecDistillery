# Training Pipeline

The training pipeline is split into native teacher/student training and
teacher-student distillation. Both paths use shared data loading, model factory,
checkpointing, tracking, and evaluation utilities.

## Distillation From Config

```bash
python scripts/recdistill/train_student_from_config.py \
  --config config/experiments/recdistill/de_citeulike_001.yaml
```

## Direct Distillation

```bash
python scripts/recdistill/train_student.py \
  --dataset citeulike \
  --teacher-framework recbole \
  --teacher-model BPRMF \
  --student-framework recbole \
  --student-backbone LGCN \
  --lambda-de 0.1
```

## Experiment Runners

::: recdistill.experiment_runner

::: recdistill.native_runner

## Trainers

::: recdistill.trainers.base

::: recdistill.trainers.distillation

## Factories

::: recdistill.factories

## Shared Training Utilities

::: recdistill.training
