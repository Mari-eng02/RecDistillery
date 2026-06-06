# Framework Overview

RecDistillery is organized around a teacher-student distillation workflow for
recommender systems. It keeps external recommender frameworks behind adapters
and exposes a unified PyTorch training loop for teacher training, student
training, distillation, checkpointing, and evaluation.

## Core Package

The main package is `recdistill/`:

```text
recdistill/
  data/                 dataset loading and interaction batches
  teachers/             teacher import, registry, state, serialization
  distillers/           DE, RRD, UnKD, HTD, FTD, and composite distillers
  samplers/             negative and teacher-guided sampling
  trainers/             training loop abstractions
  checkpointing.py      teacher, student, and distilled-student artifacts
  config_integration.py config composition helpers
  evaluation.py         top-k ranking metrics
  experiment_runner.py  distillation experiment runner
  factories.py          model and distiller builders
  framework_backbone.py RecBole, Elliot, and Lenskit adapters
  native_runner.py      native teacher/student training runner
  registry.py           canonical aliases
  supported_models.py   trainable model metadata
```

## Runtime Flow

```text
prepared dataset
  -> teacher training or teacher import
  -> optional teacher evaluation
  -> student training baseline
  -> student distillation
  -> student evaluation
  -> tracked results and artifacts
```

## Main Entry Points

```text
scripts/teacher_training/teacher_training.py
scripts/student_training/student_training.py
scripts/recdistill/import_teacher.py
scripts/recdistill/train_student_from_config.py
scripts/recdistill/evaluate_teacher.py
scripts/recdistill/evaluate_students.py
```

## Supported Frameworks

The current adapter-backed training set covers models from RecBole, Elliot, and
Lenskit. The supported model table is defined in `recdistill.supported_models`
and exposed by the welcome script:

```bash
python scripts/recdistill/welcome.py
```
