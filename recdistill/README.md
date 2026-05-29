# RecDistill Core

`recdistill` is the framework core for teacher training, student training, and
recommendation distillation. The current design is framework-aware but not
framework-dependent: models come from `recommenders/{recbole,elliot,lenskit}`,
while RecDistill provides the unified PyTorch training loop, teacher import
format, distillers, samplers, evaluation, and checkpointing.

The important rule is:

```text
models imported from recommendation frameworks
  -> adapter-backed torch model
  -> RecDistill training loop
```

Externally trained teachers can also be imported as `.teacher` files and used
for distillation, even when their original model is not trainable by the
RecDistill PyTorch loop.

## Main Components

```text
recdistill/
  registry.py              canonical model/distiller aliases
  supported_models.py      torch-compatible and adapter-backed model registry
  model_validation.py      framework/model/distiller compatibility checks
  paths.py                 output path helpers
  checkpointing.py         .teacher, .student, .distilled_student checkpoints
  experiment_runner.py     RecDistillExperimentRunner
  native_runner.py         native teacher/student training runner

  data/                    interaction datasets and split loading
  distillers/              DE, RRD, UnKD, HTD, FTD, CompositeDistiller
  evaluation.py            top-k recommendation metrics
  factories.py             model and distiller factories
  framework_backbone.py    adapters for RecBole, Elliot, and Lenskit models
  samplers/                negative and distillation samplers
  teachers/                teacher import, adapters, native .teacher format
  trainers/                training loops
```

## Supported Training Models

Only adapter-backed torch models can be trained inside RecDistill. The current
set is:

| Framework | Models |
| --- | --- |
| RecBole | `BPRMF`, `LINE`, `LGCN`, `NGCF`, `DGCF`, `SGL`, `SPECTRALCF`, `NMF` |
| Elliot | `BPRMF`, `NMF`, `LGCN`, `NGCF`, `DGCF`, `SGL`, `ULTRAGCN` |
| Lenskit | `BPRMF`, `LGCN` |

Use the welcome script to print the supported model table:

```powershell
python scripts\recdistill\welcome.py
```

Models that are not trainable by the unified loop can still be used as teachers if they are trained externally and imported as `.teacher`.

## Training Entry Points

### Teacher Training

Train an adapter-backed teacher with:

```powershell
python scripts\teacher_training\teacher_training.py `
  --framework recbole `
  --model BPRMF `
  --dataset citeulike
```

or use a complete experiment config:

```powershell
python scripts\teacher_training\teacher_training.py `
  --config config\experiments\teacher\<experiment>.yaml
```

Teacher checkpoints are saved as `.teacher`.

### Student Training

Train a student without distillation with:

```powershell
python scripts\student_training\student_training.py `
  --framework recbole `
  --backbone LGCN `
  --dataset citeulike `
  --distillation none
```

Student checkpoints are saved as `.student`.

### Distilled Student Training

Train a distilled student from config:

```powershell
python scripts\recdistill\train_student_from_config.py `
  --config config\experiments\recdistill\<experiment>.yaml
```

or compose the config on the fly:

```powershell
python scripts\recdistill\train_student_from_config.py `
  --dataset citeulike `
  --teacher-path results\teacher\<run>\artifacts\<teacher>_best.teacher `
  --distiller de `
  --student LGCN `
  --student-framework recbole
```

Distilled student checkpoints are saved as `.distilled_student`.

When `--config` is omitted, the scripts compose a config from
`config/dataset/`, `config/teacher/`, `config/student/`,
`config/distillation/`, and `config/composites/`, then save the generated file
as a run artifact under:

```text
results/<kind>/<timestamp>_<framework>_<model>_<dataset>_<experiment_id>/config/
```

## Teacher Import Contract

The neutral runtime teacher object is `TeacherState`:

```python
TeacherState(
    user_embeddings=...,
    item_embeddings=...,
    metadata={...},
    scorer=None,
)
```

Teachers can be represented in three ways:

- embedding-based: user embeddings + item embeddings
- score-based: dense precomputed user-item score matrix
- ranking/top-k-based: precomputed top-k items, optionally with top-k scores

Only three import adapters are registered:

- `CheckpointAdapter` for generic torch checkpoints with serialized teacher state, embeddings, scores, or top-k tensors.
- `PredictionsJsonAdapter` for JSON recommendation exports.
- `RecBolePthAdapter` for `.pth` checkpoints containing user/item embedding tensors.

Import a generic checkpoint:

```powershell
python scripts\recdistill\import_teacher.py `
  --input teacher_checkpoint.pt `
  --format checkpoint `
  --framework external `
  --model-name ExternalTeacher `
  --dataset citeulike `
  --embedding-dim 200
```

Import precomputed recommendation lists:

```powershell
python scripts\recdistill\import_teacher.py `
  --input predictions.json `
  --format predictions_json `
  --framework external `
  --model-name ExternalTeacher `
  --dataset citeulike
```

Import a `.pth` checkpoint with user/item embeddings:

```powershell
python scripts\recdistill\import_teacher.py `
  --input model.pth `
  --format recbole_pth `
  --framework recbole `
  --model-name BPRMF `
  --dataset citeulike
```

List teacher import adapters:

```powershell
python scripts\recdistill\import_teacher.py --list-adapters
```

## Distiller Compatibility

RecDistill validates distiller/model/teacher-format compatibility before
training starts.

| Distiller | Teacher requirement | Student requirement |
| --- | --- | --- |
| `DE` | embeddings | adapter-backed torch student with compatible embedding shape |
| `HTD` | embeddings | adapter-backed torch student exposing embeddings |
| `FTD` | embeddings | adapter-backed torch student exposing embeddings |
| `RRD` | embeddings, score matrix, or top-k/ranking | adapter-backed torch student that scores items |
| `UnKD` | embeddings, score matrix, or top-k/ranking | adapter-backed torch student that scores items |


The validator also catches:

- teacher model/backbone not provided by the selected framework
- model available in imported definitions but not wired as a RecDistill adapter
- non-torch or non-trainable framework implementations
- partial embedding teachers, for example only user embeddings without item embeddings
- ambiguous teacher configs that declare multiple representations at once
- top-k score files without top-k item files

## Evaluation Flow

A complete experiment usually runs in this order:

```text
1. teacher_training.py or import_teacher.py
2. evaluate_teacher.py
3. student_training.py --distillation none
4. evaluate_students.py
5. train_student_from_config.py
6. evaluate_students.py
```


## Checkpoint Formats

Teacher checkpoints use:

```text
format_version: recdistill.teacher.v2
```

They store embeddings or a serialized scorer, metadata, mappings, and import
provenance.

Student checkpoints use:

```text
format_version: recdistill.student.v1
```

They store model weights, optimizer state, dataset/model/distiller metadata, and
the config hash.

## Design Rules

- Keep model definitions in `recommenders/{recbole,elliot,lenskit}`.
- Keep framework-specific model adaptation in `framework_backbone.py`.
- Keep teacher import behind `.teacher` and `TeacherState`.
- Train teachers and students inside RecDistill only when the model is
  adapter-backed and torch-compatible.
- Use externally trained non-torch models only through `import_teacher.py`.
- Keep distillers independent from the original framework implementation.
