# Artifacts And Results

Training, distillation, import, and evaluation runs write tracked outputs under
`results/`. Each run stores artifacts, configs, logs, and performance files.

## Result Layout

```text
results/
  teacher/
  student/
  recdistill/
```

Each run follows this structure:

```text
results/<kind>/<run>/
  artifacts/
  config/
  logs/
  perf/
```

Run names contain the framework, model, dataset, timestamp, and experiment id.
The current layout no longer creates nested result directories for individual
backbones such as `bprmf/`, `lgcn/`, or `nmf/`; model names are encoded in the
run and artifact filenames instead.

## Artifact Types

```text
.teacher            trained or imported teacher
.student            plain student baseline
.distilled_student  distilled student model
```

## Checkpointing

::: recdistill.checkpointing

## Runtime Paths

`recdistill.paths` resolves datasets, run directories, artifacts, histories, and
performance files for the flat run-based result layout.

| Helper | Purpose |
| --- | --- |
| `experiment_run_dir` | Builds `results/<kind>/<run>/`. |
| `experiment_artifact_path` | Builds paths under a run's `artifacts/` directory. |
| `teacher_artifact_path` | Resolves a native teacher `.teacher` artifact. |
| `student_artifact_path` | Resolves a plain student `.student` artifact. |
| `distilled_student_artifact_path` | Resolves a `.distilled_student` artifact. |
| `resolve_teacher_checkpoint` | Finds an explicit or latest matching teacher checkpoint. |
| `resolve_student_checkpoint` | Resolves plain or distilled student checkpoints. |

::: recdistill.paths
    options:
      filters:
        - "!^BPRMF$"
        - "!^LGCN$"
        - "!^NMF$"
        - "!^BACKBONES$"

## Tracking

::: recdistill.tracking
