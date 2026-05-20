# Experiment Launchers

Shell launchers are grouped by the pipeline they belong to.

- `recdistill/`: current RecDistill launchers for distillation, teacher checks, evaluation, result fetching.
- `baseline/`: generic launchers for teacher training and non-distilled baseline student training.

Run the scripts from the repository root, for example:

```bash
bash experiments/recdistill/train_distiller.sh DE citeulike BPRMF
bash experiments/baseline/teacher_training.sh BPRMF citeulike 0 best
```
