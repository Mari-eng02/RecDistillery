# Experiment Launchers

Shell launchers are grouped by the pipeline they belong to.

- `recdistill/`: current RecDistill launchers for distillation, teacher checks, evaluation, result fetching.
- `baseline/`: generic launchers for teacher training and non-distilled baseline student training.

Run the scripts from the repository root, for example:

```bash
bash experiments/recdistill/train_distiller.sh DE citeulike recbole BPRMF recbole LGCN 0 0
bash experiments/baseline/teacher_training.sh recbole BPRMF citeulike 0 best
```

The RecDistill launchers expose both teacher and student frameworks explicitly:
`recbole`, `elliot`, and `lenskit` are accepted when a matching config exists
under `config/models/{teacher,student}/<framework>/`.
