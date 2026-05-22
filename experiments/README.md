# Experiment Launchers

Shell launchers are grouped by the pipeline they belong to.

- `recdistill/`: current RecDistill launchers for distillation, teacher checks, evaluation, result fetching.
- `baseline/`: generic launchers for teacher training and non-distilled baseline student training.

Run the scripts from the repository root, for example:

```bash
bash experiments/recdistill/train_distiller.sh DE citeulike recbole BPRMF recbole LGCN 0 0
bash experiments/recdistill/train_distiller.sh RRD citeulike results/teachers/lenskit/ItemKNNScorer/citeulike/imported/lenskit_ItemKNNScorer_citeulike_200.teacher recbole BPRMF 0 1
bash experiments/baseline/teacher_training.sh recbole BPRMF citeulike 0 best
```

The RecDistill launchers expose both teacher and student frameworks explicitly:
`recbole`, `elliot`, and `lenskit` are accepted when a matching config exists
under `config/models/{teacher,student}/<framework>/`.

For imported teachers, pass the `.teacher` artifact path as the teacher argument.
The launchers will forward only `--teacher-path` to
`scripts/recdistill/train_student_from_config.py`; framework/model arguments are
used only for teachers that must be trained or resolved from config.

Distillation launchers write fixed-parameter runs with `--output-strategy fixed`,
so artifacts land under `fixed/wei`. Runs produced from best Bayesian/Optuna
parameters should use the `best/wei` convention.
