# Experiment Launchers

Shell launchers are grouped by the pipeline they belong to.

- `recdistill/`: current RecDistill launchers for distillation, teacher checks, evaluation, and tracked reruns.
- `baseline/`: generic launchers for teacher training and non-distilled baseline student training.

Run the scripts from the repository root, for example:

```bash
bash experiments/recdistill/train_distiller.sh DE citeulike recbole BPRMF recbole LGCN 0 0
bash experiments/recdistill/train_distiller.sh RRD citeulike results/teacher/<run>/artifacts/<teacher>_best.teacher recbole BPRMF 0 1
bash experiments/baseline/teacher_training.sh recbole BPRMF citeulike 0
```

The RecDistill launchers expose both teacher and student frameworks explicitly:
`recbole`, `elliot`, and `lenskit` are accepted when a matching config exists
under `config/{teacher,student}/<framework>/`.

For imported teachers, pass the `.teacher` artifact path as the teacher argument.
The launchers will forward only `--teacher-path` to
`scripts/recdistill/train_student_from_config.py`; framework/model arguments are
used only for teachers that must be trained or resolved from config.

Distillation launchers write into the canonical result layout:

```text
results/recdistill/<timestamp>_<student_framework>_<strategy>_<student_model>_<dataset>_<experiment_id>/
```

with artifacts, config copies, logs, and evaluation outputs kept in the run
directory.
