# Evaluation

RecDistillery evaluates teachers and students with top-k recommendation metrics.
The common metric implementation lives in `recdistill.evaluation` and is used by
the teacher and student evaluation scripts.

## Teacher Evaluation

```bash
python scripts/recdistill/evaluate_teacher.py \
  --teacher-path results/teacher/<run>/artifacts/<teacher>_best.teacher
```

## Student Evaluation

```bash
python scripts/recdistill/evaluate_students.py \
  --student-path results/student/<run>/artifacts/<student>_best.student
```

Distilled students use the same evaluator:

```bash
python scripts/recdistill/evaluate_students.py \
  --student-path results/recdistill/<run>/artifacts/<student>_best.distilled_student
```

## Metrics

The evaluation module computes ranking metrics such as precision, recall, NDCG,
and hit ratio over held-out validation or test interactions.

| Function | Purpose |
| --- | --- |
| `evaluate_teacher` | Evaluates a `TeacherState` on validation and test splits. |
| `evaluate_student` | Evaluates a trained student model with the same ranking protocol. |
| `evaluate_embeddings` | Shared embedding/scorer evaluator used by teacher and student APIs. |

::: recdistill.evaluation.evaluate_teacher

::: recdistill.evaluation.evaluate_student

::: recdistill.evaluation.evaluate_embeddings
