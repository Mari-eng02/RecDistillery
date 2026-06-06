# Students And Backbones

Students are adapter-backed PyTorch models trained either as plain recommendation
baselines or as distilled models. The framework adapters hide RecBole, Elliot,
and Lenskit implementation details behind a shared training surface.

## Student Training

Train a baseline student without distillation:

```bash
python scripts/student_training/student_training.py \
  --framework recbole \
  --backbone LGCN \
  --dataset citeulike \
  --distillation none
```

Complete student experiment configs live in:

```text
config/experiments/student/
```

## Backbone Contract

Adapter-backed models expose the methods consumed by trainers and distillers:

```text
forward(users, pos_items, neg_items)
score_items(users, items)
user_embeddings()
item_embeddings()
```

## Framework Backbones

::: recdistill.framework_backbone

## Supported Models

::: recdistill.supported_models

## Model Registry

::: recdistill.registry

## Model Validation

::: recdistill.model_validation
