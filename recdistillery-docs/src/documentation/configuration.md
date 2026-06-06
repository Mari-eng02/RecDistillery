# Configuration

RecDistillery experiments are assembled from YAML configuration files. The
config tree separates reusable defaults from complete experiment definitions.

## Config Layout

```text
config/
  dataset/       dataset definitions
  teacher/       teacher model defaults
  student/       student model defaults
  distillation/  distiller defaults
  optimization/  optimizer and scheduler defaults
  runtime/       runtime options
  evaluation/    metric and top-k defaults
  composites/    reusable experiment templates
  experiments/   complete runnable experiments
```

## Complete Experiments

```text
config/experiments/teacher/
config/experiments/student/
config/experiments/recdistill/
```

## Config Integration

::: recdistill.config_integration

## Config Loader

::: config.config_loader

## Schemas

::: config.schemas
