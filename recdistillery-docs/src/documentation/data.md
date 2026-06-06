# Data And Datasets

RecDistillery trains and evaluates recommendation models from interaction
splits stored under `data/<dataset>/`. Dataset preprocessing is based on
DataRec, while the training pipeline consumes encoded PyTorch-ready interaction
objects from `recdistill.data`.

## Dataset Layout

Each dataset is expected to expose the canonical split files:

```text
data/<dataset>/
  train.tsv
  val.tsv
  test.tsv
```

The example dataset configs live in:

```text
config/dataset/
  amazon_cd.yaml
  bookcrossing.yaml
  citeulike.yaml
```

## Preparation Scripts

Dataset-specific preparation entry points are stored in
`scripts/data_preparation/`:

```text
scripts/data_preparation/amazon_cd_2014.py
scripts/data_preparation/bookcrossing.py
scripts/data_preparation/citeulike.py
```

These scripts prepare the split files consumed by the training and evaluation loaders.
They are intended as examples, while the framework can be extended to any dataset either 
directly available in DataRec or loaded through its multi-format data interfaces.

## Runtime Data Objects

::: recdistill.data.batch

::: recdistill.data.interactions

## DataRec Loading

::: recdistill.data.datarec_loader
