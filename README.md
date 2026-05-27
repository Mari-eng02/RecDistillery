# RecDistillery

RecDistillery is a modular framework for Knowledge Distillation in Recommender Systems, built around a simple idea:

> Great recommendation models, like fine spirits, should be distilled — not diluted.

Large teacher models contain rich collaborative knowledge, complex ranking behaviors, structural information, and latent relational patterns. RecDistillery provides the tools to extract, refine, and transfer this knowledge into lightweight student models through unified and reproducible distillation pipelines.

The current framework supports:

- teacher import from `.pth` or `.pt` embedding checkpoints or predictions `.json` exports into `.teacher`;
- adapter-backed teacher/student training from RecBole, Elliot, and Lenskit model definitions through a torch-based training pipeline;
- 6 distillers: DE, RRD, DE+RRD, HTD, FTD, UnKD.

Imported recommendation model definitions currently come from RecBole, Elliot, and Lenskit. 
RecDistill trains only this subset exposed through PyTorch adapters:
- RecBole:
  - BPRMF (aliases: BPR, BPRMF)
  - LINE (aliases: LINE)
  - LGCN (aliases: LGCN, LightGCN)
  - NGCF (aliases: NGCF)
  - DGCF (aliases: DGCF)
  - SGL (aliases: SGL)
  - SPECTRALCF (aliases: SpectralCF, SPECTRALCF)
  - NMF (aliases: NMF, NeuMF)
- Elliot:
  - BPRMF (aliases: BPR, BPRMF)
  - LGCN (aliases: LGCN, LightGCN)
  - NGCF (aliases: NGCF)
  - DGCF (aliases: DGCF)
  - SGL (aliases: SGL)
  - ULTRAGCN (aliases: UltraGCN, ULTRAGCN)
  - NMF (aliases: NMF, NeuMF)
- Lenskit:
  - BPRMF (aliases: BPRMF)
  - LGCN (aliases: LGCN, LightGCN)

The dataset preprocessing and standardized data management are based on `DataRec`:
- DataRec Documentation: https://www.datarechub.com/
- DataRec Datasets: https://www.datarechub.com/datasets_nav/


# Table of Contents

- [Architecture](#architecture)
- [Repository Setup](#repository-setup)
  - [Prerequisites](#prerequisites)
  - [Clone The Repository](#clone-the-repository)
  - [Configure The Environment](#configure-the-environment)
  - [Download and Prepare Datasets](#download-and-prepare-datasets)
- [Workflow](#workflow)
  - [Compatibility Check](#compatibility-check)
  - [Adding New Model Adapters](#adding-new-model-adapters)
  - [Teacher Import](#teacher-import)
  - [Teacher Training](#teacher-training)
  - [Student Training](#student-training)
  - [Student Distillation](#student-distillation)
  - [Experiment Launchers](#experiment-launchers)
- [Configs And Experiments](#configs-and-experiments)
- [Evaluation](#evaluation)
- [Results Structure](#results-structure)
- [Timing Analysis](#timing-analysis)

---

# Architecture

The framework core lives under `recdistill/`.

```text
recdistill/
  data/                  Interaction datasets and DataRec-compatible loading
  distillers/            DE, RRD, UnKD, HTD, FTD, CompositeDistiller (DE+RRD)
  teachers/              Teacher adapters, import/export, native .teacher format
  samplers/              Negative and distillation samplers
  trainers/              Training loops and optimization helpers
  framework_backbone.py  Framework adapter and model registry
  config_integration.py  RecDistill config composition and loading
  experiment_runner.py   RecDistillExperimentRunner
  native_runner.py       Native teacher/student model training runner
  checkpointing.py       Teacher/student checkpoint management
  model_validation.py    Compatibility and request validation
  supported_models.py    Torch-compatible model registry
  training.py            Shared training utilities
  tracking.py            Experiment logging and metadata
  paths.py               Canonical path resolution helpers
  registry.py            Canonical aliases for models and distillers
```

---

# Repository Setup

## Prerequisites

- Python 3.10+
- Conda
- CUDA 11.8+ (recommended)

Optional:

- Apple Silicon MPS support

## Clone The Repository

```bash
git clone <REPO_URL>
cd RecSys-Distillation
```

## Configure The Environment

CUDA:

```bash
bash setup/setup_environment.sh cuda
```

Apple Silicon:

```bash
bash setup/setup_environment.sh mps
```

CPU only:

```bash
bash setup/setup_environment.sh
```

After installation, activate the environment with:

```bash
source setup/activate_env.sh
```

This activates the default `distillation` environment and sets the project `PYTHONPATH`.

## Download and Prepare Datasets

Expected datasets are:

- Amazon-CD
- BookCrossing
- CiteULike

The preprocessing utilities are based on `DataRec`, so additional datasets can also be used if they are available in that ecosystem. See DataReHub for more dataset references and formats: https://www.datarechub.com/.


Run:

```bash
bash setup/setup_directories.sh
bash setup/setup_datasets.sh
```

The split files generated are stored under:

```text
data/<dataset>/train.tsv
data/<dataset>/val.tsv
data/<dataset>/test.tsv
```

DataRec-compatible loading lives in `recdistill/data/datarec_loader.py`.

Dataset sources:

- Amazon CDs & Vinyl: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- BookCrossing: https://www.kaggle.com/datasets/jirakst/bookcrossing
- CiteULike: https://www.datarechub.com/assets/pages/datasets/citeulike_a/

---

# Workflow

## Compatibility Check

Before launching training, you can list the imported model definitions that are torch-compatible, plus the adapter-backed subset already wired to the current RecDistill unified loop:

```bash
python scripts/recdistill/welcome.py
```

Non-Torch teachers can still be used after conversion to `.teacher` with `import_teacher.py`.

## Adding New Model Adapters

If a model appears in the imported framework definitions but is not listed as adapter-backed, RecDistill cannot train it in the unified PyTorch loop yet. To make it trainable, add a backbone adapter.

1. Add or extend the framework adapter in `recdistill/framework_backbone.py`.
   The adapter must be a `torch.nn.Module` and expose the same surface used by the distillers:

```python
forward(users, pos_items, neg_items) -> FrameworkBatchOutput
score_items(users, items) -> torch.Tensor
user_embeddings() -> torch.Tensor
item_embeddings() -> torch.Tensor
```

2. Wire the model inside the relevant adapter class:
   `RecBoleBackboneAdapter`, `ElliotBackboneAdapter`, or `LensKitBackboneAdapter`.

3. Register the model as trainable in `recdistill/supported_models.py` by adding a `TrainableBackbone` entry with framework, canonical model name, aliases, adapter name, and implementation path.

4. Add the model alias in `recdistill/registry.py` if the model should be resolved from multiple names.

5. Add model configs for teacher and/or student:

```text
config/teacher/<framework>/<model>.yaml
config/student/<framework>/<model>.yaml
```

6. Run the compatibility check again:

```bash
python scripts/recdistill/welcome.py --verbose
```


## Teacher Import

RecDistillery treats the `.teacher` file as the framework-neutral teacher format. Teacher import is handled by a small set of generic adapters that convert external artifacts into either user/item embeddings or precomputed top-k rankings.

The official path is `scripts/recdistill/import_teacher.py`. The import script currently registers only:

- `CheckpointAdapter`: generic torch checkpoints containing a serialized teacher state, embeddings, scores, or top-k tensors.
- `PredictionsJsonAdapter`: JSON prediction exports with `user`, `item`, and optional `score`/`rating`/`rank` fields.
- `RecBolePthAdapter`: `.pth` checkpoints containing user and item embedding tensors.

List the active adapters with:

```bash
python scripts/recdistill/import_teacher.py --list-adapters
```

Import a generic checkpoint:

```bash
python scripts/recdistill/import_teacher.py \
  --input path/to/teacher_checkpoint.pt \
  --format checkpoint \
  --framework external \
  --model-name ExternalTeacher \
  --dataset citeulike \
  --embedding-dim 200
```

Import precomputed recommendation lists:

```bash
python scripts/recdistill/import_teacher.py \
  --input path/to/predictions.json \
  --format predictions_json \
  --framework external \
  --model-name ExternalTeacher \
  --dataset citeulike
```

Import a `.pth` checkpoint with user/item embeddings:

```bash
python scripts/recdistill/import_teacher.py \
  --input path/to/model.pth \
  --format recbole_pth \
  --framework recbole \
  --model-name BPRMF \
  --dataset citeulike
```

Imported teachers are always saved with the canonical teacher layout:

```text
results/teachers/<framework>/<model>/<dataset>/imported/<framework>_<model>_<dataset>_<embedding_dim>.teacher
```

## Teacher Training

Teacher models can also be generated through the framework torch-based training pipeline. 
Train a native RecDistill teacher and save it as a `.teacher` artifact with:

```bash
python scripts/teacher_training/teacher_training.py \
  --framework recbole \
  --model BPRMF \
  --dataset citeulike
```

Alternatively, use a complete experiment config file:

```bash
python scripts/teacher_training/teacher_training.py \
  --config config/experiments/teacher/<experiment>.yaml
```

This script trains a teacher model, exports a `.teacher` file, and makes it available for later distillation or evaluation.

To check that the teacher is readable and compatible before using it in distillation, run this smoke test:

```bash
python scripts/recdistill/teacher_smoke.py \
  --teacher-framework <framework> \
  --teacher-model <model> \
  --dataset <dataset> \
  --embedding-dim <embedding_dim> \
  --top-k 20
```

For imported teachers, pass the artifact explicitly:

```bash
python scripts/recdistill/teacher_smoke.py \
  --teacher-path results/teachers/<framework>/<model>/<dataset>/imported/<teacher>.teacher \
  --teacher-framework <framework> \
  --teacher-model <model> \
  --dataset <dataset> \
  --embedding-dim <embedding_dim>
```

## Student Training (no distillation)

Train a student model without distillation with:

```bash
python scripts/student_training/student_training.py \
  --framework recbole \
  --backbone LGCN \
  --dataset citeulike \
  --distillation none
```

Or use a config file for the student training run:

```bash
python scripts/student_training/student_training.py \
  --config config/experiments/student/<experiment>.yaml
```

The output is a `.student` artifact that can be evaluated or later compared with distilled students.

## Student Distillation

RecDistill exposes two main student training entry points:

- `scripts/recdistill/train_student.py` for direct student training and distillation from a saved teacher.
- `scripts/recdistill/train_student_from_config.py` for config-based, reproducible RecDistill experiments.

Example config-driven distillation:

```bash
python scripts/recdistill/train_student_from_config.py \
  --dataset citeulike \
  --teacher-model BPRMF \
  --teacher-framework recbole \
  --distiller de \
  --student-backbone LGCN \
  --student-framework recbole
```

For imported teachers, pass only the imported artifact path. Do not pass `--teacher-model` or `--teacher-framework`; those are reserved for teacher models that RecDistill resolves/trains through its adapter-backed registry.

```bash
python scripts/recdistill/train_student_from_config.py \
  --dataset citeulike \
  --teacher-path results/teachers/lenskit/ItemKNNScorer/citeulike/imported/lenskit_ItemKNNScorer_citeulike_200.teacher \
  --distiller rrd \
  --student-backbone BPRMF \
  --student-framework recbole
```

Direct distillation from CLI uses the same canonical teacher/student argument names:

```bash
python scripts/recdistill/train_student.py \
  --dataset citeulike \
  --teacher-framework recbole \
  --teacher-model BPRMF \
  --teacher-embedding-dim 200 \
  --student-framework recbole \
  --student-backbone LGCN \
  --student-embedding-dim 20 \
  --lambda-de 0.1
```

Straight student training produces `.student` artifacts, while distilled student training produces `.distilled_student` artifacts. Both are PyTorch artifact payloads containing model state, config, metadata, and history.

## Experiment Launchers

Shell launchers are grouped by:

```text
experiments/recdistill/      RecDistill launchers
experiments/baseline/        Teacher and baseline student launchers
```

RecDistill examples:

```bash
bash experiments/recdistill/train_distiller.sh <distiller> <dataset> [teacher_framework] [teacher_model|ALL] [student_framework] [student_backbone|SAME|ALL] [gpu] [dry_run]
bash experiments/recdistill/train_single_distiller.sh <distiller> <dataset> <teacher_framework> <teacher_model> [student_framework] [student_backbone|SAME] [gpu] [dry_run]
```

Baseline example:

```bash
bash experiments/baseline/teacher_training.sh [teacher_framework] [teacher_model] <dataset> [gpu] [grid|best]
```

---

# Configs And Experiments

The canonical config package is `config/`.

```text
config/
  dataset/            Dataset definitions
  teacher/            Teacher model defaults
  student/            Student model defaults
  distillation/       Distiller defaults
  optimization/       Optimization defaults
  runtime/            Runtime defaults
  evaluation/         Evaluation defaults
  composites/         Composable templates
  experiments/        Complete experiment configs
```

Complete experiment configs are under:

```text
config/experiments/teacher/
config/experiments/student/
config/experiments/recdistill/
```

On-the-fly generated configs are saved there too.

---

# Evaluation

The evaluation in RecDistillery is done as a top-k ranking, typically @20. 
During student training, the pipeline calculates the top-k recommendations, compares them with the validation/test ground truth, calculates precision, recall, ndcg, and hr and finally saves the best artifact using val.ndcg as selection metric.

The formulas are implemented here: `recdistill/evaluation.py`. Then the metrics are averaged across evaluable users.

Run `scripts/recdistill/evaluate_teacher.py` to evaluate `.teacher` artifacts and `scripts/recdistill/evaluate_students.py` to evaluate `.student` or `.distilled_student` artifacts. Both scripts produce separate JSON/TSV files.

Teacher evaluation example:

```bash
python scripts/recdistill/evaluate_teacher.py \
  --dataset citeulike \
  --framework recbole \
  --model BPRMF \
  --embedding-dim 200 \
  --top-k 20
```

Plain student evaluation example:

```bash
python scripts/recdistill/evaluate_students.py \
  --dataset citeulike \
  --distiller plain \
  --student-framework recbole \
  --student-backbone LGCN \
  --student-embedding-dim 20 \
  --top-k 20
```

Distilled student evaluation example:

```bash
python scripts/recdistill/evaluate_students.py \
  --dataset citeulike \
  --distiller de \
  --teacher-framework recbole \
  --teacher-model BPRMF \
  --student-framework recbole \
  --student-backbone LGCN \
  --student-embedding-dim 20 \
  --top-k 20
```

---

# Results structure

The setup script creates the base runtime directories used by the pipeline, including `results/`, and `data/`.

Teacher artifacts trained with fixed parameters are saved under:

```text
results/teachers/<framework>/<model>/<dataset>/fixed/wei/<framework>_<model>_<dataset>_<embedding_dim>.teacher
```

Teacher artifacts trained with best parameters from Bayesian search are saved under:

```text
results/teachers/<framework>/<model>/<dataset>/best/wei/<framework>_<model>_<dataset>_<embedding_dim>.teacher
```

Imported teacher artifacts are saved under:

```text
results/teachers/<framework>/<model>/<dataset>/imported/<framework>_<model>_<dataset>_<embedding_dim>.teacher
```

Plain student artifacts trained with fixed parameters are saved under:

```text
results/students/<framework>/<model>/<dataset>/fixed/wei/<framework>_<model>_<dataset>_<embedding_dim>.student
```

Plain student artifacts trained with best parameters from Bayesian search are saved under:

```text
results/students/<framework>/<model>/<dataset>/best/wei/<framework>_<model>_<dataset>_<embedding_dim>.student
```

Distilled student artifacts follow the selected strategy namespace:

```text
results/recdistill/<distiller>/<teacher_framework>/<teacher_model>/<student_framework>/<student_model>/<dataset>/<strategy>/wei/<teacher_framework>_<teacher_model>_<student_framework>_<student_model>_<dataset>_<student_embedding_dim>.distilled_student
```

Use `fixed/wei` for runs with fixed parameters and `best/wei` for reruns with best parameters found by Bayesian search. Evaluation metrics go under the sibling `perf/` directory. Imported teachers go under `imported/`.

---

# Timing Analysis

Timing analysis is implemented through tracked student runs. The same run that collects the final metrics also records the values needed to measure:
- total training time,
- average epoch time,
- KD overhead,
- per-epoch training and validation history.

## Main Command

```bash
python3 scripts/recdistill/train_student_from_config.py \
  --config <config_file> \
  --track
```

## Outputs

```text
results/recdistill/<distiller>/<teacher_framework>/<teacher_model>/<student_framework>/<student_model>/<dataset>/tracked/<run_id>/
```

