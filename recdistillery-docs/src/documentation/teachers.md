# Teachers

Teachers can be trained inside RecDistillery or imported from external
artifacts. In both cases, the runtime representation is a framework-neutral
`TeacherState` saved as a `.teacher` artifact.

## Teacher Training

Train a native teacher with:

```bash
python scripts/teacher_training/teacher_training.py \
  --framework recbole \
  --model BPRMF \
  --dataset citeulike
```

Complete experiment configs are stored in:

```text
config/experiments/teacher/
```

## Teacher Import

External teachers are converted through `scripts/recdistill/import_teacher.py`.
The import system supports generic checkpoints, prediction JSON exports, and
RecBole `.pth` checkpoints.

```bash
python scripts/recdistill/import_teacher.py --list-adapters
```

## Teacher State

`TeacherState` is the normalized runtime object used after training or import.
It hides framework-specific checkpoint formats and gives distillers a single
interface.

| Object | Represents | Key attributes |
| --- | --- | --- |
| `TeacherState` | A trained or imported teacher. | `user_embeddings`, `item_embeddings`, `metadata`, `scorer` |
| `PrecomputedScoresScorer` | A dense user-item score matrix. | `scores` |
| `PrecomputedTopKScorer` | A sparse top-k ranking export. | `topk_items`, `topk_scores`, `fill_value`, `num_items_override` |
| `TeacherScorer` | Protocol for scorer-only teachers. | `to`, `score_items_for_user` |

::: recdistill.teachers.state

## Teacher Sources

`TeacherSource` describes where an external teacher comes from and which hints
the adapter registry can use during import.

| Attribute | Meaning |
| --- | --- |
| `path` | Main artifact path, such as `.teacher`, `.pth`, `.pt`, `.ckpt`, or `.json`. |
| `framework` | Framework hint used for adapter resolution. |
| `format` | File/representation hint used for adapter resolution. |
| `model_name` | Optional model name saved into teacher metadata. |
| `adapter` | Explicit custom adapter import path. |
| `metadata` | Dataset, id mapping, provenance, and any additional import metadata. |

::: recdistill.teachers.source

## Loading And Serialization

::: recdistill.teachers.loaders

::: recdistill.teachers.serialization

## Adapter Registry

The registry maps teacher sources to import adapters. Callers can register new
adapters, list available keys, resolve the adapter for a source, or load a
`TeacherState` directly.

| Function | Purpose |
| --- | --- |
| `register_teacher_adapter` | Adds an adapter and optional aliases. |
| `available_teacher_adapters` | Lists registered adapter keys. |
| `resolve_teacher_adapter` | Selects the adapter that can load a `TeacherSource`. |
| `load_teacher_state` | Resolves and loads a `TeacherState`. |

::: recdistill.teachers.registry

## Import Adapters

Import adapters convert external artifacts into the shared `TeacherState`
format.

| Adapter | Accepted sources | Output representation |
| --- | --- | --- |
| `CheckpointAdapter` | `.teacher`, `.pt`, `.pth`, `.ckpt` payloads. | Embeddings, dense scores, or top-k scorer. |
| `PredictionsJsonAdapter` | JSON prediction rows or column-oriented prediction exports. | `PrecomputedTopKScorer`. |
| `RecBolePthAdapter` | RecBole `.pth` checkpoints with embedding tensors. | Embedding-backed `TeacherState`. |

::: recdistill.teachers.adapters.checkpoint

::: recdistill.teachers.adapters.predictions_json

::: recdistill.teachers.adapters.recbole_pth
