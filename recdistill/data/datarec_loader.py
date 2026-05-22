from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import get_config_loader
from recdistill.data.interactions import InteractionDataset


USER_COLUMNS = ("user", "userId", "user_id", "uid", 0)
ITEM_COLUMNS = ("item", "itemId", "item_id", "iid", 1)
RATING_COLUMNS = ("rating", "ratings", "score", 2)
TIMESTAMP_COLUMNS = ("timestamp", "time", "ts", 3)
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class LoadedSplit:
    name: str
    path: Path
    frame: pd.DataFrame
    backend: str


@dataclass(frozen=True)
class EncodedDataset:
    frames: dict[str, pd.DataFrame]
    num_users: int
    num_items: int
    backend: str


def datarec_available() -> bool:
    return _datarec_reader() is not None


def load_split_frame(
    dataset_name: str,
    split_name: str,
    *,
    repo_root: Path | str = ".",
    use_datarec: bool = True,
) -> LoadedSplit:
    """Load one dataset split through DataRec, falling back to pandas if unavailable."""
    path = _split_path(dataset_name=dataset_name, split_name=split_name, repo_root=repo_root)
    columns = _column_names(dataset_name)
    return load_frame_from_path(path, dataset_name=dataset_name, split_name=split_name, columns=columns, use_datarec=use_datarec)


def load_frame_from_path(
    path: Path | str,
    *,
    dataset_name: str,
    split_name: str,
    columns: list[str] | None = None,
    use_datarec: bool = True,
) -> LoadedSplit:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} split not found at {path}")
    columns = columns or ["userId", "itemId", "rating"]

    reader = _datarec_reader() if use_datarec else None
    if reader is not None:
        data = reader(
            str(path),
            sep="\t",
            user_col=columns[0],
            item_col=columns[1],
            rating_col=columns[2] if len(columns) > 2 else None,
            timestamp_col=columns[3] if len(columns) > 3 else None,
            header=None,
            cols=columns,
            dataset_name=dataset_name,
            version_name=split_name,
        )
        return LoadedSplit(
            name=split_name,
            path=path,
            frame=_datarec_to_frame(data),
            backend="datarec",
        )

    frame = pd.read_csv(path, sep="\t", header=None, names=columns)
    return LoadedSplit(name=split_name, path=path, frame=frame, backend="pandas")


def load_train_dataset(
    dataset_name: str,
    teacher_num_users: int,
    teacher_num_items: int,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
) -> tuple[InteractionDataset, int]:
    train_dict, dropped = load_ground_truth_split(
        dataset_name=dataset_name,
        split_name="train",
        num_users=teacher_num_users,
        num_items=teacher_num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    return (
        InteractionDataset.from_train_dict(
            train_dict=train_dict,
            num_users=teacher_num_users,
            num_items=teacher_num_items,
        ),
        dropped,
    )


def load_eval_split(
    dataset_name: str,
    split_name: str,
    teacher_num_users: int,
    teacher_num_items: int,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
) -> tuple[dict[int, set[int]], int]:
    return load_ground_truth_split(
        dataset_name=dataset_name,
        split_name=split_name,
        num_users=teacher_num_users,
        num_items=teacher_num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def load_interaction_dataset(
    dataset_name: str,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
    num_users: int | None = None,
    num_items: int | None = None,
) -> InteractionDataset:
    if user_mapping is None and item_mapping is None:
        encoded = load_encoded_dataset(dataset_name)
        if num_users is None:
            num_users = encoded.num_users
        if num_items is None:
            num_items = encoded.num_items
        split, _ = _encoded_frame_to_split(encoded.frames["train"], num_users=num_users, num_items=num_items)
    else:
        split, _ = _frame_to_split(
            load_split_frame(dataset_name, "train").frame,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            num_users=num_users,
            num_items=num_items,
        )
        if num_users is None:
            num_users = max(split.keys(), default=-1) + 1
        if num_items is None:
            num_items = max((item for items in split.values() for item in items), default=-1) + 1
    return InteractionDataset.from_train_dict(split, num_users=num_users, num_items=num_items)


def resolve_teacher_dataset_mappings(
    metadata: dict[str, Any] | None,
    *,
    dataset_name: str,
) -> tuple[dict[int, int] | dict[str, int] | None, dict[int, int] | dict[str, int] | None, str]:
    metadata = metadata or {}
    user_mapping = metadata.get("public_to_local_user_id")
    item_mapping = metadata.get("public_to_local_item_id")
    if not user_mapping and not item_mapping:
        return None, None, "datarec"

    if (
        _looks_like_identity_mapping(user_mapping)
        and _looks_like_identity_mapping(item_mapping)
        and _dataset_has_sparse_numeric_ids(dataset_name)
    ):
        return None, None, "datarec_identity_teacher_mapping_ignored"

    return user_mapping, item_mapping, "teacher_metadata"


def load_ground_truth_split(
    dataset_name: str,
    split_name: str,
    num_users: int | None = None,
    num_items: int | None = None,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
) -> tuple[dict[int, set[int]], int]:
    if user_mapping is None and item_mapping is None:
        encoded = load_encoded_dataset(dataset_name)
        normalized_split = "val" if split_name == "validation" else split_name
        if normalized_split not in encoded.frames:
            raise ValueError(f"Unknown split: {split_name}")
        return _encoded_frame_to_split(
            encoded.frames[normalized_split],
            num_users=num_users,
            num_items=num_items,
        )

    loaded = load_split_frame(dataset_name, split_name)
    return _frame_to_split(
        loaded.frame,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        num_users=num_users,
        num_items=num_items,
    )


def load_encoded_dataset(dataset_name: str, *, repo_root: Path | str = ".") -> EncodedDataset:
    """Load all splits and apply one shared DataRec user/item encoding."""
    return _load_encoded_dataset_cached(dataset_name, str(Path(repo_root)))


@lru_cache(maxsize=None)
def _load_encoded_dataset_cached(dataset_name: str, repo_root: str) -> EncodedDataset:
    loaded_splits = {
        split: load_split_frame(dataset_name, split, repo_root=repo_root, use_datarec=True)
        for split in SPLIT_ORDER
    }
    frames = {split: loaded.frame.copy() for split, loaded in loaded_splits.items()}
    encoded_frames, num_users, num_items = _encode_split_frames_with_datarec(frames)
    backends = {loaded.backend for loaded in loaded_splits.values()}
    backend = "datarec" if backends == {"datarec"} else "+".join(sorted(backends))
    return EncodedDataset(frames=encoded_frames, num_users=num_users, num_items=num_items, backend=backend)


def _encode_split_frames_with_datarec(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int, int]:
    encoder_cls = _datarec_incremental_encoder() or _FallbackIncrementalEncoder
    user_encoder = encoder_cls()
    item_encoder = encoder_cls()
    encoded: dict[str, pd.DataFrame] = {}

    for split in SPLIT_ORDER:
        frame = frames[split].copy()
        user_col = _pick_column(frame.columns, USER_COLUMNS)
        item_col = _pick_column(frame.columns, ITEM_COLUMNS)
        frame[user_col] = user_encoder.encode_many(frame[user_col].tolist())
        frame[item_col] = item_encoder.encode_many(frame[item_col].tolist())
        frame[user_col] = frame[user_col].astype("int64")
        frame[item_col] = frame[item_col].astype("int64")
        encoded[split] = frame

    return encoded, len(user_encoder), len(item_encoder)


def _encoded_frame_to_split(
    frame: pd.DataFrame,
    *,
    num_users: int | None,
    num_items: int | None,
) -> tuple[dict[int, set[int]], int]:
    user_col = _pick_column(frame.columns, USER_COLUMNS)
    item_col = _pick_column(frame.columns, ITEM_COLUMNS)
    split: dict[int, set[int]] = {}
    dropped = 0

    for user_raw, item_raw in frame[[user_col, item_col]].itertuples(index=False, name=None):
        try:
            user = int(user_raw)
            item = int(item_raw)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if user < 0 or item < 0:
            dropped += 1
            continue
        if num_users is not None and user >= num_users:
            dropped += 1
            continue
        if num_items is not None and item >= num_items:
            dropped += 1
            continue
        split.setdefault(user, set()).add(item)
    return split, dropped


class _FallbackIncrementalEncoder:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self._forward: dict[Any, int] = {}
        self._reverse: list[Any] = []

    def __len__(self) -> int:
        return len(self._forward)

    def encode_one(self, key: Any) -> int:
        if key in self._forward:
            return self._forward[key]
        idx = self.offset + len(self._forward)
        self._forward[key] = idx
        self._reverse.append(key)
        return idx

    def encode_many(self, values: Iterable[Any]) -> list[int]:
        return [self.encode_one(value) for value in values]


def resolve_local_id(raw_value: Any, mapping: dict[int, int] | dict[str, int] | None) -> int | None:
    candidates: list[Any] = [raw_value]
    raw_string = str(raw_value).strip()
    candidates.append(raw_string)
    try:
        parsed_int = int(raw_string)
    except (TypeError, ValueError):
        parsed_int = None
    if parsed_int is not None:
        candidates.extend([parsed_int, str(parsed_int)])

    if mapping is not None:
        mapped_value = None
        for candidate in candidates:
            if candidate in mapping:
                mapped_value = mapping[candidate]
                break
    else:
        mapped_value = parsed_int if parsed_int is not None else raw_string

    if mapped_value is None:
        return None
    try:
        return int(mapped_value)
    except (TypeError, ValueError):
        return None


def _frame_to_split(
    frame: pd.DataFrame,
    *,
    user_mapping: dict[int, int] | dict[str, int] | None,
    item_mapping: dict[int, int] | dict[str, int] | None,
    num_users: int | None,
    num_items: int | None,
) -> tuple[dict[int, set[int]], int]:
    user_col = _pick_column(frame.columns, USER_COLUMNS)
    item_col = _pick_column(frame.columns, ITEM_COLUMNS)
    split: dict[int, set[int]] = {}
    dropped = 0
    user_encoder: dict[Any, int] = {}
    item_encoder: dict[Any, int] = {}

    for row in frame[[user_col, item_col]].itertuples(index=False, name=None):
        user = _resolve_or_encode(row[0], user_mapping, user_encoder, prefer_numeric=num_users is not None)
        item = _resolve_or_encode(row[1], item_mapping, item_encoder, prefer_numeric=num_items is not None)
        if user is None or item is None:
            dropped += 1
            continue
        if user < 0 or item < 0:
            dropped += 1
            continue
        if num_users is not None and user >= num_users:
            dropped += 1
            continue
        if num_items is not None and item >= num_items:
            dropped += 1
            continue
        split.setdefault(user, set()).add(item)
    return split, dropped


def _resolve_or_encode(
    raw_value: Any,
    mapping: dict[int, int] | dict[str, int] | None,
    encoder: dict[Any, int],
    prefer_numeric: bool,
) -> int | None:
    resolved = resolve_local_id(raw_value, mapping)
    if resolved is not None:
        return resolved
    if mapping is not None or prefer_numeric:
        return resolved

    key = str(raw_value).strip()
    if key == "":
        return None
    if key not in encoder:
        encoder[key] = len(encoder)
    return encoder[key]


def _looks_like_identity_mapping(mapping: Any) -> bool:
    if not isinstance(mapping, dict) or not mapping:
        return False
    checked = 0
    for key, value in mapping.items():
        try:
            if int(str(key).strip()) != int(value):
                return False
        except (TypeError, ValueError):
            return False
        checked += 1
        if checked >= 1000:
            break
    return True


def _dataset_has_sparse_numeric_ids(dataset_name: str) -> bool:
    for split_name in SPLIT_ORDER:
        try:
            frame = load_split_frame(dataset_name, split_name).frame
        except Exception:
            continue
        for candidates in (USER_COLUMNS, ITEM_COLUMNS):
            column = _pick_column(frame.columns, candidates)
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or values.empty:
                continue
            max_id = int(values.max())
            unique_count = int(values.nunique())
            if max_id + 1 > unique_count:
                return True
    return False


def _infer_numeric_shape_from_splits(dataset_name: str) -> tuple[int | None, int | None]:
    try:
        encoded = load_encoded_dataset(dataset_name)
    except Exception:
        return None, None
    return encoded.num_users, encoded.num_items


def _split_path(dataset_name: str, split_name: str, repo_root: Path | str) -> Path:
    dataset = get_config_loader().load_dataset_config(dataset_name)
    if split_name == "train":
        raw_path = dataset.train_path
    elif split_name in {"val", "validation"}:
        raw_path = dataset.validation_path
    elif split_name == "test":
        raw_path = dataset.test_path
    else:
        raise ValueError(f"Unknown split: {split_name}")
    if raw_path is None:
        raise ValueError(f"Dataset {dataset_name} has no configured {split_name} path.")

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.exists():
        raise FileNotFoundError(f"{split_name} split not found at {path}")
    return path


def _column_names(dataset_name: str) -> list[str]:
    dataset = get_config_loader().load_dataset_config(dataset_name)
    configured = getattr(dataset, "column_names", None)
    if configured:
        names = [str(name) for name in configured]
    else:
        names = ["userId", "itemId", "rating"]
    return names[:3] if len(names) > 3 else names


def _datarec_reader():
    try:
        from datarec.io import read_transactions_tabular

        return read_transactions_tabular
    except Exception:
        pass
    try:
        from datarec.io.readers.transactions.tabular import read_transactions_tabular

        return read_transactions_tabular
    except Exception:
        return None


def _datarec_incremental_encoder():
    try:
        from datarec.data.utils import IncrementalEncoder

        return IncrementalEncoder
    except Exception:
        return None


def _datarec_to_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    for attr_path in (
        ("data",),
        ("df",),
        ("dataframe",),
        ("transactions",),
        ("interactions",),
        ("rawdata", "transactions"),
        ("rawdata", "interactions"),
        ("rawdata", "data"),
        ("rawdata", "df"),
        ("raw_data", "transactions"),
        ("raw_data", "interactions"),
        ("raw_data", "data"),
        ("raw_data", "df"),
    ):
        value = _get_nested_attr(data, attr_path)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, (list, tuple, dict)):
            return pd.DataFrame(value)

    for method in ("to_pandas", "to_dataframe", "as_dataframe"):
        candidate = getattr(data, method, None)
        if callable(candidate):
            value = candidate()
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if isinstance(value, (list, tuple, dict)):
                return pd.DataFrame(value)

    raise TypeError(
        "Unable to extract a pandas DataFrame from DataRec output. "
        f"Unsupported object type: {type(data)!r}"
    )


def _get_nested_attr(value: Any, attr_path: Iterable[str]) -> Any:
    current = value
    for attr in attr_path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _pick_column(columns: Iterable[Any], candidates: Iterable[Any]) -> Any:
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, int) and 0 <= candidate < len(cols):
            return cols[candidate]
    raise ValueError(f"Unable to find any of {list(candidates)} in columns {cols}")
