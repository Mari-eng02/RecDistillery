# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data abstractions and data set access.
"""

from __future__ import annotations

from recommenders.lenskit.diagnostics import FieldError

from ._adapt import from_interactions_df
from ._attributes import EntityAttribute
from ._batches import BatchedRange
from ._builder import DatasetBuilder
from ._collection import (
    GenericKey,
    ItemListCollection,
    ItemListCollector,
    ListILC,
    MutableItemListCollection,
    QueryIDKey,
    UserIDKey,
    key_dict,
)
from ._container import DataContainer
from ._dataset import Dataset
from ._entities import EntitySet
from ._flatten import flatten_dict, unflatten_dict
from ._items import ItemList
from ._query import QueryInput, QueryItemSource, RecQuery
from ._relationships import MatrixRelationshipSet, RelationshipSet
from ._vocab import Vocabulary
from .types import ID, NPID, FeedbackType

__all__ = [
    "Dataset",
    "DatasetBuilder",
    "DataContainer",
    "EntitySet",
    "RelationshipSet",
    "MatrixRelationshipSet",
    "EntityAttribute",
    "FieldError",
    "from_interactions_df",
    "ID",
    "NPID",
    "FeedbackType",
    "ItemList",
    "ItemListCollection",
    "ItemListCollector",
    "MutableItemListCollection",
    "ListILC",
    "UserIDKey",
    "QueryIDKey",
    "GenericKey",
    "key_dict",
    "flatten_dict",
    "unflatten_dict",
    "Vocabulary",
    "RecQuery",
    "QueryInput",
    "QueryItemSource",
    "BatchedRange",
]
