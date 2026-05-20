from __future__ import annotations

from dataclasses import dataclass

import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from recdistill.data.interactions import InteractionDataset
from recdistill.registry import canonical_model_name


@dataclass
class FrameworkBatchOutput:
    pos_scores: torch.Tensor
    neg_scores: torch.Tensor
    base_loss: torch.Tensor


def _bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    *,
    l2_reg: float = 0.0,
    embeddings: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    if l2_reg > 0 and embeddings:
        reg = sum(embedding.norm(2).pow(2) for embedding in embeddings)
        loss = loss + float(l2_reg) * 0.5 * reg / max(1, pos_scores.numel())
    return loss


class RecBoleDatasetAdapter:
    def __init__(self, dataset: InteractionDataset):
        self.dataset = dataset
        self.uid_field = "user_id"
        self.iid_field = "item_id"
        self.inter_num = len(dataset.interactions)
        users = torch.tensor([user for user, _ in dataset.interactions], dtype=torch.long)
        items = torch.tensor([item for _, item in dataset.interactions], dtype=torch.long)
        self.inter_feat = {
            self.uid_field: users,
            self.iid_field: items,
        }

    def num(self, field: str) -> int:
        if field == "user_id":
            return self.dataset.num_users
        if field == "item_id":
            return self.dataset.num_items
        raise KeyError(f"Unsupported RecBole field: {field}")

    def inter_matrix(self, form: str = "coo"):
        rows = [user for user, _ in self.dataset.interactions]
        cols = [item for _, item in self.dataset.interactions]
        data = [1.0] * len(rows)
        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.dataset.num_users, self.dataset.num_items),
            dtype="float32",
        )
        if form == "coo":
            return matrix
        if form == "csr":
            return matrix.tocsr()
        raise ValueError(f"Unsupported sparse matrix form: {form}")


class RecBoleBackboneAdapter(nn.Module):
    _GRAPH_BACKBONES = {"LGCN", "NGCF", "DGCF", "SGL", "SPECTRALCF"}

    def __init__(
        self,
        *,
        backbone: str,
        dataset: InteractionDataset,
        embedding_dim: int,
        l2_reg: float = 0.0,
        lightgcn_layers: int = 2,
        neumf_mlp_dims: tuple[int, ...] = (64, 32, 16, 8),
        neumf_dropout: float = 0.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.backbone = canonical_model_name(backbone)
        self.dataset = dataset
        self.embedding_dim = int(embedding_dim)
        self.l2_reg = float(l2_reg)
        self.device_name = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._build_config(
            embedding_dim=int(embedding_dim),
            l2_reg=float(l2_reg),
            lightgcn_layers=int(lightgcn_layers),
            neumf_mlp_dims=tuple(int(v) for v in neumf_mlp_dims),
            neumf_dropout=float(neumf_dropout),
        )
        self.recbole_dataset = RecBoleDatasetAdapter(dataset)
        self.model = self._build_model()
        self.model.to(self.device_name)

    @property
    def can_score_items_together(self) -> bool:
        return True

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> FrameworkBatchOutput:
        users = users.to(self.device_name)
        pos_items = pos_items.to(self.device_name)
        neg_items = neg_items.to(self.device_name)
        interaction = {
            "user_id": users,
            "item_id": pos_items,
            "neg_item_id": neg_items,
        }
        if self.backbone == "NMF":
            pos_scores = self.model.forward(users, pos_items)
            neg_scores = self.model.forward(users, neg_items)
            base_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            if self.l2_reg > 0:
                user_emb = self.model.user_mf_embedding(users)
                pos_emb = self.model.item_mf_embedding(pos_items)
                neg_emb = self.model.item_mf_embedding(neg_items)
                reg = 0.5 * (
                    user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
                ) / max(1, users.size(0))
                base_loss = base_loss + self.l2_reg * reg
        elif self.backbone in self._GRAPH_BACKBONES:
            user_table, item_table = self._graph_embeddings()
            pos_scores = (user_table[users] * item_table[pos_items]).sum(dim=-1)
            neg_scores = (user_table[users] * item_table[neg_items]).sum(dim=-1)
            base_loss = self._model_loss(interaction)
        else:
            if self.backbone == "LINE":
                user_e = self.model.user_embedding(users)
                pos_e = self.model.item_embedding(pos_items)
                neg_e = self.model.item_embedding(neg_items)
            else:
                user_e, pos_e = self.model.forward(users, pos_items)
                neg_e = self.model.get_item_embedding(neg_items)
            pos_scores = (user_e * pos_e).sum(dim=-1)
            neg_scores = (user_e * neg_e).sum(dim=-1)
            base_loss = self._model_loss(interaction)
            if self.l2_reg > 0:
                reg = 0.5 * (
                    user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)
                ) / max(1, users.size(0))
                base_loss = base_loss + self.l2_reg * reg

        return FrameworkBatchOutput(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            base_loss=base_loss,
        )

    def compute_base_loss(self, batch_output: FrameworkBatchOutput) -> torch.Tensor:
        return batch_output.base_loss

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        users = users.to(self.device_name)
        items = items.to(self.device_name)
        if items.ndim == 1:
            return self._score_pairs(users, items)
        expanded_users = users.unsqueeze(-1).expand(-1, items.size(1))
        return self._score_pairs(expanded_users, items)

    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        items = torch.arange(num_items, dtype=torch.long, device=self.device_name)
        users = torch.full_like(items, int(user))
        return self.score_items(users, items)

    def get_all_user_embeddings(self) -> torch.Tensor:
        if self.backbone in self._GRAPH_BACKBONES:
            user_table, _ = self._graph_embeddings()
            return user_table
        if self.backbone == "NMF":
            return self.model.user_mf_embedding.weight
        return self.model.user_embedding.weight

    def get_all_item_embeddings(self) -> torch.Tensor:
        if self.backbone in self._GRAPH_BACKBONES:
            _, item_table = self._graph_embeddings()
            return item_table
        if self.backbone == "NMF":
            return self.model.item_mf_embedding.weight
        return self.model.item_embedding.weight

    def _score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.backbone == "NMF":
            return torch.sigmoid(self.model.forward(users, items))
        if self.backbone in self._GRAPH_BACKBONES:
            user_table, item_table = self._graph_embeddings()
            return (user_table[users] * item_table[items]).sum(dim=-1)
        if self.backbone == "LINE":
            user_e = self.model.user_embedding(users)
            item_e = self.model.item_embedding(items)
            return (user_e * item_e).sum(dim=-1)
        user_e = self.model.get_user_embedding(users)
        item_e = self.model.get_item_embedding(items)
        return (user_e * item_e).sum(dim=-1)

    def _graph_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone == "SGL":
            return self.model.forward(self.model.train_graph)
        return self.model.forward()

    def _model_loss(self, interaction: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = self.model.calculate_loss(interaction)
        if isinstance(loss, (tuple, list)):
            tensor_losses = [value for value in loss if torch.is_tensor(value)]
            if tensor_losses:
                return sum(tensor_losses)
        return loss

    def _build_config(
        self,
        *,
        embedding_dim: int,
        l2_reg: float,
        lightgcn_layers: int,
        neumf_mlp_dims: tuple[int, ...],
        neumf_dropout: float,
    ) -> dict:
        config = {
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "NEG_PREFIX": "neg_",
            "LABEL_FIELD": "label",
            "device": self.device_name,
            "embedding_size": embedding_dim,
        }
        if self.backbone == "LGCN":
            config.update(
                {
                    "n_layers": lightgcn_layers,
                    "reg_weight": l2_reg,
                    "require_pow": False,
                }
            )
        if self.backbone == "LINE":
            config.update(
                {
                    "order": 1,
                    "second_order_loss_weight": 1.0,
                }
            )
        if self.backbone == "NGCF":
            config.update(
                {
                    "hidden_size_list": [embedding_dim] * max(1, lightgcn_layers),
                    "node_dropout": 0.0,
                    "message_dropout": neumf_dropout,
                    "reg_weight": l2_reg,
                }
            )
        if self.backbone == "DGCF":
            n_factors = 4 if embedding_dim % 4 == 0 else 1
            config.update(
                {
                    "n_factors": n_factors,
                    "n_iterations": 2,
                    "n_layers": lightgcn_layers,
                    "reg_weight": l2_reg,
                    "cor_weight": 0.0,
                    "train_batch_size": 512,
                }
            )
        if self.backbone == "SGL":
            config.update(
                {
                    "n_layers": lightgcn_layers,
                    "type": "ED",
                    "drop_ratio": 0.1,
                    "ssl_tau": 0.2,
                    "reg_weight": l2_reg,
                    "ssl_weight": 0.1,
                }
            )
        if self.backbone == "SPECTRALCF":
            config.update(
                {
                    "n_layers": lightgcn_layers,
                    "reg_weight": l2_reg,
                }
            )
        if self.backbone == "NMF":
            config.update(
                {
                    "mf_embedding_size": embedding_dim,
                    "mlp_embedding_size": embedding_dim,
                    "mlp_hidden_size": list(neumf_mlp_dims),
                    "dropout_prob": neumf_dropout,
                    "mf_train": True,
                    "mlp_train": True,
                    "use_pretrain": False,
                    "mf_pretrain_path": None,
                    "mlp_pretrain_path": None,
                }
            )
        return config

    def _build_model(self) -> nn.Module:
        if self.backbone == "LINE":
            from recommenders.recbole.model.general_recommender.line import LINE

            return LINE(self.config, self.recbole_dataset)
        if self.backbone == "LGCN":
            from recommenders.recbole.model.general_recommender.lightgcn import LightGCN

            return LightGCN(self.config, self.recbole_dataset)
        if self.backbone == "NGCF":
            from recommenders.recbole.model.general_recommender.ngcf import NGCF

            return NGCF(self.config, self.recbole_dataset)
        if self.backbone == "DGCF":
            from recommenders.recbole.model.general_recommender.dgcf import DGCF

            return DGCF(self.config, self.recbole_dataset)
        if self.backbone == "SGL":
            from recommenders.recbole.model.general_recommender.sgl import SGL

            model = SGL(self.config, self.recbole_dataset)
            model.graph_construction()
            return model
        if self.backbone == "SPECTRALCF":
            from recommenders.recbole.model.general_recommender.spectralcf import SpectralCF

            return SpectralCF(self.config, self.recbole_dataset)
        if self.backbone == "NMF":
            from recommenders.recbole.model.general_recommender.neumf import NeuMF

            return NeuMF(self.config, self.recbole_dataset)
        from recommenders.recbole.model.general_recommender.bpr import BPR

        return BPR(self.config, self.recbole_dataset)


class ElliotBackboneAdapter(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        dataset: InteractionDataset,
        embedding_dim: int,
        l2_reg: float = 0.0,
        lightgcn_layers: int = 2,
        neumf_mlp_dims: tuple[int, ...] = (64, 32, 16, 8),
        neumf_dropout: float = 0.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.backbone = canonical_model_name(backbone)
        self.dataset = dataset
        self.embedding_dim = int(embedding_dim)
        self.l2_reg = float(l2_reg)
        self.device_name = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(
            lightgcn_layers=int(lightgcn_layers),
            neumf_mlp_dims=neumf_mlp_dims,
            neumf_dropout=neumf_dropout,
        )
        self.model.to(self.device_name)
        if hasattr(self.model, "device"):
            self.model.device = self.device_name

    @property
    def can_score_items_together(self) -> bool:
        return False

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> FrameworkBatchOutput:
        pos_scores = self._score_pairs(users, pos_items)
        neg_scores = self._score_pairs(users, neg_items)
        base_loss = _bpr_loss(
            pos_scores,
            neg_scores,
            l2_reg=self.l2_reg,
            embeddings=self._regularized_embeddings(users, pos_items, neg_items),
        )
        return FrameworkBatchOutput(pos_scores=pos_scores, neg_scores=neg_scores, base_loss=base_loss)

    def compute_base_loss(self, batch_output: FrameworkBatchOutput) -> torch.Tensor:
        return batch_output.base_loss

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if items.ndim == 1:
            return self._score_pairs(users, items)
        expanded_users = users.unsqueeze(-1).expand(-1, items.size(1))
        return self._score_pairs(expanded_users, items)

    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        items = torch.arange(num_items, dtype=torch.long, device=self.device_name)
        users = torch.full_like(items, int(user))
        return self.score_items(users, items)

    def get_all_user_embeddings(self) -> torch.Tensor:
        self._ensure_supported_for_training()
        return self.model.user_mf_embedding.weight

    def get_all_item_embeddings(self) -> torch.Tensor:
        self._ensure_supported_for_training()
        return self.model.item_mf_embedding.weight

    def _score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        users = users.to(self.device_name)
        items = items.to(self.device_name)
        if self.backbone == "NMF":
            return self.model.forward((users, items), training=True).squeeze(-1)
        self._raise_unsupported()

    def _regularized_embeddings(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.backbone == "NMF":
            return (
                self.model.user_mf_embedding(users.to(self.device_name)),
                self.model.item_mf_embedding(pos_items.to(self.device_name)),
                self.model.item_mf_embedding(neg_items.to(self.device_name)),
            )
        self._raise_unsupported()

    def _build_model(
        self,
        *,
        lightgcn_layers: int,
        neumf_mlp_dims: tuple[int, ...],
        neumf_dropout: float,
    ) -> nn.Module:
        del lightgcn_layers
        if self.backbone == "NMF":
            from recommenders.elliot.neural.NeuMF.neural_matrix_factorization_torch_model import (
                NeuralMatrixFactorizationTorchModel,
            )

            return NeuralMatrixFactorizationTorchModel(
                num_users=self.dataset.num_users,
                num_items=self.dataset.num_items,
                embed_mf_size=self.embedding_dim,
                embed_mlp_size=self.embedding_dim,
                mlp_hidden_size=tuple(neumf_mlp_dims),
                dropout=float(neumf_dropout),
                is_mf_train=True,
                is_mlp_train=True,
                learning_rate=0.001,
            )
        self._raise_unsupported()

    def _ensure_supported_for_training(self) -> None:
        if self.backbone != "NMF":
            self._raise_unsupported()

    def _raise_unsupported(self):
        raise NotImplementedError(
            f"Unsupported Elliot backbone for RecDistill adapter: {self.backbone}. "
            "Elliot BPRMF is NumPy/manual-update and Elliot LightGCN is TensorFlow/Keras, "
            "so they are not supported as true PyTorch adapters. Supported: NeuMF."
        )


class LensKitBackboneAdapter(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        dataset: InteractionDataset,
        embedding_dim: int,
        l2_reg: float = 0.0,
        lightgcn_layers: int = 2,
        neumf_mlp_dims: tuple[int, ...] = (64, 32, 16, 8),
        neumf_dropout: float = 0.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.backbone = canonical_model_name(backbone)
        self.dataset = dataset
        self.embedding_dim = int(embedding_dim)
        self.l2_reg = float(l2_reg)
        self.device_name = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(
            lightgcn_layers=int(lightgcn_layers),
            neumf_mlp_dims=neumf_mlp_dims,
            neumf_dropout=float(neumf_dropout),
        )

    @property
    def can_score_items_together(self) -> bool:
        return self.backbone != "NMF"

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> FrameworkBatchOutput:
        pos_scores = self._score_pairs(users, pos_items)
        neg_scores = self._score_pairs(users, neg_items)
        base_loss = _bpr_loss(
            pos_scores,
            neg_scores,
            l2_reg=self.l2_reg,
            embeddings=self._regularized_embeddings(users, pos_items, neg_items),
        )
        return FrameworkBatchOutput(pos_scores=pos_scores, neg_scores=neg_scores, base_loss=base_loss)

    def compute_base_loss(self, batch_output: FrameworkBatchOutput) -> torch.Tensor:
        return batch_output.base_loss

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if items.ndim == 1:
            return self._score_pairs(users, items)
        expanded_users = users.unsqueeze(-1).expand(-1, items.size(1))
        return self._score_pairs(expanded_users, items)

    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        items = torch.arange(num_items, dtype=torch.long, device=self.device_name)
        users = torch.full_like(items, int(user))
        return self.score_items(users, items)

    def get_all_user_embeddings(self) -> torch.Tensor:
        if self.backbone == "LGCN":
            embeddings = self.model.get_embedding(self.edge_index)
            return embeddings[: self.dataset.num_users]
        if self.backbone == "BPRMF":
            return self.model.u_embed.weight
        self._raise_unsupported()

    def get_all_item_embeddings(self) -> torch.Tensor:
        if self.backbone == "LGCN":
            embeddings = self.model.get_embedding(self.edge_index)
            return embeddings[self.dataset.num_users :]
        if self.backbone == "BPRMF":
            return self.model.i_embed.weight
        self._raise_unsupported()

    def _score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        users = users.to(self.device_name)
        items = items.to(self.device_name)
        if self.backbone == "BPRMF":
            return self.model(users, items)
        original_shape = users.shape
        edge_label_index = torch.stack(
            [users.reshape(-1), self._item_nodes(items).reshape(-1)],
            dim=0,
        )
        scores = self.model(self.edge_index, edge_label_index)
        return scores.reshape(original_shape)

    def _regularized_embeddings(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        users = users.to(self.device_name)
        pos_items = pos_items.to(self.device_name)
        neg_items = neg_items.to(self.device_name)
        if self.backbone == "LGCN":
            return (
                self.model.embedding.weight[users],
                self.model.embedding.weight[self._item_nodes(pos_items)],
                self.model.embedding.weight[self._item_nodes(neg_items)],
            )
        if self.backbone == "BPRMF":
            return (
                self.model.u_embed(users),
                self.model.i_embed(pos_items),
                self.model.i_embed(neg_items),
            )
        self._raise_unsupported()

    def _build_model(
        self,
        *,
        lightgcn_layers: int,
        neumf_mlp_dims: tuple[int, ...],
        neumf_dropout: float,
    ) -> nn.Module:
        if self.backbone == "BPRMF":
            from recommenders.lenskit.flexmf._model import FlexMFModel

            rng = torch.Generator(device="cpu")
            model = FlexMFModel(
                self.embedding_dim,
                self.dataset.num_users,
                self.dataset.num_items,
                rng,
                user_bias=False,
                item_bias=False,
                layers=0,
            )
            nn.init.xavier_normal_(model.u_embed.weight)
            nn.init.xavier_normal_(model.i_embed.weight)
            return model.to(self.device_name)
        if self.backbone == "LGCN":
            from recommenders.lenskit.graphs.lightgcn import LightGCN

            self.edge_index = self._edge_index().to(self.device_name)
            model = LightGCN(
                num_nodes=self.dataset.num_users + self.dataset.num_items,
                embedding_dim=self.embedding_dim,
                num_layers=lightgcn_layers,
            )
            return model.to(self.device_name)
        self._raise_unsupported()

    def _edge_index(self) -> torch.Tensor:
        rows: list[int] = []
        cols: list[int] = []
        for user, item in self.dataset.interactions:
            item_node = self.dataset.num_users + item
            rows.extend([user, item_node])
            cols.extend([item_node, user])
        return torch.tensor([rows, cols], dtype=torch.long)

    def _item_nodes(self, items: torch.Tensor) -> torch.Tensor:
        return items + self.dataset.num_users

    def _raise_unsupported(self):
        raise NotImplementedError(
            f"Unsupported LensKit backbone for RecDistill adapter: {self.backbone}. "
            "LensKit does not provide a native NeuMF implementation in this import; supported: BPRMF, LGCN."
        )


def build_framework_backbone_adapter(
    *,
    framework: str,
    backbone: str,
    dataset: InteractionDataset,
    embedding_dim: int,
    l2_reg: float = 0.0,
    lightgcn_layers: int = 2,
    neumf_mlp_dims: tuple[int, ...] = (64, 32, 16, 8),
    neumf_dropout: float = 0.0,
    device: torch.device | str | None = None,
) -> nn.Module:
    framework_key = str(framework or "recbole").strip().lower()
    adapter_cls: type[nn.Module]
    if framework_key == "recbole":
        adapter_cls = RecBoleBackboneAdapter
    elif framework_key == "elliot":
        adapter_cls = ElliotBackboneAdapter
    elif framework_key == "lenskit":
        adapter_cls = LensKitBackboneAdapter
    else:
        raise ValueError(f"Unsupported student framework: {framework}. Choose from recbole, elliot, lenskit.")
    return adapter_cls(
        backbone=backbone,
        dataset=dataset,
        embedding_dim=embedding_dim,
        l2_reg=l2_reg,
        lightgcn_layers=lightgcn_layers,
        neumf_mlp_dims=neumf_mlp_dims,
        neumf_dropout=neumf_dropout,
        device=device,
    )
