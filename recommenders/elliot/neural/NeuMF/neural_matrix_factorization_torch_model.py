"""
PyTorch implementation of the NeuMF architecture used in Elliot.
"""

from abc import ABC
from collections import OrderedDict
import random

import numpy as np
import torch


class NeuralMatrixFactorizationTorchModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size,
                 embed_mlp_size,
                 mlp_hidden_size,
                 dropout,
                 is_mf_train,
                 is_mlp_train,
                 learning_rate=0.01,
                 random_seed=42,
                 name="NeuralMatrixFactorizationTorchModel",
                 **kwargs):
        super().__init__()

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.embed_mlp_size = embed_mlp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout
        self.is_mf_train = is_mf_train
        self.is_mlp_train = is_mlp_train

        self.user_mf_embedding = torch.nn.Embedding(self.num_users, self.embed_mf_size)
        self.item_mf_embedding = torch.nn.Embedding(self.num_items, self.embed_mf_size)
        self.user_mlp_embedding = torch.nn.Embedding(self.num_users, self.embed_mlp_size)
        self.item_mlp_embedding = torch.nn.Embedding(self.num_items, self.embed_mlp_size)

        torch.nn.init.xavier_uniform_(self.user_mf_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_mf_embedding.weight)
        torch.nn.init.xavier_uniform_(self.user_mlp_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_mlp_embedding.weight)

        mlp_layers = []
        input_size = self.embed_mlp_size * 2
        for idx, units in enumerate(self.mlp_hidden_size):
            mlp_layers.append((f"linear_{idx}", torch.nn.Linear(input_size, units)))
            mlp_layers.append((f"relu_{idx}", torch.nn.ReLU()))
            if self.dropout > 0:
                mlp_layers.append((f"dropout_{idx}", torch.nn.Dropout(self.dropout)))
            input_size = units
        self.mlp_layers = torch.nn.Sequential(OrderedDict(mlp_layers))

        if self.is_mf_train and self.is_mlp_train:
            predict_in = self.embed_mf_size + self.mlp_hidden_size[-1]
        elif self.is_mf_train:
            predict_in = self.embed_mf_size
        elif self.is_mlp_train:
            predict_in = self.mlp_hidden_size[-1]
        else:
            raise RuntimeError("mf_train and mlp_train can not be False at the same time")

        self.predict_layer = torch.nn.Linear(predict_in, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.BCELoss()

        self.to(self.device)

    def _flatten_last_dim(self, tensor):
        original_shape = tensor.shape[:-1]
        flat_tensor = tensor.reshape(-1, tensor.shape[-1])
        return flat_tensor, original_shape

    def _restore_last_dim(self, tensor, original_shape):
        return tensor.reshape(*original_shape, tensor.shape[-1])

    def forward(self, inputs, training=False, **kwargs):
        users, items = inputs

        users = users.to(self.device, dtype=torch.long)
        items = items.to(self.device, dtype=torch.long)

        user_mf_e = self.user_mf_embedding(users)
        item_mf_e = self.item_mf_embedding(items)
        user_mlp_e = self.user_mlp_embedding(users)
        item_mlp_e = self.item_mlp_embedding(items)

        if self.is_mf_train:
            mf_output = user_mf_e * item_mf_e

        if self.is_mlp_train:
            mlp_input = torch.cat([user_mlp_e, item_mlp_e], dim=-1)
            flat_mlp_input, mlp_shape = self._flatten_last_dim(mlp_input)
            mlp_output = self.mlp_layers(flat_mlp_input)
            mlp_output = self._restore_last_dim(mlp_output, mlp_shape)

        if self.is_mf_train and self.is_mlp_train:
            output = torch.cat([mf_output, mlp_output], dim=-1)
        elif self.is_mf_train:
            output = mf_output
        else:
            output = mlp_output

        flat_output, output_shape = self._flatten_last_dim(output)
        logits = self.predict_layer(flat_output)
        logits = self._restore_last_dim(logits, output_shape)

        return torch.sigmoid(logits)

    def train_step(self, batch):
        users, items, labels = batch
        users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)

        self.train()
        predictions = self.forward((users, items), training=True).squeeze(-1)
        loss = self.loss(predictions, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def predict(self, inputs, training=False, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(inputs, training=training)

    def get_recs(self, inputs, training=False, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(inputs, training=training).squeeze(-1)

    def get_top_k(self, preds, train_mask, k=100):
        preds = torch.as_tensor(preds, device=self.device, dtype=torch.float32)
        mask = torch.as_tensor(train_mask, device=self.device, dtype=torch.bool)
        return torch.topk(
            torch.where(mask, preds, torch.tensor(-np.inf, device=self.device)),
            k=k,
            sorted=True
        )
