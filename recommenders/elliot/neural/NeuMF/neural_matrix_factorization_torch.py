"""
PyTorch wrapper for the NeuMF recommender.
"""

import pickle
import numpy as np
import torch
from tqdm import tqdm

from recommenders.elliot.base_recommender_model import BaseRecommenderModel
from recommenders.elliot.base_recommender_model import init_charger
from recommenders.elliot.neural.NeuMF import custom_sampler as cs
from recommenders.elliot.neural.NeuMF.neural_matrix_factorization_torch_model import \
    NeuralMatrixFactorizationTorchModel
from recommenders.elliot.recommender_utils_mixin import RecMixin
from recdistill.paths import NMF, teacher_weights_path


class NeuMFTorch(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 10, int, None),
            ("_dropout", "dropout", "drop", 0, None, None),
            ("_batch_eval", "batch_eval", "batch_eval", 256, None, None),
            ("_is_mf_train", "is_mf_train", "mftrain", True, None, None),
            ("_is_mlp_train", "is_mlp_train", "mlptrain", True, None, None),
            ("_negative_samples", "negative_samples", "neg", 4, int, None)
        ]
        self.autoset_params()

        if hasattr(self._params, "m"):
            self._negative_samples = int(getattr(self._params, "m"))
            self.logger.info(f"Parameter m detected. Overriding negative_samples with {self._negative_samples}")

        self._mlp_hidden_size = (self._factors * 4, self._factors * 2, self._factors)
        self._mlp_factors = self._factors
        self.dataset_name = getattr(params.meta, "dataset_name", getattr(config, "dataset", getattr(data.config, "dataset", None)))
        self._save_model = getattr(params.meta, "save_model", False)

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = cs.Sampler(self._data.i_train_dict, self._negative_samples)
        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NeuralMatrixFactorizationTorchModel(
            self._num_users,
            self._num_items,
            self._factors,
            self._mlp_factors,
            self._mlp_hidden_size,
            self._dropout,
            self._is_mf_train,
            self._is_mlp_train,
            self._learning_rate,
            self._seed
        )

    @property
    def name(self):
        return "NeuMFTorch" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions * (self._negative_samples + 1) // self._batch_size),
                      disable=not self._verbose) as t:
                for batch in self._sampler.step(self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss / max(steps, 1))

        if self._save_model:
            self.save_model()

    def save_model(self):
        save_path = teacher_weights_path(
            model=NMF,
            dataset=self.dataset_name,
            embedding_dim=self._factors,
            phase='best'
        )
        print(f"Saving NeuMFTorch model to '{save_path}'...")

        state_dict = self._model.state_dict()
        user_emb = torch.cat(
            [
                state_dict["user_mf_embedding.weight"].detach().cpu(),
                state_dict["user_mlp_embedding.weight"].detach().cpu(),
            ],
            dim=1,
        )
        item_emb = torch.cat(
            [
                state_dict["item_mf_embedding.weight"].detach().cpu(),
                state_dict["item_mlp_embedding.weight"].detach().cpu(),
            ],
            dim=1,
        )

        with open(save_path, "wb") as file:
            pickle.dump(
                {
                    "user_emb": user_emb,
                    "item_emb": item_emb,
                    "user_id": self._data.private_users,
                    "item_id": self._data.private_items,
                    "model_name": "NeuMFTorch",
                    "representation": "concat(user_mf, user_mlp) / concat(item_mf, item_mlp)",
                    "mf_factors": self._factors,
                    "mlp_factors": self._mlp_factors,
                    "mlp_hidden_size": self._mlp_hidden_size,
                    "dropout": self._dropout,
                    "is_mf_train": self._is_mf_train,
                    "is_mlp_train": self._is_mlp_train,
                    "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
                },
                file,
            )
        print(f"NeuMFTorch model saved correctly in '{save_path}'")

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        device = self._model.device

        self._model.eval()
        with torch.no_grad():
            for _, offset in enumerate(range(0, self._num_users, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, self._num_users)
                user_range = torch.arange(offset, offset_stop, device=device, dtype=torch.long)
                predictions = torch.empty((offset_stop - offset, self._num_items),
                                          device=device,
                                          dtype=torch.float32)

                for item_offset in range(0, self._num_items, self._batch_eval):
                    item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                    item_range = torch.arange(item_offset, item_offset_stop, device=device, dtype=torch.long)

                    users = user_range[:, None].expand(-1, item_offset_stop - item_offset)
                    items = item_range[None, :].expand(offset_stop - offset, -1)

                    p = self._model.get_recs((users, items))
                    predictions[:, item_offset:item_offset_stop] = p

                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath, map_location=self._model.device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.evaluate()
            return True
        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")
