import numpy as np
import pandas as pd

import random
import copy

from typing import Optional


class BaggingRegression:
    def __init__(self, estimator=None, n_estimators: int = 10, max_samples: float = 1.0,
                 random_state: int = 42, oob_score: Optional[float] = None):
        """
        :param estimator: one model from BaseModelsRegression.py
        :param n_estimators: estimators in Bagging
        :param max_samples: max sample size
        :param random_state: random seed
        :param oob_score: volume of oob sample
        """
        self.metrics = {
            'mae': lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse': lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2': lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []
        self.oob_score = oob_score if oob_score in self.metrics else None
        self.oob_score_ = 0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        # keep in mind how we for samples in Bagging
        # we're using bootstrap - this means that objects can be included in fitting sample more that one time
        random.seed(self.random_state)
        init_rows_cnt = x.shape[0]
        rows_smpl_cnt = round(init_rows_cnt * self.max_samples)

        sample_rows_idxs = {}
        oob_preds_dict = {}

        for i in range(self.n_estimators):
            sample_rows_idxs[i] = random.choices(range(init_rows_cnt), k=rows_smpl_cnt)

        for i in range(self.n_estimators):
            sample_rows_idx = sample_rows_idxs[i]
            x_sample = x.iloc[sample_rows_idx, :].reset_index(drop=True)
            y_sample = y.iloc[sample_rows_idx].reset_index(drop=True)

            x_oob = x.drop(index=sample_rows_idx).reset_index(drop=True)
            y_oob = y.drop(index=sample_rows_idx).reset_index(drop=True)

            base_model = copy.copy(self.estimator)
            base_model.fit(x_sample)

            oob_preds = base_model.predict(x_oob)
            for i, pred in zip(y.drop(index=sample_rows_idx).index, oob_preds):
                oob_preds_dict.setdefault(i, []).append(pred)

            self.estimators.append(base_model)

        y_oob = list(map(lambda x: np.mean(oob_preds_dict[x]), oob_preds_dict.keys()))
        self.oob_score_ = self.metrics[self.oob_score](
            y[list(oob_preds_dict.keys())].values, y_oob
        )

    def predict(self, x: pd.DataFrame) -> np.array:
        y_preds = []
        print(self.estimators[0])
        for estimator in self.estimators:
            y_pred = estimator.predict(x)
            y_preds.append(y_pred)
        return np.array(y_preds).mean(axis=0)

    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "BaggingRegression class: " + ", ".join(params)
