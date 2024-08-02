import numpy as np
import pandas as pd

import BaseModelsClassification

import random
import copy

from typing import Optional


class BaggingClassification:
    def __init__(self, estimator=None, n_estimators: int = 10, max_samples: float = 1.0,
                 random_state: int = 42, oob_score: Optional[float] = None):
        """
        :param estimator: one model from BaseModelsRegression.py
        :param n_estimators: estimators in Bagging
        :param max_samples: max sample size
        :param random_state: random seed
        :param oob_score: volume of oob sample
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []
        self.oob_score = oob_score
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
            base_model.fit(x_sample, y_sample)

            oob_preds = base_model.predict(x_oob, )
            for i, pred in zip(y.drop(index=sample_rows_idx).index, oob_preds):
                oob_preds_dict.setdefault(i, []).append(pred)

            self.estimators.append(base_model)

        y_oob = list(map(lambda x: np.mean(oob_preds_dict[x]), oob_preds_dict.keys()))
        self.oob_score_ = BaseModelsClassification.TableMetrics(np.array(y[list(oob_preds_dict.keys())].values),
                                                                np.array(y_oob), self.oob_score).score()

    def predict(self, x: pd.DataFrame, type: str) -> np.array:
        pred_probs = np.array([list(model.predict_proba(x)) for model in self.estimators])
        if type == 'mean':
            pred = pred_probs.mean(axis=0)
            pred = (pred > 0.5).astype(int)
        elif type == 'vote':
            pred_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=0,
                                             arr=(pred_probs > 0.5).astype(int))
            pred = pred_votes.argmax(axis=0)
        return pred

    def predict_proba(self, x) -> np.array:
        pred = np.array([list(model.predict_proba(x)) for model in self.estimators]).mean(axis=0)
        return pred

    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyBaggingClf class: " + ", ".join(params)
