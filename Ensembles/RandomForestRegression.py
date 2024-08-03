import numpy as np
import pandas as pd
import BaseModelsRegression

import random
from typing import Optional


class MyForestReg:
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 random_state: int = 42, oob_score: Optional[str] = None):
        # forest parameters
        self.n_estimators = n_estimators
        self.max_features = max_features if 0.0 <= max_features <= 1.0 else 0.5
        self.max_samples = max_samples if 0.0 <= max_samples <= 1.0 else 0.5
        # trees parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.metrics = {
            'mae': lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse': lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2': lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        # feature importance - sum of all feature importance in trees in forest
        self.fi = {}
        # adding random state
        self.random_state = random_state
        self.oob_score = oob_score if oob_score in ('mae', 'mse', 'rmse', 'mape', 'r2') else None
        self.oob_score_ = 0
        # keeping samples from oob
        # format: {row_index: [vals]}
        self.oob_samples_table = {}
        # leafs in all trees in forest
        self.leafs_cnt = 0
        # keeping all trees in forest
        self.forest = []

    def __str__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'MyForestReg class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.fi = {col: 0 for col in x.columns}

        random.seed(self.random_state)
        init_cols = list(x.columns)
        cols_smpl_cnt = round(x.shape[1] * self.max_features)
        init_rows_cnt = x.shape[0]
        rows_smpl_cnt = round(x.shape[0] * self.max_samples)
        for estimator in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            # find values for oob samples
            rows_oob = list(set(range(init_rows_cnt)) - set(rows_idx))
            x_sample = x.loc[rows_idx, cols_idx]
            y_sample = y.loc[rows_idx]
            x_oob = x.loc[rows_oob]
            tree = BaseModelsRegression.TreeRegression(max_depth=self.max_depth,
                                                       min_samples_split=self.min_samples_split,
                                                       max_leafs=self.max_leafs, bins=self.bins)
            # y_size - need for oob score. It calcs on learning samples. Especially we need her size.
            tree.fit(x_sample, y_sample, y_size=x.shape[0])
            self.forest.append(tree)
            self.leafs_cnt += tree.leafs_cnt
            for col in tree.fi:
                self.fi[col] += tree.fi[col]
            # predict for oob
            pred = tree.predict(x_oob)
            for score in range(len(rows_oob)):
                if rows_oob[score] in self.oob_samples_table:
                    self.oob_samples_table[rows_oob[score]].append(pred[score])
                else:
                    self.oob_samples_table[rows_oob[score]] = [pred[score]]
        if self.oob_score:
            oob_indexes = list(self.oob_samples_table.keys())
            preds_mean = []
            for pred in self.oob_samples_table:
                preds_mean.append(sum(self.oob_samples_table[pred]) / len(self.oob_samples_table[pred]))
            self.oob_score_ = self.metrics[self.oob_score].calc(preds_mean, y.loc[oob_indexes])
        return

    def predict(self, x: pd.DataFrame) -> np.array:
        res = self.forest[0].predict(x)
        for tree in range(1, len(self.forest)):
            res += self.forest[tree].predict(x)
        return res / self.n_estimators
