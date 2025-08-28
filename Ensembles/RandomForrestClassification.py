# ----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import random
import BaseModelsClassification
from typing import Optional

# ----------------------------------------------------------------------------------------------------------------------------------------

class MyForestClf:
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 criterion: str = 'entropy', random_state: int = 42, oob_score: Optional[str] = None):
        # forest parameters
        self.n_estimators = n_estimators
        self.max_features = max_features if 0.0 <= max_features <= 1.0 else 0.5
        self.max_samples = max_samples if 0.0 <= max_samples <= 1.0 else 0.5
        # trees parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion if criterion in ('gini', 'entropy') else 'entropy'
        # feature importans - sum of all feature importance in trees in forest
        self.fi = {}
        # adding random state
        self.random_state = random_state
        self.oob_score = oob_score if oob_score in ('precision', 'recall', 'accuracy', 'roc_auc', 'f1') else None
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
        return 'MyForestClf class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        n_samples, n_features = x.shape
        self.fi = {col: 0 for col in x.columns}

        random.seed(self.random_state)
        init_cols = list(x.columns)
        init_rows_cnt = n_samples
        cols_smpl_cnt = int(np.round(self.max_features * n_features))
        rows_smpl_cnt = int(np.round(self.max_samples * n_samples))
        for estimator in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            # find values for oob samples
            cols_oob = list(set(init_cols) - set(cols_idx))
            rows_oob = list(set(range(init_rows_cnt)) - set(rows_idx))
            x_sample = x.loc[rows_idx, cols_idx]
            y_sample = y.loc[rows_idx]
            x_oob = x.loc[rows_oob]
            tree = BaseModelsClassification.TreeClassificator(max_depth=self.max_depth,
                                                              min_samples_split=self.min_samples_split,
                                                              max_leafs=self.max_leafs, bins=self.bins,
                                                              criterion=self.criterion)
            tree.fit(x_sample, y_sample, y_size=x.shape[0])
            self.forest.append(tree)
            self.leafs_cnt += tree.leafs_cnt
            for col in tree.fi:
                self.fi[col] += tree.fi[col]
            # predict for oob
            pred = tree.predict_proba(x_oob)
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
            preds_mean = np.array(preds_mean)
            if self.oob_score != 'roc_auc':
                preds_mean[preds_mean <= 0.5], preds_mean[preds_mean > 0.5] = 0, 1
            else:
                # need to round for roc_auc(check metrics code)
                preds_mean = np.round(preds_mean, 10)
            # self.oob_score_ = self.metrics[self.oob_score].calc(preds_mean, np.array(y.loc[oob_indexes]))
            self.oob_score_ = BaseModelsClassification.TableMetrics(preds_mean, np.array(y.loc[oob_indexes]),
                                                                    self.oob_score).score()
        return

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        res = self.forest[0].predict_proba(x)
        for tree in range(1, len(self.forest)):
            res += self.forest[tree].predict_proba(x)
        return res / self.n_estimators

    def predict(self, x: pd.DataFrame, type: str = 'mean') -> np.array:
        res = None
        if type == 'mean':
            res = self.predict_proba(x)
            res[res <= 0.5], res[res > 0.5] = 0, 1
        elif type == 'vote':
            res = self.forest[0].predict(x)
            for tree in range(1, len(self.forest)):
                res += self.forest[tree].predict(x)
            res = list(map(lambda x: self.n_estimators - x, res))
            for vote in range(len(res)):
                if res[vote] <= self.n_estimators - res[vote]:
                    res[vote] = 1
                else:
                    res[vote] = 0
        return np.array(res, dtype=int)

# ----------------------------------------------------------------------------------------------------------------------------------------
