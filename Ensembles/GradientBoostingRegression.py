import numpy as np
import pandas as pd

import random
from typing import Optional
import BaseModelsRegression


class BoostRegression:
    def __init__(self, n_estimators: int = 10, learning_rate: float | callable = 0.1, max_depth: int = 5, min_samples_split: int = 2,
                 max_leafs: int = 20, bins: int = 16, loss: str = 'MSE', metric: Optional[str] = None,
                 max_features: float = 0.5, max_samples: float = 0.5, random_state: int = 42, reg: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.loss = loss
        self.metric = metric
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.pred_0 = None
        self.trees = []
        self.fi = {}
        self.metrics = {
            'mae': lambda y_pred, y: np.mean(np.abs(y_pred)),
            'mse': lambda y_pred, y: np.mean(y_pred ** 2),
            'rmse': lambda y_pred, y: np.sqrt(np.mean(y_pred ** 2)),
            'mape': lambda y_pred, y: np.mean(np.abs(y_pred / y)) * 100,
            'r2': lambda y_pred, y: 1 - np.sum(y_pred ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.best_score = 0

    def __str__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'BoostRegression class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=None) -> None:
        self.fi = {col: 0.0 for col in x.columns}
        n_samples, n_features = x.shape

        random.seed(self.random_state)
        init_cols = list(x.columns)
        cols_smpl_cnt = round(x.shape[1] * self.max_features)
        init_rows_cnt = x.shape[0]
        rows_smpl_cnt = round(x.shape[0] * self.max_samples)

        # in other case calc median(best start for MAE)
        self.pred_0 = y.mean() if self.loss == 'MSE' else y.median()
        r = y - self.pred_0
        # leafs count for regularization
        leafs_cnt = 0
        for estimator in range(1, self.n_estimators + 1):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            x_sample = x.loc[rows_idx, cols_idx]
            y_sample = y.loc[rows_idx]
            # calc grad
            if self.loss == 'MSE':
                grad = 2 * r
            else:
                grad = np.sign(r)
            tree = BaseModelsRegression.TreeRegression(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs, bins=self.bins
            )
            grad_sample = grad.loc[rows_idx]
            # train tree on anty grad
            tree.fit(x_sample, -grad_sample, y_size=n_samples)
            # correct predictions
            grad_pred = tree.grad(x_sample)
            for leaf in grad_pred:
                diff = r.loc[grad_pred[leaf]]
                if self.loss == 'MSE':
                    leaf.value_leaf = diff.mean() + self.reg * leafs_cnt
                # calc for MAE
                else:
                    leaf.value_leaf = diff.median() + self.reg * leafs_cnt
            leafs_cnt += tree.leafs_cnt
            # fi
            for col in tree.fi:
                self.fi[col] += tree.fi[col]
            # adding tree
            self.trees.append(tree)
            pred = tree.predict(x)
            if isinstance(self.learning_rate, float) or isinstance(self.learning_rate, int):
                r -= self.learning_rate * pred
            else:
                # for example: lambda iter: 0.5 * (0.85 ** iter)
                lr = self.learning_rate(estimator)
                r -= lr * pred
            if self.metric is None:
                self.best_score = self.metrics[self.loss](r, y_sample)
            else:
                self.best_score = self.metrics[self.metric](r, y_sample)
            if verbose and self.metric and verbose % estimator == 0:
                print(
                    f'{estimator}.Loss[{self.loss}]: {self.metrics[self.loss](r, y_sample)} | {self.metric}: {self.best_score}')
            elif verbose and verbose % estimator == 0:
                print(f'{estimator}.Loss[{self.loss}]: {self.metrics[self.loss](r, y_sample)}')
        return

    def predict(self, x: pd.DataFrame) -> np.array:
        pred = self.pred_0
        for index, tree in enumerate(self.trees, start=1):
            if isinstance(self.learning_rate, float) or isinstance(self.learning_rate, int):
                pred += self.learning_rate * tree.predict(x)
            else:
                lr = self.learning_rate(index)
                pred += lr * tree.predict(x)
        return pred
