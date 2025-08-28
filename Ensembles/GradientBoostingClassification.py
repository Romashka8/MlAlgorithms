# ----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import random
import BaseModelsRegression
from typing import Optional, Callable

# ----------------------------------------------------------------------------------------------------------------------------------------

class BoostingClassificator:
    def __init__(self, n_estimators: int = 10, learning_rate: float | callable = 0.1, max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20, bins: int = 16, loss: str = 'MSE', metric: Optional[str] = None,
                 max_features: float = 0.5, max_samples: float = 0.5, random_state: int = 42, reg: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.metric = metric
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.best_score = 0
        self.pred_0 = None
        self.trees = []
        self.total_leafs = 0
        self._threshold = 0.5
        self.fi = {}

    def __str__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'BoostingClassificator class: ' + res

    def _confusion_matrix(self, y: np.array, y_pred: np.array) -> tuple:
        y_pred = np.where(y_pred > self._threshold, 1, 0)
        TP = np.logical_and(y == 1, y_pred == 1).sum()
        FN = np.logical_and(y == 1, y_pred == 0).sum()
        FP = np.logical_and(y == 0, y_pred == 1).sum()
        TN = np.logical_and(y == 0, y_pred == 0).sum()

        return TP, FN, FP, TN

    def _accuracy(self, y: np.array, y_pred: np.array) -> float:
        TP, FN, FP, TN = self._confusion_matrix(y, y_pred)
        return (TP + TN) / (TP + FN + FP + TN)

    def _precision(self, y: np.array, y_pred: np.array) -> float:
        TP, _, FP, _ = self._confusion_matrix(y, y_pred)
        return TP / (TP + FP)

    def _recall(self, y: np.array, y_pred: np.array) -> float:
        TP, FN, _, _ = self._confusion_matrix(y, y_pred)
        return TP / (TP + FN)

    def _f1(self, y: np.array, y_pred: np.array, beta: float = 1) -> float:
        recall = self._recall(y, y_pred)
        precision = self._precision(y, y_pred)

        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    def _roc_auc(self, y: np.array, y_prob: np.array) -> float:
        positives = np.sum(y == 1)
        negatives = np.sum(y == 0)

        y_prob = np.round(y_prob, 10)

        sorted_idx = np.argsort(-y_prob)
        y_sorted = np.array(y)[sorted_idx]
        y_prob_sorted = y_prob[sorted_idx]

        roc_auc_score = 0

        for prob, pred in zip(y_prob_sorted, y_sorted):
            if pred == 0:
                roc_auc_score += (
                        np.sum(y_sorted[y_prob_sorted > prob])
                        + np.sum(y_sorted[y_prob_sorted == prob]) / 2
                )

        roc_auc_score /= positives * negatives

        return roc_auc_score

    def get_metric(self) -> callable:
        if self.metric == "accuracy":
            return self._accuracy
        elif self.metric == "precision":
            return self._precision
        elif self.metric == "recall":
            return self._recall
        elif self.metric == "f1":
            return self._f1
        elif self.metric == "roc_auc":
            return self._roc_auc

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: Optional[int] = None) -> None:
        self.fi = {col: 0.0 for col in x.columns}
        n_samples, n_features = x.shape

        random.seed(self.random_state)
        init_cols = list(x.columns)
        cols_smpl_cnt = round(x.shape[1] * self.max_features)
        init_rows_cnt = x.shape[0]
        rows_smpl_cnt = round(x.shape[0] * self.max_samples)

        eps = 1e-15
        self.pred_0 = np.log(y.mean() / (1 - y.mean()) + eps)
        curr_pred = pd.Series(self.pred_0 * np.ones(x.shape[0]))
        for estimator in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            X_sample = x.loc[rows_idx, cols_idx]
            y_sample = y.loc[rows_idx]
            #
            tree = BaseModelsRegression.TreeRegression(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs, bins=self.bins
            )
            # calc grad
            grad = np.exp(curr_pred) / (1 + np.exp(curr_pred)) - y
            grad_sample = grad.loc[rows_idx]
            tree.fit(X_sample, -grad_sample, y_size=n_samples)
            # correct predictions
            grad_pred = tree.grad(X_sample)
            for leaf in grad_pred:
                p = curr_pred.loc[grad_pred[leaf]]
                # convert logodds to probs
                p = np.exp(p) / (1 + np.exp(p))
                y_true = y.loc[grad_pred[leaf]]
                # calc gamma
                gamma = np.sum(y_true - p) / np.sum(p * (1 - p))
                leaf.value_leaf = gamma + self.reg * self.total_leafs
            self.trees.append(tree)
            self.total_leafs += tree.leafs_cnt
            # fi
            for col in tree.fi:
                self.fi[col] += tree.fi[col]
            pred = tree.predict(x)
            lr = self.learning_rate(estimator) if isinstance(self.learning_rate, Callable) else self.learning_rate
            curr_pred += lr * pred
            curr_loss = -np.mean(y * curr_pred - np.log(1 + np.exp(curr_pred)))
            line_to_print = f"{estimator}. {curr_loss}"
            if self.metric:
                p = np.exp(curr_pred) / (1 + np.exp(curr_pred))
                curr_metric = self.get_metric()(y, np.array(p))
                line_to_print += f" | {self.metric}: {curr_metric}"
                self.best_score = curr_metric
            else:
                self.best_score = curr_loss
            if verbose and estimator % verbose == 0:
                print(line_to_print)
        return

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        pred = 0
        for estimator, tree in enumerate(self.trees):
            lr = self.learning_rate(estimator) if isinstance(self.learning_rate, Callable) else self.learning_rate
            pred += lr * tree.predict(x)
        pred = self.pred_0 + pred
        pred = np.exp(pred) / (1 + np.exp(pred))
        return pred

    def predict(self, x) -> np.array:
        res = self.predict_proba(x)
        res[res <= 0.5], res[res > 0.5] = 0, 1
        return list(map(int, res))

# ----------------------------------------------------------------------------------------------------------------------------------------
