import numpy as np
import pandas as pd

from typing import Optional, Callable
import random

'''
This module represents base models, adapted for ensembles
'''


class LinearRegression:
    def __init__(self,
                 n_iter: int = 100,
                 learning_rate: float | Callable = 0.1,
                 metric: Optional[str] = None,
                 reg: Optional[str] = None,
                 l1_coef: float = 0.0,
                 l2_coef: float = 0.0,
                 sgd_sample: Optional[int] = None,
                 random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.metrics = {
            'mae': lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse': lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2': lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: Optional[int] = False) -> None:
        n_samples, n_features = x.shape
        ones = np.ones((n_samples, 1))
        x = np.hstack((ones, x))
        x = pd.DataFrame(x)
        y = pd.Series(y)

        random.seed(self.random_state)

        self.weights = np.ones(n_features + 1)

        for iter in range(1, self.n_iter + 1):
            sample_rows_idx = list(range(n_samples))
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(sample_rows_idx, self.sgd_sample)
            if isinstance(self.sgd_sample, float):
                k = int(n_samples * self.sgd_sample)
                sample_rows_idx = random.sample(sample_rows_idx, k)

            x_sample = x.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]

            y_pred = x @ self.weights
            loss = self.__calc_loss(y, y_pred)

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= lr * self.__calc_grad(x_sample, y_sample)

            if self.metric is not None:
                self.best_score = self.metrics[self.metric](y_sample, y_pred)

            if verbose and iter % verbose == 0:
                print(f"{iter if iter != 0 else 'start'} | loss: {loss}",
                      f"| {self.metric}: {self.best_score}" if self.metric else '', f"| learning_rate: {lr}")

    def predict(self, x: pd.DataFrame) -> np.array:
        ones = np.ones((x.shape[0], 1))
        x = np.hstack((ones, x))
        return x @ self.weights

    def __calc_loss(self, y: pd.Series | np.array, y_pred: pd.Series | np.array) -> np.array:
        loss = np.sum((y - y_pred) ** 2) / len(y)

        if self.reg == 'l1' or self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == 'l2' or self.reg == 'elasticnet':
            loss += self.l2_coef * np.sum(self.weights ** 2)

        return loss

    def __calc_grad(self, x: pd.DataFrame, y: pd.Series) -> np.array | pd.Series:
        n_samples, _ = x.shape
        grad = 2 * (x @ self.weights - y) @ x / n_samples

        if self.reg:
            if self.reg == 'l1' or self.reg == 'elasticnet':
                grad += self.l1_coef * np.sign(self.weights)
            if self.reg == 'l2' or self.reg == 'elasticnet':
                grad += self.l2_coef * 2 * self.weights

        return grad

    def get_coef(self) -> np.array | list:
        return self.weights[1:]

    def get_best_score(self) -> float:
        return self.best_score

    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "LinearRegression class: " + ", ".join(params)


class KNNRegression:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = None
        self.metrics = {
            'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine': lambda x1, x2: 1 - (np.dot(x1, x2.T) / (
                        np.sqrt(np.sum(x1 ** 2, axis=1)[:, np.newaxis]) * np.sqrt(np.sum(x2 ** 2, axis=1))))
        }
        self.metric = metric
        self.weight = weight

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.train_size = x.shape

    def predict(self, x: pd.DataFrame) -> np.array:
        predictions = []
        x = np.array(x)
        for test_row in x:
            distances = self.metrics[self.metric](self.x, test_row.reshape(1, -1))
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y[k_nearest_indices]
            if self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weighted_sum = np.sum(rank_weights * k_nearest_labels)
                predictions.append(weighted_sum / np.sum(rank_weights))
            elif self.weight == 'distance':
                distances = distances[k_nearest_indices]
                distances[distances == 0] = 1e-10
                dist_weights = 1 / distances
                weighted_sum = np.sum(dist_weights * k_nearest_labels)
                predictions.append(weighted_sum / np.sum(dist_weights))
            else:
                predictions.append(np.mean(k_nearest_labels))
        return np.array(predictions)


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None


class TreeRegression:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.fi = {}
        self.split_values = {}
        self.tree = None

    def fit(self, x: pd.DataFrame, y: pd.Series, y_size: Optional[int] = None):
        if y_size is None:
            y_size = len(y)
        self.tree = None
        self.split_values = {}
        self.fi = {col: 0 for col in x.columns}

        def create_tree(root, x_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()

            y_root_unique_size = len(y_root.unique())
            if y_root_unique_size == 0 or y_root_unique_size == 1 or \
                    depth >= self.max_depth or len(y_root) < self.min_samples_split \
                    or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = y_root.mean()
                return root

            col_name, split_value, gain = self.get_best_split(x_root, y_root)

            self.fi[col_name] += gain * len(y_root) / y_size

            x_left = x_root[x_root[col_name] <= split_value]
            y_left = y_root[x_root[col_name] <= split_value]

            x_right = x_root[x_root[col_name] > split_value]
            y_right = y_root[x_root[col_name] > split_value]

            if len(x_left) == 0 or len(x_right) == 0:
                root.side = side
                root.value_leaf = y_root.mean()
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, x_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, x_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, x, y)

    def predict(self, x: pd.DataFrame) -> np.array:
        y_pred = []
        for _, row in x.iterrows():
            node = self.tree
            while node.feature:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.value_leaf)
        return np.array(y_pred)

    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value_leaf}")

    def get_best_split(self, x: pd.DataFrame, y: pd.Series) -> (str, float, float):
        mse_0 = self.mse(y)

        col_name = None
        split_value = None
        gain = -float('inf')

        for col in x.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(x[col])
                if self.bins is None or len(x_unique_values) - 1 < self.bins:
                    self.split_values[col] = np.array([(x_unique_values[i - 1] + \
                                                        x_unique_values[i]) / 2 for i in
                                                       range(1, len(x_unique_values))])
                else:
                    _, self.split_values[col] = np.histogram(x[col], bins=self.bins)

            for split_value_i in self.split_values[col]:
                mask = x[col] <= split_value_i
                left_split, right_split = y[mask], y[~mask]

                mse_left = self.mse(left_split)
                mse_right = self.mse(right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                mse_i = weight_left * mse_left + weight_right * mse_right

                gain_i = mse_0 - mse_i
                if gain < gain_i:
                    col_name = col
                    split_value = split_value_i
                    gain = gain_i

        return col_name, split_value, gain

    def mse(self, t: np.array | pd.Series) -> float:
        t_mean = np.mean(t)
        return ((t - t_mean) ** 2).mean()

    def __str__(self):
        return f"TreeRegression class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
