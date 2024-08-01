import numpy as np
import pandas as pd

from typing import Optional, Callable
import random


class TableMetrics:
    def __init__(self, y_true: np.array | list, y_pred_proba: np.array | list, metric: str):
        self.y_true = np.array(y_true)
        self.y_pred = np.array((y_pred_proba > 0.5).astype(int))
        self.y_pred_proba = y_pred_proba
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.metric = metric

    def score(self) -> Optional[float]:
        if self.metric == 'accuracy':
            return self.accuracy()
        elif self.metric == 'precision':
            return self.precision()
        elif self.metric == 'recall':
            return self.recall()
        elif self.metric == 'f1':
            return self.f1_score()
        elif self.metric == 'roc_auc':
            return self.roc_auc()
        elif self.metric == 'false_positive_rate':
            return self.false_positive_rate()
        return None

    def accuracy(self) -> float:
        if self.tp + self.tn + self.fp + self.fn == 0:
            return 0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn)

    def f1_score(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def roc_auc(self) -> float:
        sqr = 0
        n_ones = np.sum(self.y_true == 1)
        n_zeroes = np.sum(self.y_true == 0)
        m = n_ones * n_zeroes
        trip = sorted(zip(self.y_pred_proba, self.y_true), reverse=True)
        for _, true in trip:
            if true == 1:
                sqr += n_zeroes
            else:
                n_zeroes -= 1
        return sqr / m


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
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
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False) -> None:
        n_samples, n_features = x.shape
        ones = np.ones((n_samples, 1))
        x = np.hstack((ones, x))

        self.weights = np.ones((n_features + 1, 1))

        random.seed(self.random_state)

        EPS = 1e-15
        y = np.array(y)
        for iter in range(1, self.n_iter + 1):
            sample_rows_idx = range(n_samples)
            batch_size = self.sgd_sample if self.sgd_sample else n_samples
            if isinstance(batch_size, float):
                batch_size = int(n_samples * batch_size)
            sample_rows_idx = random.sample(range(n_samples), batch_size)

            x_sample = x[sample_rows_idx, :]
            y_sample = y[sample_rows_idx]

            z = x_sample @ self.weights
            y_pred = self.sigmoid(z).flatten()
            logloss = -np.mean(y_sample * np.log(y_pred + EPS) - (1 - y_sample) * np.log(1 - y_pred + EPS))

            grad = ((y_pred - y_sample) @ x_sample) / batch_size
            grad = grad.reshape(-1, 1)

            if self.reg:
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    grad += self.l1_coef * np.sign(self.weights)
                    logloss += self.l1_coef * np.sum(np.abs(self.weights))
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    grad += self.l2_coef * 2 * self.weights
                    logloss += self.l2_coef * np.sum(self.weights ** 2)

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= lr * grad

            self.scores = TableMetrics(y_sample, y_pred, self.metric)

            if verbose and iter % verbose == 0:
                print(f"{iter if iter != 0 else 'start'} | loss: {logloss}",
                      f"| {self.metric}: {self.scores.score()}" if self.metric else '', f"| learning_rate: {lr}")

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        n_samples, _ = x.shape
        ones = np.ones((n_samples, 1))
        x = np.hstack((ones, x))
        return self.sigmoid(x @ self.weights)

    def predict(self, x: pd.DataFrame) -> np.array:
        proba = self.predict_proba(x)
        return (proba > 0.5).astype(int)

    def get_coef(self) -> np.array | list:
        return self.weights[1:]

    def get_best_score(self) -> float:
        self.scores.roc_auc()
        return self.scores.score()

    def sigmoid(self, z: np.array | pd.Series) -> np.array:
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = 0
        self.x = None
        self.y = None

        self.metrics = {
            'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine': lambda x1, x2: 1 - (x1 @ x2) / (np.sqrt(np.sum(x1 ** 2, axis=1)) * np.sqrt(np.sum(x2 ** 2)))
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.x = x.copy()
        self.y = y.copy()
        self.train_size = self.x.shape

    def predict(self, x: pd.DataFrame):
        predictions = []
        for _, test_row in x.iterrows():
            distances = self.metrics[self.metric](self.x.values, test_row.values)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y.iloc[k_nearest_indices]
            if self.weight == 'uniform':
                counts = k_nearest_labels.value_counts()
                count_1 = counts.get(1, 0)
                count_0 = counts.get(0, 0)
                predictions.append([0, 1][count_1 >= count_0])
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weights = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights[label] += rank_weights[i]
                predictions.append(np.argmax(weights))
            elif self.weight == 'distance':
                distances[distances == 0] = 1e-10
                weights = 1 / distances[k_nearest_indices]
                weights_sum = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights_sum[label] += weights[i]
                predictions.append(np.argmax(weights_sum))
        return np.array(predictions)

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        probabilities = []
        for _, test_row in x.iterrows():
            distances = self.metrics[self.metric](self.x.values, test_row.values)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y.iloc[k_nearest_indices]
            if self.weight == 'uniform':
                prob_class_1 = np.mean(k_nearest_labels)
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weights = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights[label] += rank_weights[i]
                prob_class_1 = weights[1] / np.sum(weights)
            elif self.weight == 'distance':
                distances[distances == 0] = 1e-10
                weights = 1 / distances[k_nearest_indices]
                weights_sum = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights_sum[label] += weights[i]
                prob_class_1 = weights_sum[1] / np.sum(weights_sum)
            probabilities.append(prob_class_1)
        return np.array(probabilities)


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None


class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = None, criterion: str = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.__sum_tree_values = 0
        self.split_values = {}
        self.criterion = criterion
        self.fi = {}
        self.tree = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.fi = {col: 0 for col in x.columns}

        def create_tree(root, x_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, split_value, ig = self.get_best_split(x_root, y_root)

            proportion_ones = len(y_root[y_root == 1]) / len(y_root) if len(y_root) else 0

            if proportion_ones == 0 or proportion_ones == 1 or depth >= self.max_depth or \
                    len(y_root) < self.min_samples_split or \
                    (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            self.fi[col_name] += len(y_root) / len(y) * ig

            x_left = x_root.loc[x_root[col_name] <= split_value]
            y_left = y_root.loc[x_root[col_name] <= split_value]

            x_right = x_root.loc[x_root[col_name] > split_value]
            y_right = y_root.loc[x_root[col_name] > split_value]

            if len(x_left) == 0 or len(x_right) == 0:
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, x_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, x_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, x, y)

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        for _, row in x.iterrows():
            node = self.tree
            while node.feature is not None:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            yield node.value_leaf

    def predict(self, x: pd.DataFrame) -> np.array:
        y_pred = np.array(list(self.predict_proba(x)))
        return (y_pred >= 0.5).astype(int)

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
        count_labels = y.value_counts()
        p_zero = count_labels / count_labels.sum()
        s_zero = self.__node_rule(p_zero)

        x = x.copy()
        x.loc[:, 'y'] = y
        split_values = {}
        col_name = None
        split_value = None
        s_cur_min = float('inf')

        for col in x.columns[:-1]:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(x[col])
                if self.bins is not None and len(x_unique_values) - 1 >= self.bins:
                    _, self.split_values[col] = np.histogram(x[col], bins=self.bins)
                else:
                    self.split_values[col] = np.array(
                        [(x_unique_values[i - 1] + x_unique_values[i]) / 2 for i in range(1, len(x_unique_values))])

            for split_value_cur in self.split_values[col]:
                left_split = x['y'][x[col] <= split_value_cur]
                right_split = x['y'][x[col] > split_value_cur]

                left_count_labels = left_split.value_counts()
                p_left = left_count_labels / left_count_labels.sum()
                s_left = self.__node_rule(p_left, left_split)

                right_count_labels = right_split.value_counts()
                p_right = right_count_labels / right_count_labels.sum()
                s_right = self.__node_rule(p_right, right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                s_cur = weight_left * s_left + weight_right * s_right
                if s_cur_min > s_cur:
                    s_cur_min = s_cur
                    col_name = col
                    split_value = split_value_cur

        ig = s_zero - s_cur_min
        return col_name, split_value, ig

    def __node_rule(self, p, split=pd.Series()):
        if self.criterion == 'entropy':
            return -np.sum(p * np.log2(p)) if not split.empty else 0
        elif self.criterion == 'gini':
            return 1 - np.sum(p ** 2)

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}"
