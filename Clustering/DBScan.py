import numpy as np
import pandas as pd


class DBSCAN:
    def __init__(self, eps: int = 3, min_samples: int = 3, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.metrics = {
            'euclidean': self.euclidean,
            'chebyshev': self.chebyshev,
            'manhattan': self.manhattan,
            'cosine': self.cosine
        }

    def __str__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'DBSCAN class: ' + res

    def fit_predict(self, x: pd.DataFrame) -> list:
        n_samples, _ = x.shape
        labels = [-1] * n_samples
        cluster_id = 0

        for idx, row in x.iterrows():
            if labels[idx] != -1:
                continue

            neighbors = self.__region_query(x, idx)
            if len(neighbors) < self.min_samples:
                labels[idx] = -1
            else:
                self.__expand_cluster(x, labels, idx, neighbors, cluster_id)
                cluster_id += 1

        return labels

    def euclidean(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return np.sqrt(np.sum((a - b) ** 2))

    def chebyshev(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return np.max(np.abs(a - b))

    def manhattan(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return np.sum(np.abs(a - b))

    def cosine(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return 1 - np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

    def __region_query(self, x: pd.DataFrame, point_idx: int) -> list:
        neighbors = []
        for idx, row in x.iterrows():
            if self.metrics[self.metric](x.iloc[point_idx], row) < self.eps:
                neighbors.append(idx)
        return neighbors

    def __expand_cluster(self, x: pd.DataFrame, labels: list | np.array | pd.Series,
                         point_id: int, neighbors: list | np.array | pd.Series, cluster_id: int):
        labels[point_id] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                new_neighbors = self.__region_query(x, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1
