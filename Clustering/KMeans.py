import numpy as np
import pandas as pd


class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def __repr__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'KMeans class: ' + res

    def fit(self, x: pd.DataFrame) -> None:
        n_samples, n_feature = x.shape
        np.random.seed(seed=self.random_state)

        min_ = np.min(x, axis=0)
        max_ = np.max(x, axis=0)

        self.cluster_centers_ = None
        self.inertia_ = np.inf

        for _ in range(self.n_init):
            clusters = np.array([np.random.uniform(min_, max_) \
                                 for _ in range(self.n_clusters)])

            for _ in range(self.max_iter):
                y_pred = []
                for _, row in x.iterrows():
                    row = row.values
                    # argmin return index - this means for us that we get a cluster!
                    y_pred.append(np.argmin(np.array([self.euclidean(row, cluster) \
                                                      for cluster in clusters])))

                clusters_map = [[] for _ in range(self.n_clusters)]

                for i, pred in enumerate(y_pred):
                    clusters_map[pred].append(x.iloc[i].values)

                for i in range(self.n_clusters):
                    if clusters_map[i]:
                        clusters[i] = np.mean(clusters_map[i], axis=0)

            wscc = 0
            for i, pred in enumerate(y_pred):
                wscc += self.euclidean(x.iloc[i].values, clusters[pred]) ** 2

            if wscc < self.inertia_:
                self.inertia_ = wscc
                self.cluster_centers_ = clusters

    def euclidean(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, x: pd.DataFrame) -> np.array:
        y_pred = []
        for _, row in x.iterrows():
            row = row.values
            y_pred.append(np.argmin(np.array([self.euclidean(row, cluster) \
                                              for cluster in self.cluster_centers_])))
        return y_pred
