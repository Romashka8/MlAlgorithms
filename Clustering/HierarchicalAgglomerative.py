import numpy as np
import pandas as pd


class Agglomerative:
    def __init__(self, n_clusters: int = 3, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric

        self.metrics = {
            'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2)),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2)),
            'cosine': lambda x1, x2: 1 - (x1 @ x2) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))
        }

    def __repr__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'MyAgglomerative class: ' + res

    def euclidean(self, a: np.array | pd.Series, b: np.array | pd.Series) -> np.array | pd.Series:
        return np.sqrt(np.sum((a - b) ** 2))

    def fit_predict(self, x) -> np.array:
        n_samples, _ = x.shape
        x_values = {cluster: [values, [values]] \
                    for cluster, values in enumerate(x.values, 1)}

        clusters = list(range(n_samples + 1))

        while len(x_values) > self.n_clusters:
            d = {}
            for cluster1, values1 in x_values.items():
                for cluster2, values2 in x_values.items():
                    temp1, temp2 = cluster1, cluster2
                    if temp1 > temp2:
                        temp1, temp2 = temp2, temp1
                    if temp1 != temp2 and (temp1, temp2) not in d:
                        d[(temp1, temp2)] = self.metrics[self.metric](values1[0], values2[0])

            cluster1, cluster2 = min(d, key=d.get)
            x_values[cluster1][1].extend(x_values[cluster2][1])
            x_values[cluster1][0] = np.mean(x_values[cluster1][1], axis=0)

            for i in range(1, n_samples + 1):
                if clusters[i] == cluster2:
                    clusters[i] = cluster1

            del x_values[cluster2]

        return clusters[1:]
