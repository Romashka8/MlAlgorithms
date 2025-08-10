# ----------------------------------------------------------------------------------------------------------------------------------------

import abc

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------

class MetricModelBase(abc.ABC):
    
    metrics = {
        'euclidean': np.vectorize(lambda x, y: ((x - y) ** 2)),
        'chebyshev': np.vectorize(lambda x, y: abs(x - y)),
        'manhattan': np.vectorize(lambda x, y: abs(x - y)),
        'cosine': np.vectorize(lambda x, y: x * y)
    }

    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        """
        :param k: neighbors quantity
        :type int, default 3
        :param metric: using metric
        :type callable, determines upper
        :param weight: type of weights in weighted KNN
        :type str
        """
        self.k = k
        self.metric = metric if metric in self.metrics.keys() else 'euclidean'
        self.weight = weight if weight in ('uniform', 'rank', 'distance') else 'uniform'
        self.train_size = None
        self.features_train = None
        self.target_train = None
        pass

    # implemented in child classes
    @abc.abstractmethod
    def __repr__(self):
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.features_train = x.copy()
        self.target_train = y.copy()
        self.train_size = (x.shape[0], x.shape[1])

    # implemented in child classes
    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> np.array:
        """
        :param x: feature matrix
        :return: prediction vector
        """
        pass

# ----------------------------------------------------------------------------------------------------------------------------------------

class UnfittedModel(BaseException):
    def __str__(self):
        return 'Model must be fitted first!'

# ----------------------------------------------------------------------------------------------------------------------------------------
