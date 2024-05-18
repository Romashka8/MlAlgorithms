import numpy as np
import pandas as pd
import random

from LinearModelnterface import LinearModelBase, UnfittedModel, BaseMetric
import LinearMetrics as lm
from typing import Union, Callable


class LinearRegression(LinearModelBase):

    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: BaseMetric = lm.MSE(),
                 sgd_sample: Union[int, float] = None, random_state: int = 42) -> object:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_metric = None
        self.weights = None

    def __repr__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'LinearRegression class: ' + res

    # fitting our model
    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric)
        y_pred = np.dot(x, self.weights)
        self.best_metric = self.metric.calc(y_pred, y)

    # return our weights(except first cause it match a free coef)
    def get_coef(self) -> np.array:
        return np.array(self.weights[1:])

    def get_best_score(self) -> float:
        if self.metric is None:
            print('Using default metric - MSE')
        return self.best_metric

    def predict(self, x: pd.DataFrame) -> np.array:
        if self.weights is None:
            raise UnfittedModel
        x = x.copy()
        x.insert(0, 'free_coef', [1] * x.shape[0])
        x = x.to_numpy()
        return np.dot(x, self.weights)

    def score(self, y_pred: np.array, y: np.array) -> float:
        return lm.R2().calc(y_pred, y)


class LinearRegressionL1(LinearRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: BaseMetric = lm.MSE(),
                 l1_coef: float = 0.1, sgd_sample: Union[int, float] = None, random_state: int = 42):
        """
        :param l1_coef: coefficient in L1 regularisation
        """
        super().__init__(n_iter, learning_rate, metric, sgd_sample, random_state)
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.l1_coef = l1_coef if 0 <= l1_coef <= 1 else 0.1
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_metric = None
        self.weights = None

    def __repr__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'LinearRegressionL1 class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='l1', l1_coef=self.l1_coef)
        y_pred = np.dot(x, self.weights)
        self.best_metric = self.metric.calc(y_pred, y)


class LinearRegressionL2(LinearRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: BaseMetric = lm.MSE(),
                 l2_coef: float = 0.1, sgd_sample: Union[int, float] = None, random_state: int = 42):
        """
        :param l2_coef: coefficient in L2 regularisation
        """
        super().__init__(n_iter, learning_rate, metric, sgd_sample, random_state)
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.l2_coef = l2_coef if 0 <= l2_coef <= 1 else 0.1
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_metric = None
        self.weights = None

    def __repr__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'LinearRegressionL2 class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='l2', l2_coef=self.l2_coef)
        y_pred = np.dot(x, self.weights)
        self.best_metric = self.metric.calc(y_pred, y)


class LinearRegressionElasticNet(LinearRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: BaseMetric = lm.MSE(),
                 l1_coef: float = 0.1, l2_coef: float = 0.1, sgd_sample: Union[int, float] = None,
                 random_state: int = 42):
        """
        :param l1_coef: coefficient in L1 regularisation
        :type: float
        :param l2_coef: coefficient in L2 regularisation
        :type: float
        """
        super().__init__(n_iter, learning_rate, metric, sgd_sample, random_state)
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.l1_coef = l1_coef if 0 <= l1_coef <= 1 else 0.1
        self.l2_coef = l2_coef if 0 <= l2_coef <= 1 else 0.1
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_metric = None
        self.weights = None

    def __repr__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'LinearRegressionElasticNet class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='elasticnet', l1_coef=self.l1_coef,
            l2_coef=self.l2_coef)
        y_pred = np.dot(x, self.weights)
        self.best_metric = self.metric.calc(y_pred, y)
