import numpy as np
import pandas as pd

from LinearModelnterface import LinearModelBase, UnfittedModel, BaseMetric
import LinearMetrics as lm
from typing import Union, Callable


class LogisticRegression(LinearModelBase):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1,
                 metric: BaseMetric = lm.Accuracy(),
                 sgd_sample: Union[int, float] = None, random_state: int = 42) -> object:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_metric = None
        self.weights = None

    def __repr__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'MyLogReg class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent_logistic(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric)
        y_pred = 1 / (1 + np.exp(-np.dot(x, self.weights)))
        if self.metric.name != 'roc_auc':
            y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
        self.best_metric = self.metric.calc(y_pred, y)

    # return our weights(except first cause it match a free coef)
    def get_coef(self):
        return np.array(self.weights[1:])

    def get_best_score(self) -> float:
        return self.best_metric

    # return probabilitys(logiths) - vals throw sigmoida
    def predict_proba(self, x: pd.DataFrame) -> np.array:
        if self.weights is None:
            raise UnfittedModel()
        x = x.copy()
        x.insert(0, 'free_coef', np.ones(x.shape[0]))
        x = x.to_numpy()
        return 1 / (1 + np.exp(-np.dot(x, self.weights)))

    # return classes - if > 0.5 - positive, else - negative
    def predict(self, x: pd.DataFrame) -> np.array:
        if self.weights is None:
            raise UnfittedModel()
        x = x.copy()
        x.insert(0, 'free_coef', np.ones(x.shape[0]))
        x = x.to_numpy()
        probs = 1 / (1 + np.exp(-np.dot(x, self.weights)))
        probs[probs > 0.5], probs[probs <= 0.5] = 1, 0
        return np.array(list(map(int, probs)))

    # y_pred must contain probs!
    def score(self, y_pred: np.array, y: np.array) -> float:
        return lm.RocAuc().calc(y_pred, y)


class LogisticRegressionL1(LogisticRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1,
                 metric: BaseMetric = lm.Accuracy(),
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
        return 'LogisticRegressionL1 class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent_logistic(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='l1', l1_coef=self.l1_coef)
        y_pred = 1 / (1 + np.exp(-np.dot(x, self.weights)))
        if self.metric.name != 'roc_auc':
            y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
        self.best_metric = self.metric.calc(y_pred, y)


class LogisticRegressionL2(LogisticRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1,
                 metric: BaseMetric = lm.Accuracy(),
                 l2_coef: float = 0.1, sgd_sample: Union[int, float] = None, random_state: int = 42):
        """
        :param l2_coef: coefficient in L1 regularisation
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
        return 'LogisticRegressionL1 class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent_logistic(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='l2', l2_coef=self.l2_coef)
        y_pred = 1 / (1 + np.exp(-np.dot(x, self.weights)))
        if self.metric.name != 'roc_auc':
            y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
        self.best_metric = self.metric.calc(y_pred, y)


class LogisticRegressionElasticNet(LogisticRegression):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1,
                 metric: BaseMetric = lm.Accuracy(),
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
        return 'LogisticRegressionElasticNet class: ' + res

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        x = x.copy()
        self.weights = self.gradient_descent_logistic(
            x=x, y=y, n_iter=self.n_iter, learning_rate=self.learning_rate, sgd_sample=self.sgd_sample, verbose=verbose,
            random_state=self.random_state, metric=self.metric, reg='elasticnet', l1_coef=self.l1_coef,
            l2_coef=self.l2_coef)
        y_pred = 1 / (1 + np.exp(-np.dot(x, self.weights)))
        if self.metric.name != 'roc_auc':
            y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
        self.best_metric = self.metric.calc(y_pred, y)
