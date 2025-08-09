# ----------------------------------------------------------------------------------------------------------------------------------------

import abc
import random
from typing import Union, Callable

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------

# Base class of a Linear Model
class LinearModelBase(abc.ABC):
    # implemented in child classes must include at least parameters below
    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        :param n_iter: steps quantity of gradient descent
        :type: int, default 100
        :param learning_rate: step in our gradient descent
            if learning_rate lambda function of kind like lambda iter: 0.5 * (0.85 ** iter)
            then we calc learning_rate in range [1, n_iter] (including 1 and n_iter)
        :type: float or typing.Callable(lambda function), default - None
        :param metric: metric using during learning
        :type: child class of LinearMetricBase
        :param sgd_sample: quantity of samples in mini-batch SGD
        :type: int or float, else None - this means that we use classic gradient descent
        :param random_state: random seed(by default in all child - 42)
        :type: int
        """
        pass

    # implemented in child classes, must provide object representation
    @abc.abstractmethod
    def __repr__(self):
        """
        :return: string representation
        """
        pass

    # Gradient Descent Algorithm realisation
    # Using in .fit() in child classes(LinearRegression Models)
    # reg can be only ('l1', 'l2', 'elasticnet')
    # Other parameters described in __init__
    # This gradient descent using in Regression
    @staticmethod
    def gradient_descent(x: pd.DataFrame, y: pd.Series, n_iter: int, learning_rate: Union[float, Callable],
                         sgd_sample: Union[int, float], verbose: int, metric, random_state: int, reg: str = None,
                         l1_coef: float = None, l2_coef: float = None) -> np.array:
        # fixing random seed
        random.seed(random_state)
        # adding free coef column
        x.insert(0, 'free_coef', [1] * x.shape[0])
        x, y = x.to_numpy(), y.to_numpy()
        n, n_features = x.shape[0], x.shape[1]
        weights_init = np.ones(n_features)

        for i in range(n_iter):
            y_pred = np.dot(x, weights_init)
            # adding regularisation
            if reg:
                if reg == 'l1':
                    reg = l1_coef * abs(weights_init).sum()
                    reg_grad = l1_coef * np.sign(weights_init)
                elif reg == 'l2':
                    reg = l2_coef * (weights_init ** 2).sum()
                    reg_grad = l2_coef * 2 * weights_init
                elif reg == 'elasticnet':
                    reg = l1_coef * abs(weights_init).sum() + l2_coef * (weights_init ** 2).sum()
                    reg_grad = l1_coef * np.sign(weights_init) + l2_coef * 2 * weights_init
            else:
                reg, reg_grad = 0, 0
            # calculating MSE
            loss = ((y - y_pred) ** 2).sum() / n + reg
            # calculating gradient and update our weights
            # mini-batch SGD(if True)
            if sgd_sample:
                if isinstance(sgd_sample, int):
                    sample_rows_idx = random.sample(range(x.shape[0]), sgd_sample)
                else:
                    sample_rows_idx = random.sample(range(x.shape[0]), int(np.round(sgd_sample * x.shape[0])))
                grag = 2 / len(sample_rows_idx) * np.dot((y_pred - y)[sample_rows_idx], x[sample_rows_idx]) + reg_grad
            else:
                grag = 2 / n * np.dot((y_pred - y), x) + reg_grad
            if isinstance(learning_rate, Callable):
                weights_init += (-1) * learning_rate(i + 1) * grag
            else:
                weights_init += (-1) * learning_rate * grag
            if verbose and i % verbose == 0:
                output = ''
                if i == 0:
                    output = f'start|loss: {loss}'
                else:
                    output = f'{i}|loss: {loss}'
                calc_metric = metric.calc(y_pred, y)
                output += f'|{metric.name}:{calc_metric}'
                print(output)

        return weights_init

    @staticmethod
    def gradient_descent_logistic(x: pd.DataFrame, y: pd.Series, n_iter: int, learning_rate: Union[float, Callable],
                         sgd_sample: Union[int, float], verbose: int, metric, random_state: int, reg: str = None,
                         l1_coef: float = None, l2_coef: float = None) -> np.array:
        # fixing random seed
        random.seed(random_state)
        # adding free coef column
        x.insert(0, 'free_coef', np.ones(x.shape[0]))
        x, y = x.to_numpy(), y.to_numpy()
        n, n_features = x.shape[0], x.shape[1]
        weights_init = np.ones(n_features)
        for i in range(n_iter):
            # predict probabilitys using sigmoida
            y_pred = 1 / (1 + np.exp(-np.dot(x, weights_init)))
            # calculating log loss
            # defending us from 0 in log
            if reg:
                if reg == 'l1':
                    reg = l1_coef * abs(weights_init).sum()
                    reg_grad = l1_coef * np.sign(weights_init)
                elif reg == 'l2':
                    reg = l2_coef * (weights_init ** 2).sum()
                    reg_grad = l2_coef * 2 * weights_init
                elif reg == 'elasticnet':
                    reg = l1_coef * abs(weights_init).sum() + l2_coef * (weights_init ** 2).sum()
                    reg_grad = l1_coef * np.sign(weights_init) + l2_coef * 2 * weights_init
            else:
                reg, reg_grad = 0, 0
            eps = 1e-15
            loss = (y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)).sum() / -n + reg
            if sgd_sample:
                if isinstance(sgd_sample, int):
                    sample_rows_idx = random.sample(range(x.shape[0]), sgd_sample)
                else:
                    sample_rows_idx = random.sample(range(x.shape[0]), int(np.round(sgd_sample * x.shape[0])))
                grad = np.dot((y_pred - y)[sample_rows_idx], x[sample_rows_idx]) / len(sample_rows_idx) + reg_grad
            else:
                grad = np.dot((y_pred - y), x) / n + reg_grad
            if isinstance(learning_rate, float):
                weights_init -= learning_rate * grad
            else:
                # in dynamic learning_rate change steps numeration
                weights_init -= learning_rate(i + 1) * grad
            # loss and metric output in learning process
            if verbose and i % verbose == 0:
                output = ''
                if i == 0:
                    output = f'start|loss: {loss}'
                else:
                    output = f'{i}|loss: {loss}'
                if metric.name != 'roc_auc':
                    y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
                calc_metric = metric.calc(y_pred, y)
                output += f'|{metric.name}:{calc_metric}'
                print(output)
        # y_pred = 1 / (1 + np.exp(-np.dot(x, weights_init)))
        # if metric:
        #     if metric != 'roc_auc':
        #         y_pred[y_pred > 0.5], y_pred[y_pred <= 0.5] = 1, 0
        #     best_metric = metrics[metric].calc(y_pred, y)
        return weights_init

    # implemented in child classes.
    # this method trains our models
    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        """
        :param x: features matrix
        :type: pandas.DataFrame
        :param y: target
        :type: pandas.DataSeries
        :param verbose: when(on what iteration) we output log with losses. Default = False - do not output.
        :type: int
        :return: None
        """
        pass

    # implemented in child classes
    # return our weights(except first cause it match a free coef)
    @abc.abstractmethod
    def get_coef(self) -> np.array:
        """
        :return: numpy.array
        """
        pass

    # implemented in child classes
    # return best metric result(after fitting)
    @abc.abstractmethod
    def get_best_score(self) -> float:
        """
        :return: float
        """
        pass

    # implemented in child classes
    # return predictions of fitted model
    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> np.array:
        """
        :param x: features matrix
        :return: numpy.array
        """
        pass

    # implemented in child classes
    # return score of fitted model
    @abc.abstractmethod
    def score(self, y_pred: np.array, y: np.array) -> float:
        """
        :param y_pred: predicted values
        :type: numpy.array
        :param y: real values
        :type: numpy.array
        :return: float - score
        """
        pass

# ----------------------------------------------------------------------------------------------------------------------------------------

# to prevent predict in unfitted models
class UnfittedModel(BaseException):
    def __str__(self):
        return 'Model must be fitted first!'

# ----------------------------------------------------------------------------------------------------------------------------------------

# we will represent our metrics as child classes of BaseMetric
class BaseMetric(abc.ABC):
    
    @staticmethod
    @abc.abstractmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        """
        :param y_pred: predicted values
        :type: numpy.array
        :param y: real values
        :type: numpy.array
        :return: float - metric score
        """
        pass

# ----------------------------------------------------------------------------------------------------------------------------------------
