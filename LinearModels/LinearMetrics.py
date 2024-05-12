import numpy as np

from LinearModelnterface import BaseMetric


# Mean Absolute Error
class MAE(BaseMetric):
    name = 'MAE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return abs(y - y_pred).sum() / len(y)


# Mean Squared Error
class MSE(BaseMetric):
    name = 'MSE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return ((y - y_pred) ** 2).sum() / len(y)


# Root Mean Squared Error
class RMSE(BaseMetric):
    name = 'RMSE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return np.sqrt(((y - y_pred) ** 2).sum() / len(y))


# Mean Absolute Percentage Error
class MAPE(BaseMetric):
    name = 'MAPE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return abs((y - y_pred) / y).sum() / len(y) * 100


# R2 score
class R2(BaseMetric):
    name = 'R2'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
