# ----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from LinearModelnterface import BaseMetric

# ----------------------------------------------------------------------------------------------------------------------------------------

# Regression Metrics
# Mean Absolute Error
class MAE(BaseMetric):
    name = 'MAE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return abs(y - y_pred).sum() / len(y)

# ----------------------------------------------------------------------------------------------------------------------------------------

# Mean Squared Error
class MSE(BaseMetric):
    name = 'MSE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return ((y - y_pred) ** 2).sum() / len(y)

# ----------------------------------------------------------------------------------------------------------------------------------------

# Root Mean Squared Error
class RMSE(BaseMetric):
    name = 'RMSE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return np.sqrt(((y - y_pred) ** 2).sum() / len(y))

# ----------------------------------------------------------------------------------------------------------------------------------------

# Mean Absolute Percentage Error
class MAPE(BaseMetric):
    name = 'MAPE'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return abs((y - y_pred) / y).sum() / len(y) * 100

# ----------------------------------------------------------------------------------------------------------------------------------------

# R2 score
class R2(BaseMetric):
    name = 'R2'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        return 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

# ----------------------------------------------------------------------------------------------------------------------------------------

# LogisticRegression Metrics
# Accuracy
class Accuracy(BaseMetric):
    name = 'accuracy'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        # True Positive
        tp = len(y_pred[(y_pred == 1) & (y == 1)])
        # True Negative
        tn = len(y_pred[(y_pred == 0) & (y == 0)])
        # False Positive
        fp = len(y_pred[(y_pred == 1) & (y == 0)])
        # False Negative
        fn = len(y_pred[(y_pred == 0) & (y == 1)])
        return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

# ----------------------------------------------------------------------------------------------------------------------------------------

# Precision
class Precision(BaseMetric):
    name = 'precision'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        # True Positive
        tp = len(y_pred[(y_pred == 1) & (y == 1)])
        # False Positive
        fp = len(y_pred[(y_pred == 1) & (y == 0)])
        return tp / (tp + fp)

# ----------------------------------------------------------------------------------------------------------------------------------------

# Recall
class Recall(BaseMetric):
    name = 'recall'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        # True Positive
        tp = len(y_pred[(y_pred == 1) & (y == 1)])
        # False Negative
        fn = len(y_pred[(y_pred == 0) & (y == 1)])
        return tp / (tp + fn)

# ----------------------------------------------------------------------------------------------------------------------------------------

# F1 score - particular case of F score
class F1(BaseMetric):
    name = 'f1'

    @staticmethod
    def calc(y_pred: np.array, y: np.array) -> float:
        precision = Precision().calc(y_pred, y)
        recall = Recall().calc(y_pred, y)
        return 2 * (precision * recall) / (precision + recall)

# ----------------------------------------------------------------------------------------------------------------------------------------

# Area Under ROC Curve
class RocAuc(BaseMetric):
    name = 'roc_auc'

    @staticmethod
    def calc(y_proba: np.array, y_class: np.array) -> float:
        y_proba = y_proba.round(10)
        # df = pd.concat([pd.Series(y_proba), pd.Series(y_class)], axis=1)
        df = pd.DataFrame({'0': y_proba, '1': y_class})
        df = df.sort_values(by='0', ascending=False)

        positives = df[df['1'] == 1]
        negatives = df[df['1'] == 0]
        score = 0

        for current_score in negatives['0']:
            higher = (positives > current_score).sum()
            equal = (positives == current_score).sum()
            score += higher + equal * 0.5
        score /= (positives.shape[0] * negatives.shape[0])
        return score.iloc[0]

# ----------------------------------------------------------------------------------------------------------------------------------------
