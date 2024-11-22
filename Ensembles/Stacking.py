import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, cross_val_predict


class Stacking:
    def __init__(self, estimators, final_estimator, blending: bool = False,
                 cv: int = 5, n_jobs: int = -1, random_state: int = 42):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.blending = blending
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _x_pred(self, estimator, data: list[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> np.array:
        if self.blending:
            x_train, y_train, x_val = data
            return estimator.fit(x_train, y_train).predict(x_val)
        else:
            x_train, y_train = data
            return cross_val_predict(estimator, x_train, y_train, cv=self.cv)

    @staticmethod
    def _x_test_pred(estimator, data: list[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> np.array:
        x_train, y_train, x_test = data
        return estimator.fit(x_train, y_train).predict(x_test)

    def _meta_data(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame):
        if self.blending:
            x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train,
                                                                  random_state=self.random_state)
            train_data = [x_train_v, y_train_v, x_val]
            test_data = [x_train_v, y_train_v, x_test]
            meta_y_train = y_val
        else:
            train_data = [x_train, y_train]
            test_data = [x_train, y_train, x_test]
            meta_y_train = y_train

        cv_x_train_preds = (delayed(self._x_pred)(est, train_data) for est in self.estimators)
        x_test_preds = (delayed(self._x_test_pred)(est, test_data) for est in self.estimators)

        meta_x_train = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(cv_x_train_preds))
        meta_x_test = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(x_test_preds))

        return meta_x_train.T, meta_y_train, meta_x_test.T

    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> np.array:
        meta_x_train, meta_y_train, meta_x_test = self._meta_data(x_train, y_train, x_test)

        return self.final_estimator.fit(meta_x_train, meta_y_train).predict(meta_x_test)

if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    from sklearn.metrics import mean_absolute_percentage_error

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(X, y, sep='\n')

    reg_estimators = [RandomForestRegressor(random_state=0),
                      GradientBoostingRegressor(random_state=0)]

    stacking_reg = Stacking(estimators=reg_estimators, final_estimator=RidgeCV())
    stacking_reg_pred_res = stacking_reg.fit_predict(X_train, y_train, X_test)
    stacking_mape = mean_absolute_percentage_error(y_test, stacking_reg_pred_res)
    print(f'stacking_regressor_mape {stacking_mape}')
    print(stacking_reg_pred_res, '', sep='\n')
