import pandas as pd
from sklearn.datasets import make_regression

from LinearRegression import LinearRegression, LinearRegressionL1, LinearRegressionL2, LinearRegressionElasticNet
import LinearMetrics as lm

# test
if __name__ == '__main__':
    X, y = make_regression(n_samples=1000, n_features=15, n_informative=10, noise=12, random_state=42)
    X, y = pd.DataFrame(X), pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    lin_reg = LinearRegression(n_iter=200, learning_rate=lambda iter: 0.5 * (0.8 ** iter), metric=lm.MAPE())
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X)
    print('LinearRegression', lin_reg.score(y_pred, y))

    lin_reg_l1 = LinearRegressionL1(n_iter=1000, learning_rate=0.01, metric=lm.MAE(), l1_coef=0.5, sgd_sample=100)
    lin_reg_l1.fit(X, y)
    y_pred = lin_reg_l1.predict(X)
    print('LinearRegressionL1', lin_reg_l1.score(y_pred, y))

    lin_reg_l2 = LinearRegressionL2(l2_coef=0.3)
    lin_reg_l2.fit(X, y)
    y_pred = lin_reg.predict(X)
    print('LinearRegressionL2', lin_reg_l2.score(y_pred, y))

    lin_reg_elasticnet = LinearRegressionElasticNet(l1_coef=0.1, l2_coef=0.3)
    lin_reg_elasticnet.fit(X, y)
    y_pred = lin_reg_elasticnet.predict(X)
    print('LinearRegressionElasticNet', lin_reg_elasticnet.score(y_pred, y))