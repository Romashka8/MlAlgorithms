import pandas as pd
import warnings

import KNNRegression, KNNClassification

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    from sklearn.datasets import make_regression, make_classification

    x, y = make_regression(n_samples=10, n_features=5, n_informative=3, noise=0.5, random_state=42)
    x, y = pd.DataFrame(x), pd.Series(y)
    test = KNNRegression.KNNRegression(metric='cosine')
    test.fit(x, y)
    print(test.predict(x))

    x, y = make_classification(n_samples=10, n_features=5, n_informative=3, random_state=42)
    x, y = pd.DataFrame(x), pd.Series(y)
    test = KNNClassification.KNNClassification(weight='rank')
    test.fit(x, y)
    print(test.predict(x))
