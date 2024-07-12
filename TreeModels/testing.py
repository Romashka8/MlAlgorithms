import pandas as pd

from TreeClassification import TreeClassificator
from TreeRegression import TreeRegression

# Extended tests
df = pd.read_csv('data_banknote_authentication.txt', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']

t_clf = TreeClassificator(max_depth=5, min_samples_split=200, max_leafs=10, bins=4)
t_clf.fit(X, y)
t_clf.tree_traversal()
print(t_clf.leafs_cnt, t_clf.leafs_sum)

# Extended tests
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']

t_reg = TreeRegression(max_depth=15, min_samples_split=35, max_leafs=30, bins=6)
t_reg.fit(X, y)
t_reg.tree_traversal()
print(t_reg.leafs_cnt, t_reg.leafs_sum)
