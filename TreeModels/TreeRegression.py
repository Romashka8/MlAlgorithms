import pandas as pd
import numpy as np

from typing import Optional

import TreeModelInterface


class TreeRegression(TreeModelInterface.TreeModelInterface):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = None):
        super().__init__(max_depth, min_samples_split, max_leafs, bins)

    def __repr__(self):
        # getting dict() of all attributes
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'TreeRegression class: ' + res

    @staticmethod
    def __get_mse(y: pd.Series) -> float:
        mse, n, y_mean = 0.0, y.shape[0], y.mean()
        for obj in y:
            mse += (obj - y_mean) ** 2
        return mse / y.shape[0] if y.shape[0] != 0 else 0

    def __get_best_split(self, x: pd.DataFrame, y: pd.Series) -> (str, float, float):
        if x.shape[0] != y.shape[0]: raise ValueError("X and y must be the same size")
        col_name, split_value, ig_max, ig_start = '', 0.0, 0.0, TreeRegression().__get_mse(y)
        N = y.shape[0]
        form_spliters = len(self.split_values) == 0
        for column in x.columns:
            if form_spliters:
                # unique feature values
                vals = np.sort(x[column].unique())
                if self.bins is not None and len(vals) - 1 >= self.bins:
                    spliters = np.histogram(x[column], bins=self.bins)[1][1:-1]
                else:
                    spliters = np.convolve(vals, [0.5, 0.5], 'valid')
                self.split_values[column] = spliters
            for spliter in self.split_values[column]:
                left, right = x[x[column] <= spliter].index, x[x[column] > spliter].index
                x_left, y_left = x.loc[left], y.loc[left]
                x_right, y_right = x.loc[right], y.loc[right]
                ig = ig_start - ((y_left.shape[0] / N) * TreeRegression().__get_mse(y_left) +
                                 (y_right.shape[0] / N) * TreeRegression().__get_mse(y_right))
                if ig > ig_max:
                    ig_max = ig
                    col_name = column
                    split_value = spliter
        return col_name, split_value, ig_max

    # true if we can turn node into leaf
    def __is_leaf(self, y: pd.Series, depth: int) -> bool:
        cond1 = y.shape[0] <= 1 or y.unique().shape[0] == 1
        cond2 = depth == self.max_depth
        cond3 = y.shape[0] < self.min_samples_split
        cond4 = self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs
        return cond1 or cond2 or cond3 or cond4

    def __form_tree(self, x: pd.DataFrame, y: pd.Series, node: TreeModelInterface.Node, depth: int):
        if not self.__is_leaf(y, depth):
            col_name, split_value, ig = self.__get_best_split(x, y)
            left, right = x[x[col_name] <= split_value].index, x[x[col_name] > split_value].index
            x_left, y_left = x.loc[left], y.loc[left]
            x_right, y_right = x.loc[right], y.loc[right]
            self.fi[col_name] += x.shape[0] / self.N * ig
            node.column, node.value = col_name, split_value
            node.left = TreeModelInterface.Node()
            node.right = TreeModelInterface.Node()
            self.leafs_cnt += 1
            self.__form_tree(x_left, y_left, node.left, depth + 1), self.__form_tree(x_right, y_right, node.right,
                                                                                     depth + 1)
        else:
            node.value = y.mean()
            self.leafs_sum += node.value
            return node

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        # for second and more fits we must erase our tree
        if self.tree_structure.column:
            self.tree_structure = TreeModelInterface.Node()
        for feature in x.columns:
            self.fi[feature] = 0
        self.N = x.shape[0]
        self.__form_tree(x, y, self.tree_structure, 0)
        return

    def __predict_row(self, row: pd.Series) -> float:
        tree = self.tree_structure
        while tree.column:
            if row[tree.column] <= tree.value:
                tree = tree.left
            else:
                tree = tree.right
        return tree.value

    def predict(self, x: pd.DataFrame) -> np.array:
        if self.tree_structure.column is None:
            raise TreeModelInterface.UnfittedModel()
        res = []
        for row in x.index:
            res.append(self.__predict_row(x.loc[row]))
        return np.array(res)
