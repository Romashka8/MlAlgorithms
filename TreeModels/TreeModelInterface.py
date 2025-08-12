# ----------------------------------------------------------------------------------------------------------------------------------------

import abc
import pandas as pd
import numpy as np

from typing import Optional, Any

# ----------------------------------------------------------------------------------------------------------------------------------------

# make class for Tree structure
class Node:
    def __init__(self, column: Optional[str] = None, value: Optional[float] = None, left: Optional[Any] = None,
                 right: Optional[Any] = None):
        """
        :param column: feature name
        :param value: spliter value
        :param left: left subtree
        :param right: right subtree
        """

        self.column = column
        self.value = value
        self.left = left
        self.right = right

# ----------------------------------------------------------------------------------------------------------------------------------------

class UnfittedModel(BaseException):
    def __str__(self):
        return 'Model must be fitted first!'

# ----------------------------------------------------------------------------------------------------------------------------------------

class TreeModelInterface(abc.ABC):
    # can be extended in child classes
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = None):
        """
        :param max_depth: max tree depth
        :param min_samples_split: min samples in node for its possible splitting
        :param max_leafs: max leafs in tree
        :param bins: use or not split by histogram. if not None it must be an int
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        # adding dictionary for counting feature importance
        self.fi = {}
        # all rows in dataset, need for feature importance
        self.N = 0
        # tree have at least one node and two leafs
        if self.max_leafs <= 1:
            self.max_leafs = 2
            self.max_depth = 1
        self.split_values = {}
        self.leafs_cnt = 1
        self.leafs_sum = 0
        self.tree_structure = Node()

    # implemented in child classes, must provide object representation
    @abc.abstractmethod
    def __repr__(self):
        pass

    # using for tree output
    def tree_traversal(self, tree: Node = None, depth: int = 0, side: str = 'left'):
        
        if tree is None:
            tree = self.tree_structure
        
        if tree.column is not None:
            print(f"{' ' * depth}{tree.column} > {tree.value}")
            if tree.left is not None:
                self.tree_traversal(tree.left, depth + 1, 'left')
            if tree.right is not None:
                self.tree_traversal(tree.right, depth + 1, 'right')
        
        else:
            print(f"{' ' * depth}{side} leaf = {tree.value}")

    # implemented in child classes, fitting our model
    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        pass

    # implemented in child classes, predict values
    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> np.array:
        pass

# ----------------------------------------------------------------------------------------------------------------------------------------
