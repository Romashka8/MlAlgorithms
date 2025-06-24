import numpy as np
import pandas as pd

from interface import PCA


# ----------------------------------------------------------------------------------------------------------------------------

class PCA():

	def __init__(self, n_components: int = 3):

		self.n_components = n_components

	def __repr__(self):
    
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
    
        return 'PCA class: ' + res

	def fit_transform(self, x: np.array | pd.Dataframe) -> np.array:

		if isinstance(x, pd.Dataframe): x = x.to_numpy()

		x_scaled = x - x.mean(axis=0)

        covariance_matrix = np.cov(x_scaled, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]

        selected_eigenvectors = sorted_eigenvectors[:, :self.n_components]

        x_transformed = pd.DataFrame(x_scaled @ selected_eigenvectors)

        return x_transformed

# ----------------------------------------------------------------------------------------------------------------------------
