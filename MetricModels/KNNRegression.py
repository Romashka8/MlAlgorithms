# ----------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import MetricModelnterface

# ----------------------------------------------------------------------------------------------------------------------------------------

class KNNRegression(MetricModelnterface.MetricModelBase):

    def predict(self, x: pd.DataFrame) -> np.array:

        if self.features_train is None or self.target_train is None:
            raise MetricModelnterface.UnfittedModel()
        
        predicted_vals = []
        x = x.copy()
        x_train = self.features_train.to_numpy()
        used_metric = self.metrics[self.metric]
        
        for i in range(x.shape[0]):
        
            if self.metric == 'chebyshev':
                distances = np.max(used_metric(x.iloc[i], x_train), axis=1)
        
            elif self.metric == 'cosine':
                distances = 1 - np.sum(used_metric(x.iloc[i], x_train), axis=1) / (
                        np.sqrt(np.sum(x.iloc[i] ** 2)) * np.sqrt(np.sum(x_train ** 2, axis=1)))
        
            elif self.metric == 'euclidean':
                distances = np.sqrt(np.sum(used_metric(x.iloc[i], x_train), axis=1))
        
            else:
                # finding for each object in test data distance for object in train data
                # in metric func we do not sum, for this reason we use sum here
                distances = np.sum(used_metric(x.iloc[i], x_train), axis=1)
        
            indexes = np.arange(0, self.train_size[0])
            res = pd.DataFrame(data={'index': indexes, 'distance': distances})
            res = res.sort_values(by=['distance'], ascending=True)
            nearest_neighbors = res[:self.k]['index']
            nearest_neighbors_dist = res[:self.k]['distance']
            nearest_neighbors_targets = self.target_train.iloc[nearest_neighbors].to_numpy()
        
            if self.weight == 'uniform':
                predicted_val = sum(nearest_neighbors_targets) / self.k
        
            elif self.weight == 'rank':
                count_all = lambda x: 1 / x[0]
                ranks = list(enumerate(nearest_neighbors_targets, start=1))
                all_weights = sum(list(map(count_all, ranks)))
                weights = np.array(list(map(count_all, ranks)))
                weights /= all_weights
                predicted_val = np.dot(nearest_neighbors_targets, weights)
        
            elif self.weight == 'distance':
                count_all = lambda x: 1 / x
                all_weights = sum(list(map(count_all, nearest_neighbors_dist)))
                weights = np.array(list(map(count_all, nearest_neighbors_dist)))
                weights /= all_weights
                predicted_val = np.dot(nearest_neighbors_targets, weights)
        
            predicted_vals.append(predicted_val)
        
        return np.array(predicted_vals)

    def __repr__(self):

        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        
        return 'KNNRegression class: ' + res

# ----------------------------------------------------------------------------------------------------------------------------------------
