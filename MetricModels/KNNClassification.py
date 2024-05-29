import pandas as pd
import numpy as np

import warnings
import MetricModelnterface

warnings.filterwarnings('ignore')


class KNNClassification(MetricModelnterface.MetricModelBase):
    def __repr__(self):
        atr = self.__dict__
        res = ''.join([i + '=' + str(atr[i]) + ',' + ' ' for i in atr])[:-2]
        return 'KNNClassification class: ' + res

    def predict(self, x: pd.DataFrame) -> np.array:
        if self.features_train is None or self.target_train is None:
            raise MetricModelnterface.UnfittedModel()
        predicted_classes = []
        x = x.copy()
        x_train = self.features_train.to_numpy()
        used_metric = self.metrics[self.metric]
        for i in range(x.shape[0]):
            if self.metric == 'chebyshev':
                distances = np.max(used_metric(x.iloc[i], x_train), axis=1)
            elif self.metric == 'cosine':
                distances = 1 - np.sum(used_metric(x.iloc[i], x_train), axis=1) / (
                            np.sqrt(np.sum(x.iloc[i] ** 2)) * np.sqrt(np.sum(x_train ** 2, axis=1)))
            else:
                # finding for each object in test data distance for object in train data
                # in metric func we do not sum, for this reason we use sum here
                distances = np.sum(used_metric(x.iloc[i], x_train), axis=1)
            indexes = np.arange(0, self.train_size[0])
            res = pd.DataFrame(data={'index': indexes, 'distance': distances})
            res = res.sort_values(by=['distance'], ascending=True)
            nearest_neighbors = res[:self.k]['index']
            nearest_neighbors_dist = res[:self.k]['distance']
            nearest_neighbors_classes = self.target_train.iloc[nearest_neighbors].to_numpy()
            if self.weight == 'uniform':
                negative = len(nearest_neighbors_classes[nearest_neighbors_classes == 0])
                positive = len(nearest_neighbors_classes[nearest_neighbors_classes == 1])
                # calculating mode
                predicted_class = 1 if positive >= negative else 0
            elif self.weight == 'rank':
                count_all = lambda x: 1 / x[0]
                count_pos = lambda x: 1 / x[0] if x[1] == 1 else 0
                ranks = list(enumerate(nearest_neighbors_classes, start=1))
                all = sum(list(map(count_all, ranks)))
                pos = sum(list(map(count_pos, ranks)))
                predicted_class = 1 if (pos / all) > (1 - (pos / all)) else 0
            elif self.weight == 'distance':
                count_all = lambda x: 1 / x[1]
                count_pos = lambda x: 1 / x[1] if x[0] == 1 else 0
                dst = list(zip(nearest_neighbors_classes, nearest_neighbors_dist.to_numpy()))
                all = sum(list(map(count_all, dst)))
                pos = sum(list(map(count_pos, dst)))
                predicted_class = 1 if (pos / all) > (1 - (pos / all)) else 0
            predicted_classes.append(predicted_class)
        return np.array(predicted_classes)

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        if self.features_train is None or self.target_train is None:
            raise MetricModelnterface.UnfittedModel()
        predicted_probs = []
        x = x.copy()
        x_train = self.features_train.to_numpy()
        used_metric = self.metrics[self.metric]
        for i in range(x.shape[0]):
            # finding for each object in test data distance for object in train data
            # in metric func we do not sum, for this reason we use sum here
            if self.metric == 'chebyshev':
                distances = np.max(used_metric(x.iloc[i], x_train), axis=1)
            elif self.metric == 'cosine':
                distances = 1 - np.sum(used_metric(x.iloc[i], x_train), axis=1) / (
                            np.sqrt(np.sum(x.iloc[i] ** 2)) * np.sqrt(np.sum(x_train ** 2, axis=1)))
            else:
                distances = np.sum(used_metric(x.iloc[i], x_train), axis=1)
            indexes = np.arange(0, self.train_size[0])
            res = pd.DataFrame(data={'index': indexes, 'distance': distances})
            res = res.sort_values(by=['distance'], ascending=True)
            nearest_neighbors = res[:self.k]['index']
            nearest_neighbors_dist = res[:self.k]['distance']
            nearest_neighbors_classes = self.target_train.iloc[nearest_neighbors].to_numpy()
            if self.weight == 'uniform':
                positive = len(nearest_neighbors_classes[nearest_neighbors_classes == 1])
                # calculating prob of class 1
                prob = positive / len(nearest_neighbors_classes)
            elif self.weight == 'rank':
                count_all = lambda x: 1 / x[0]
                count_pos = lambda x: 1 / x[0] if x[1] == 1 else 0
                ranks = list(enumerate(nearest_neighbors_classes, start=1))
                all = sum(list(map(count_all, ranks)))
                pos = sum(list(map(count_pos, ranks)))
                prob = pos / all
            elif self.weight == 'distance':
                count_all = lambda x: 1 / x[1]
                count_pos = lambda x: 1 / x[1] if x[0] == 1 else 0
                dst = list(zip(nearest_neighbors_classes, nearest_neighbors_dist.to_numpy()))
                all = sum(list(map(count_all, dst)))
                pos = sum(list(map(count_pos, dst)))
                prob = pos / all
            predicted_probs.append(prob)
        return np.array(predicted_probs)