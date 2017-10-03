from sklearn.decomposition import PCA
import numpy as np
from random import shuffle


class NoSignalException(Exception):
    pass

class NotTrainedException(Exception):
    pass

class PCAModel(object):

    def __init__(self):
        self.model = None

    def train(self, data):
        self.model = PCA(n_components=None)
        self.data = data
        n_features = len(data[0]["sample"])
        n_samples = len(data)
        self.training = np.zeros([n_samples, n_features])
        for i in range(n_samples):
            for j in range(n_features):
                self.training[i,j] = data[i]["sample"][j]
        self.model.fit(self.training)

    def classify(self, sample):
        if self.model == None:
            raise NotTrainedException()
        dt = self.project(np.array(self.data[0]["sample"])) - self.project(np.array(sample))
        mind = np.sqrt(dt.dot(dt))
        res = self.data[0]["label"]
        for t in self.data:
            dt = self.project(np.array(t["sample"])) - self.project(np.array(sample))
            d = np.sqrt(dt.dot(dt))
            if d < mind:
                mind = d
                res = t["label"]
        return (res, mind)

    def project(self, sample):
        eigens = self.get_eigenvectors()
        res = []
        for e in eigens:
            res.append(e.dot(np.array(sample) - self.get_mean()))
        return np.array(res)

    def get_eigenvectors(self):
        return self.model.components_

    def get_mean(self):
        return self.model.mean_
