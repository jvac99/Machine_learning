import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNR:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x))
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distances.append(euclidean_distance(x, x_train))
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        return np.mean(k_neighbor_labels)
