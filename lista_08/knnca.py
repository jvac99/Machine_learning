import numpy as np


def dist_euclidiana(X, linha):
    X_ = (X - linha) ** 2
    return np.sqrt(np.sum(X_, axis=1))


class KNNCA:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self._classifica(x))
        return np.array(y_pred)

    def _classifica(self, x):
        idx_kNN = self._obter_idx_kNN(x)
        count = np.bincount(self.y_train[idx_kNN])
        return np.argmax(count)

    def _obter_idx_kNN(self, x):
        dist_euc = dist_euclidiana(self.X_train, x)
        idx_sort = np.argsort(dist_euc)
        return idx_sort[0:self.k]
