import numpy as np


def dist_euclidiana(X, linha):
    X_ = (X - linha) ** 2
    return np.sqrt(np.sum(X_, axis=1))


class KNNRE:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def obter_idx_kNN(self, X, linha, k):
        dist_euc = dist_euclidiana(X, linha)
        idx_sort = np.argsort(dist_euc)
        return idx_sort[0:k]

    def regressao(X, y, linha, k):
        idx_kNN = obter_idx_kNN(X, linha, k=k)
        return np.mean(y[idx_kNN])

    def predict(self, X_train, y_train, X_test, k):
        lista = []
        for linha in X_test:
            lista.append(regressao(X_train, y_train, linha, k))
        return np.array(lista)
