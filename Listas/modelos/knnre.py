from collections import Counter
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

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self._regressao(x))
        return np.array(y_pred)

    def _regressao(self, x):
        idx_kNN = self._obter_idx_kNN(x)
        return np.mean(self.y_train[idx_kNN])

    def _obter_idx_kNN(self, x):
        dist_euc = dist_euclidiana(self.X_train, x)
        idx_sort = np.argsort(dist_euc)
        return idx_sort[0:self.k]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_boston()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3
    clf = KNNRE(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNNRE Regressor accuracy", accuracy(y_test, predictions))
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("KNN Regressor sklearn accuracy", accuracy(y_test, y_pred))
