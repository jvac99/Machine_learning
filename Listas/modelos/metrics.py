import numpy as np


def mae(y, y_pred):
    return np.sum(np.absolute(y - y_pred)) / y.shape[0]


def mae(y_verdadeiros, y_estimados):
    y_dif = np.array(y_verdadeiros) - np.array(y_estimados)
    return sum(abs(y_dif))/len(y_verdadeiros)


def mse(y, y_pred):
    return np.sum((y - y_pred) ** 2) / y.shape[0]


def rmse(y, y_pred):
    return mse(y, y_pred) ** 0.5
