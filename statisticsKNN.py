import numpy as np 
from statistics import mode

def m_dist(a, b):
    return np.abs(a - b).sum()

class KNN:
    def _init_(self, k):
        self.k = k

def fit(self, X, y):
    self.X = X
    self.y = y

def predict(self, X):
    dist = [m_dist(x, X) for x in self.X]
    ki = np.argsort(dist)[: self.k]
    labels = [self.y[x] for x in ki]
    return mode(labels)

def predictions(self, X):
    return np.array(pred=[y for y in X])