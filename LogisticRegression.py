import numpy as np 

class LogisticRegression:
    def _init_(self, lr=0.001, iters=1000):
        self.lr = lr 
        self.iters = iters 
        self.weights = None 
        self.bias = None 

def fit(self, X, y):
    samples, features = X.shape
    self.weights = np.zeros(features)
    self.bias = 0 

    for _ in range(self.iters):
        L_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sig(L_model)

        dw = (1 / samples) * np.dot(X.T, y_pred - y)
        db = (1 / samples) * np.sum(y_pred - y)

        self.weights -= self.lr * dw 
        self.bias -= self.lr *db

def predict(self, X):
    L_model = np.dot(X, self.weights) + self.bias 
    y_pred = self.sig(L_model)

    y_pred_class = []

    for i in y_pred:
        if i > 0.5:
            y_pred_class.append(1)
        else:
            y_pred_class.append(0)
    return y_pred_class

def sig(self, x):
    return 1 / (1 + np.exp(-x))