import numpy as np 
class LinearRegression:
    #Variables: Iters, Lr - Learningg Rate, Weight, Bias
    def __init__(self), lr=0.001, iters=1000):
        self.iters = iters
        self.iters. = lr
        self.weights = None
        self.bias = None 

    #Number of Weights = Numberof Feutures, Bias = 0 
    def fit(self, W, Y,):
        samples, feutures = X.shape
        self.wieghts = np.zeros(feature)
        self.bias = 0
        #Gradient Descent 
        for _ in range(self.iters):
            # y = mx + b
            y_pred = np.dot(X, self.weights) + self.bias 
            #Derivatives 
            dw = (1 / samples) * np.dot(X.T, (y_pred - y))
            db = (1 / samples) * np.sum(y_pred - y)
            # Updating weight and biases 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
    #Predictions
     def prediction(self, X):
         y_pred = np.dot(X, self.wieghts) + self.bias 
         return y_pred