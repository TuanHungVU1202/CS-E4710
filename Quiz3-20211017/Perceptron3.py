#https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1
import numpy as np


class Perceptron3:
    def fit(self, X, y, n_iter=20):

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Add 1 for the bias term
        self.weights = np.zeros((n_features + 1,))

        # Add column of 1s
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

        for i in range(n_iter):
            for j in range(n_samples):
                if y[j] * np.dot(self.weights, X[j, :]) <= 0:
                    self.weights += y[j] * X[j, :]

    def predict(self, X):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return

        n_samples = X.shape[0]
        # Add column of 1s
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)

        return y

    def score(self, X, y):
        pred_y = self.predict(X)

        return np.mean(y == pred_y)
