import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


class Perceptron:
    # constructor
    def __init__(self):
        self.w = None
        self.b = None

    # model
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    # predictor to predict on the data based on w
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y, epochs=1, lr=1):
        self.w = np.ones(X.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print("Max Accuracy", max_accuracy)
        #
        # plt.plot(list(accuracy.values()))
        # plt.ylim([0, 1])
        # plt.show()