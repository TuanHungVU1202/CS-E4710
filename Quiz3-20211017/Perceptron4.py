# from Mr.HuuTiep Vu
import numpy as np


class Perceptron4:

    def __init__(self):
        self.activation_func = self._unit_step_func
        self.w = None

    # def predict(self, X):
    #     """’ predict label of each row of X, given w
    #     X: a 2-d numpy array of shape (N, d), each row is a datapoint
    #     w_init: a 1-d numpy array of shape (d)  """
    #     return np.sign(X.dot(self.w))

    # def fit(self, X, y, w_init):
    def fit(self, X, y):
        """ perform perceptron learning algorithm
        X: a 2-d numpy array of shape (N, d), each row is a datapoint
        y: a 1-d numpy array of shape (N), label of each row of X. y[i] = 1/-1
        w_init: a 1-d numpy array of shape (d) """
        n_samples, n_features = X.shape

        # init parameters
        w_init = np.zeros(n_features)
        # w = w_init
        self.w = w_init
        # while True:
        for i in range (0,20):
            # pred = self.predict(w, X)
            pred = self.predict(X)
            # find indexes of misclassified points
            mis_idxs = np.where(np.equal(pred, y) == False)[0]
            # number of misclassified points
            num_mis = mis_idxs.shape[0]
            if num_mis == 0:  # no more misclassified points
                return w
            # random pick one misclassified point
            random_id = np.random.choice(mis_idxs, 1)[0]
            # update w
            # w = w + y[random_id] * X[random_id]
            w = self.w + y[random_id] * X[random_id]

# not from aTiep file
    def predict(self, X):
        linear_output = np.dot(X, self.w)
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)