## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from Perceptron import Perceptron
from Perceptron2 import Perceptron2
## ###################################################

def predict(w, X):
    """’ predict label of each row of X, given w
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    w_init: a 1-d numpy array of shape (d)  """
    return np.sign(X.dot(w))
def perceptron1(X, y, w_init):
    """ perform perceptron learning algorithm
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    y: a 1-d numpy array of shape (N), label of each row of X. y[i] = 1/-1
    w_init: a 1-d numpy array of shape (d) """
    w = w_init
    while True:
        pred = predict(w, X)
        # find indexes of misclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        # number of misclassified points
        num_mis = mis_idxs.shape[0]
        if num_mis == 0: # no more misclassified points
            return w
        # random pick one misclassified point
        random_id = np.random.choice(mis_idxs, 1)[0]
        # update w
        w = w + y[random_id]*X[random_id]

#################
def step_func(z):
    return 1.0 if (z > 0) else 0.0

def perceptron2(X, y, lr, epochs):
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n + 1, 1))

    # Empty list to store how many examples were
    # misclassified at every iteration.
    n_miss_list = []

    # Training.
    for epoch in range(epochs):

        # variable to store #misclassified.
        n_miss = 0

        # looping for every example.
        for idx, x_i in enumerate(X):

            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # Calculating prediction/hypothesis.
            y_hat = step_func(np.dot(x_i.T, theta))

            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += lr * ((y[idx] - y_hat) * x_i)

                # Incrementing by 1.
                n_miss += 1

        # Appending number of misclassified examples
        # at every iteration.
        n_miss_list.append(n_miss)

    return theta, n_miss_list

################
# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
## to convert the {0,1} output into {-1,+1}
y = 2 * y - 1

print(X.shape, y.shape)
mdata, ndim = X.shape

nfold = 5  ## number of folds
## initialize the learning parameters for all folds
f1 = np.zeros(nfold)
maxmargin_train = np.zeros(nfold)
## split the data into 5-folds
cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)

"""
To do ....
"""

perceptron = Perceptron()
perceptron2 = Perceptron2()
for train_index, test_index in cselection.split(X):
    # perceptron.fit(X[train_index], y[train_index], 20, 1)
    perceptron2.fit(X[train_index], y[train_index])
    # Ypred = perceptron.predict(X[test_index])
    Ypred = perceptron2.predict(X[test_index])
    f1 = f1_score(y[test_index], Ypred, average='macro')


print('The average F1:', np.mean(f1))
print('The average maximum margin achieved in the training:', np.mean(maxmargin_train))



