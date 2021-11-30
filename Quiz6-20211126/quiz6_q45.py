import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# the data
from sklearn.datasets import make_blobs
# linear models
from sklearn.linear_model import Perceptron, LinearRegression
# multi-class models
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier


# Create the dataset
C = 4
n = 800
X, y = make_blobs(n, centers=C, random_state=0)

np.random.seed(0)
order = np.random.permutation(n)
tr = order[:int(n/2)]
tst = order[int(n/2):]

Xt = X[tst, :]
yt = y[tst]
X = X[tr, :]
y = y[tr]

# use perceptron with default parameters as the base classifier for the multi-class methods
linear_classifier = Perceptron()

# .....
