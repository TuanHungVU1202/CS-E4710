import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ========================================================================
# dataset

n_tot = 200
# two blobs, not completely separated
X, y = make_blobs(n_tot, centers=2, cluster_std=3.0, random_state=2)

# plt.figure()
# colors = ["g", "b"]
# for ii in range(2):
#     class_indices = np.where(y==ii)[0]
#     plt.scatter(X[class_indices, 0], X[class_indices, 1], c=colors[ii])
# plt.title("full dataset")
# plt.show()

# divide data into training and testing
# NOTE! Test data is not needed in solving the exercise
# But it can be interesting to investigating how that behaves w.r.t. training set
# performance and the bounds :)
np.random.seed(42)
order = np.random.permutation(n_tot)
train = order[:100]
# test = order[100:]

Xtr = X[train, :]
ytr = y[train]
# Xtst = X[test, :]
# ytst = y[test]

# ========================================================================
# classifier

# The perceptron algorithm will be encountered later in the course
# How exactly it works is not relevant yet, it's enough to just know it's a binary classifier
from sklearn.linear_model import Perceptron as binary_classifier

# # It can be used like this:
# bc = binary_classifier()
# bc.fit(Xtr, ytr)  # train the classifier on training data
# preds = bc.predict(Xtst)  # predict with test data

# ========================================================================
# setup for analysing the Rademacher complexity

# consider these sample sizes
print_at_n = [20, 50, 100]
# when analysing Rademacher complexity, take always n first samples from training set, n as in this array

delta = 0.05
size = 100
total = 0
for counter in range(size):
    # todo solution
    count = 0  # We create the variable count for the average risk
    Xtraining = Xtr[:50, :]
    Ytraining = ytr[:50]
    Emp_risk = np.zeros(100)  # Initializing an array for empirical risk
    Yrandom = np.zeros(50)  # Initializing an array for random labels
    Aver_risk = 0  # Average of empirical risk
    bc = binary_classifier()  # Classifier
    for rango in range(0, 100):  # To calculate the average risks, 100 samples of random sets
        for length in range(0, 50):
            Yrandom[length] = random.randint(0, 1)  # Preparing the random labels
        bc.fit(Xtraining, Yrandom)  # Creating the model
        Ypred = bc.predict(Xtraining)  # Seeing how well the model predicts (test data is not necessary)
        for j in range(0, 50):  # Comparing both labels to count
            if not Ypred[j] == Yrandom[j]:
                count += 1  # Counting
        Emp_risk[rango] = count / 50  # Calculating empirical risk for each M sample
    for each in range(0, 100):  # Sum all the empirical risks for the numerator of average
        Aver_risk += Emp_risk[each]
    T_aver_risk = Aver_risk / 100  # Average empirical risk
    Rademacher_risk = (1 / 2) - T_aver_risk  # Second term of Rademacher
    Term = 3 * math.sqrt((math.log(2 / 0.05)) / (2 * 50))  # Third term of Rademacher(natural logarithm)
    R = T_aver_risk + Rademacher_risk + Term  # Generalization bound
    total = total + R

mean_total = total/size
print(mean_total)