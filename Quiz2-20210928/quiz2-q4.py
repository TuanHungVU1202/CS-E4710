import array
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
# training set set to 20, 50 and 100
dataset_size = 100
train = order[:dataset_size]
# test = order[100:]

Xtr = X[train, :]
ytr = y[train]
# print(ytr.astype(str))
# Xtst = X[test, :]
# ytst = y[test]

# ========================================================================
# classifier

# The perceptron algorithm will be encountered later in the course
# How exactly it works is not relevant yet, it's enough to just know it's a binary classifier
from sklearn.linear_model import Perceptron as binary_classifier

# # It can be used like this:
bc = binary_classifier()
# bc.fit(Xtr, ytr)  # train the classifier on training data
# preds = bc.predict(Xtst)  # predict with test data

# ========================================================================
# setup for analysing the Rademacher complexity

# consider these sample sizes
print_at_n = [20, 50, 100]
# when analysing Rademacher complexity, take always n first samples from training set, n as in this array

delta = 0.05

# todo solution
# for each data in training sets
size = 100
total = 0
for counter in range(size):
    index = 0
    random_label_arr = np.zeros(dataset_size)
    number_of_misclassification = 0
    for index in range(dataset_size):
        # generate random label
        random_label = random.choice([0, 1])
        random_label_arr[index] = random_label
        index += 1
    bc.fit(Xtr, random_label_arr)
    Ypred = bc.predict(Xtr)
    for index in range(dataset_size):
        if Ypred[index] != random_label_arr[index]:
            number_of_misclassification += 1
    empirical_error = number_of_misclassification/dataset_size
    # R risk
    rademacher_complexity = 1/2 - number_of_misclassification/dataset_size
    third_term = 3 * math.sqrt((math.log(2/0.05))/(2*dataset_size))

    # R bound
    r_bound = empirical_error + rademacher_complexity + third_term
    total = total + r_bound

mean_total = total/size
print(mean_total)