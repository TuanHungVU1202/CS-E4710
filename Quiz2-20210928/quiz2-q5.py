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
dataset_size = 50
train = order[:dataset_size]

test = order[dataset_size:]

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
# bc = binary_classifier()
# bc.fit(Xtr, ytr)  # train the classifier on training data
# preds = bc.predict(Xtst)  # predict with test data

# ========================================================================
# setup for analysing the Rademacher complexity

# consider these sample sizes
print_at_n = [20, 50, 100]
# when analysing Rademacher complexity, take always n first samples from training set, n as in this array

delta = 0.05
# get dimension
d = len(Xtr[0])
perceptron = d + 1
# todo solution
# for each data in training sets
count = 0
# Xtraining = Xtr[:dataset_size,:]
# Ytraining = ytr[:dataset_size]
emp_risk = np.zeros(100)
total_emp_risk = 0
bc = binary_classifier()
d = 2 + 1 #Perceptron VC dim is d + 1
for rango in range(0, 100):
    bc.fit(Xtr, ytr) #Now with training samples, not random labels
    Ypred = bc.predict(Xtr) #Seeing how good is the model
    for j in range(0, dataset_size):
        if not Ypred[j] == ytr[j]: #Seeing if these match or not and counting them
            count += 1
    emp_risk[rango] = count/dataset_size
    count = 0
for each in range(0, 100):
    total_emp_risk += emp_risk[each]
aver_emp_risk = total_emp_risk/100
term_one = math.sqrt((2*math.log((2.71828*dataset_size)/d))/(dataset_size/d)) #Second term of the formula
term_two = math.sqrt((math.log(1/0.05))/(2 * dataset_size)) #Third term of the formula
bound = aver_emp_risk + term_one + term_two #VC Generalization bound
print("Average risk: {}, Term one: {}, Term two: {}".format(aver_emp_risk,term_one,term_two))
print("VC-Dimension GB " + str(dataset_size) + " samples: " + str(bound))