import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
import math

# ========================================================================
# dataset

n_tot = 200
# two blobs, not completely separated
X, y = make_blobs(n_tot, centers=2, cluster_std=3.0, random_state=2)

plt.figure()
colors = ["g", "b"]
for ii in range(2):
    class_indices = np.where(y==ii)[0]
    plt.scatter(X[class_indices, 0], X[class_indices, 1], c=colors[ii])
plt.title("full dataset")
plt.show()

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

# todo solution

#TODO: Question 4 (We try first with 20 samples, the others are only changing 20 for m samples)
count = 0 #We create the variable count for the average risk
Xtraining = Xtr[:20,:]
Ytraining = ytr[:20]
Yrandom = np.zeros(20) #Initializing an array for random labels
Epsilon = np.zeros(100)#Initializing an array for empirical risk
Total_epsilon = 0 #Average of empirical risk
bc = binary_classifier() #Classifier
for rango in range(0, 100): #To calculate the average risks, 100 samples of random sets
    for length in range(0,20):
        Yrandom[length] = random.randint(0, 1) #Preparing the random labels
    bc.fit(Xtraining, Yrandom) #Creating the model
    Ypred = bc.predict(Xtraining) #Seeing how well the model predicts (test data is not necessary)
    for j in range(0, 20): #Comparing both labels to count
        if not Ypred[j] == Yrandom[j]:
            count += 1 #Counting
    Epsilon[rango] = count/20 #Calculating empirical risk for each M sample
    count = 0
for each in range(0, 100):
    Total_epsilon += Epsilon[each]
bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
Ypred = bc.predict(Xtraining) #Seeing how good is the model
for j in range(0, 20):
    if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
        count += 1
    Emp_risk = count/20
Average_epsilon = Total_epsilon/100
Rademacher_risk = (1/2) - Average_epsilon #Second term of Rademacher
Term = 3 * math.sqrt((math.log(2/0.05))/(2*20)) #Third term of Rademacher(natural logarithm)
R = Emp_risk + Rademacher_risk + Term #Generalization bound
print("Rademacher " + str(20) + " samples: " + str(R))


count = 0 #We create the variable count for the average risk
Xtraining = Xtr[:50,:]
Ytraining = ytr[:50]
Yrandom = np.zeros(50) #Initializing an array for random labels
Epsilon = np.zeros(100)#Initializing an array for empirical risk
Total_epsilon = 0 #Average of empirical risk
bc = binary_classifier() #Classifier
for rango in range(0, 100): #To calculate the average risks, 100 samples of random sets
    for length in range(0,50):
        Yrandom[length] = random.randint(0, 1) #Preparing the random labels
    bc.fit(Xtraining, Yrandom) #Creating the model
    Ypred = bc.predict(Xtraining) #Seeing how well the model predicts (test data is not necessary)
    for j in range(0, 50): #Comparing both labels to count
        if not Ypred[j] == Yrandom[j]:
            count += 1 #Counting
    Epsilon[rango] = count/50 #Calculating empirical risk for each M sample
    count = 0
for each in range(0, 100):
    Total_epsilon += Epsilon[each]
bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
Ypred = bc.predict(Xtraining) #Seeing how good is the model
for j in range(0, 50):
    if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
        count += 1
    Emp_risk = count/50
Average_epsilon = Total_epsilon/100
Rademacher_risk = (1/2) - Average_epsilon #Second term of Rademacher
Term = 3 * math.sqrt((math.log(2/0.05))/(2*50)) #Third term of Rademacher(natural logarithm)
R = Emp_risk + Rademacher_risk + Term #Generalization bound
print("Rademacher " + str(50) + " samples: " + str(R))

count = 0 #We create the variable count for the average risk
Xtraining = Xtr[:100,:]
Ytraining = ytr[:100]
Yrandom = np.zeros(100) #Initializing an array for random labels
Epsilon = np.zeros(100)#Initializing an array for empirical risk
Total_epsilon = 0 #Average of empirical risk
bc = binary_classifier() #Classifier
for rango in range(0, 100): #To calculate the average risks, 100 samples of random sets
    for length in range(0,100):
        Yrandom[length] = random.randint(0, 1) #Preparing the random labels
    bc.fit(Xtraining, Yrandom) #Creating the model
    Ypred = bc.predict(Xtraining) #Seeing how well the model predicts (test data is not necessary)
    for j in range(0, 100): #Comparing both labels to count
        if not Ypred[j] == Yrandom[j]:
            count += 1 #Counting
    Epsilon[rango] = count/100 #Calculating empirical risk for each M sample
    count = 0
for each in range(0, 100):
    Total_epsilon += Epsilon[each]
bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
Ypred = bc.predict(Xtraining) #Seeing how good is the model
for j in range(0, 100):
    if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
        count += 1
    Emp_risk = count/100
Average_epsilon = Total_epsilon/100
Rademacher_risk = (1/2) - Average_epsilon #Second term of Rademacher
Term = 3 * math.sqrt((math.log(2/0.05))/(2*100)) #Third term of Rademacher(natural logarithm)
R = Emp_risk + Rademacher_risk + Term #Generalization bound
print("Rademacher " + str(20) + " samples: " + str(R))






#TODO: Question 5
count = 0
Xtraining = Xtr[:20,:]
Ytraining = ytr[:20]
Emp_risk = np.zeros(100)
Total_emp_risk = 0
bc = binary_classifier()
d = 2 + 1 #Perceptron VC dim is d + 1
for rango in range(0, 100):
    bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
    Ypred = bc.predict(Xtraining) #Seeing how good is the model
    for j in range(0, 20):
        if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
            count += 1
    Emp_risk[rango] = count/20
    count = 0
for each in range(0, 100):
    Total_emp_risk += Emp_risk[each]
T_aver_risk = Total_emp_risk/100
Term_one = math.sqrt((2*math.log((2.71828*20)/d))/(20/d)) #Second term of the formula
Term_two = math.sqrt((math.log(1/0.05))/(2 * 20)) #Third term of the formula
R = T_aver_risk + Term_one + Term_two #VC Generalization bound
print("Average risk: {}, Term one: {}, Term two: {}".format(T_aver_risk,Term_one,Term_two))
print("VC-Dimension GB " + str(20) + " samples: " + str(R))

count = 0
Xtraining = Xtr[:50,:]
Ytraining = ytr[:50]
Emp_risk = np.zeros(100)
Yrandom = np.zeros(50)
Total_emp_risk = 0
bc = binary_classifier()
d = 2 + 1 #Perceptron VC dim is d + 1
for rango in range(0, 100):
    bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
    Ypred = bc.predict(Xtraining) #Seeing how good is the model
    for j in range(0, 50):
        if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
            count += 1
    Emp_risk[rango] = count/50
    count = 0
for each in range(0, 100):
    Total_emp_risk += Emp_risk[each]
T_aver_risk = Total_emp_risk/100
Term_one = math.sqrt((2*math.log((2.71828*50)/d))/(50/d)) #Second term of the formula
Term_two = math.sqrt((math.log(1/0.05))/(2 * 50)) #Third term of the formula
R = T_aver_risk + Term_one + Term_two #VC Generalization bound
print("Average risk: {}, Term one: {}, Term two: {}".format(T_aver_risk,Term_one,Term_two))
print("VC-Dimension GB " + str(50) + " samples: " + str(R))

count = 0
Xtraining = Xtr[:100,:]
Ytraining = ytr[:100]
Emp_risk = np.zeros(100)
Yrandom = np.zeros(100)
Total_emp_risk = 0
bc = binary_classifier()
d = 2 + 1 #Perceptron VC dim is d + 1
for rango in range(0, 100):
    bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
    Ypred = bc.predict(Xtraining) #Seeing how good is the model
    for j in range(0, 100):
        if not Ypred[j] == Ytraining[j]: #Seeing if these match or not and counting them
            count += 1
    Emp_risk[rango] = count/100
    count = 0
for each in range(0, 100):
    Total_emp_risk += Emp_risk[each]
T_aver_risk = Total_emp_risk/100
Term_one = math.sqrt((2*math.log((2.71828*100)/d))/(100/d)) #Second term of the formula
Term_two = math.sqrt((math.log(1/0.05))/(2 * 100)) #Third term of the formula
R = T_aver_risk + Term_one + Term_two #VC Generalization bound
print("Average risk: {}, Term one: {}, Term two: {}".format(T_aver_risk, Term_one, Term_two))
print("VC-Dimension GB " + str(100) + " samples: " + str(R))