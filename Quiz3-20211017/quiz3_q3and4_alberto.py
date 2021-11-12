## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler

from Perceptron3 import Perceptron3
from Perceptron4 import Perceptron4

## ###################################################
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
iscale = 0 # 0 for no scaling (Question 3), 1 for scaling by the maximum absolute value (Question 4)

if iscale == 1:
    X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

ifold = 0
# Following the example of the KFold function in sklearn
for index_train, index_test in cselection.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]
    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]
    print("Training size: ", mtrain)
    print("Test size: ", mtest)

    # initialize the weight vector
    w = np.zeros(ndim)

    # setting the number of iterations for the Questions
    if iscale == 0:
        niter = 20 # Question 3
    else:
        niter = 8 # Question 4
    # Run the perceptron
    for iter in range(niter):
        for i in range(mtrain):
            functional_margin = ytrain[i] * np.dot(w,Xtrain[i])
            if functional_margin <= 0:
                w += ytrain[i] * Xtrain[i]
            if functional_margin > maxmargin_train[ifold]:
                maxmargin_train[ifold] = functional_margin
    print("Fold: ", ifold, "Maximum margin achieved in the training in the fold: ", '%8.4f'%maxmargin_train[ifold])

    # Test
    yprediction = np.dot(Xtest, w)
    yprediction = np.sign(yprediction)

    # Based on Lecture 1, slide "Confusion Matrix"
    t_p = np.sum((ytest > 0)*(yprediction > 0))
    t_n = np.sum((ytest <= 0) * (yprediction <= 0))
    f_p = np.sum((ytest <= 0) * (yprediction > 0))
    f_n = np.sum((ytest > 0) * (yprediction <= 0))

    precision = t_p/(t_p + f_p)
    recall = t_p/(t_p + f_n)

    f1[ifold] = 2 * precision * recall/(precision + recall)
    print("t_p, f_p, f_n, t_n:",t_p, f_p, f_n, t_n)
    print("Fold, F1, precision, recall:", ifold, '%6.2f'%f1[ifold], '%6.2f'%precision, '%6.2f'%recall)

    ifold += 1

# Print the results
# F1 Score
print("The average F1:", np.mean(f1))

# Maximum margin
print("The average maximum margin achieved in the training:", np.mean(maxmargin_train))