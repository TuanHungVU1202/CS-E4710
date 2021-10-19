## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler

from Perceptron import Perceptron
from Perceptron2 import Perceptron2
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
# normalized_arr = None
# for element in X:
#     # print(X)
#     abs_max = np.amax(np.abs(element))
#     # print(abs_max)
#     normalized_arr = X * (1.0/abs_max)
#     print(normalized_arr)

transformer = MaxAbsScaler().fit(X)
normalized_arr = transformer.transform(X)
print(normalized_arr)

perceptron = Perceptron()
perceptron2 = Perceptron2()
perceptron3 = Perceptron3()
perceptron4 = Perceptron4()
for train_index, test_index in cselection.split(X):
    # perceptron.fit(X[train_index], y[train_index], 20, 0.01)
    # perceptron2.fit(X[train_index], y[train_index])
    perceptron3.fit(X[train_index], y[train_index])
    # perceptron4.fit(X[train_index], y[train_index])
    # Ypred = perceptron.predict(X[test_index])
    # Ypred = perceptron2.predict(X[test_index])
    Ypred = perceptron3.predict(X[test_index])
    # Ypred = perceptron4.predict(X[test_index])
    f1 = f1_score(y[test_index], Ypred, average='binary')

#for normalized matrix
for train_index, test_index in cselection.split(normalized_arr):
    # perceptron.fit(normalized_arr[train_index], y[train_index], 20, 0.01)
    # perceptron2.fit(normalized_arr[train_index], y[train_index])
    perceptron3.fit(normalized_arr[train_index], y[train_index])
    # perceptron4.fit(normalized_arr[train_index], y[train_index])
    # Ypred = perceptron.predict(normalized_arr[test_index])
    # Ypred = perceptron2.predict(normalized_arr[test_index])
    Ypred = perceptron3.predict(normalized_arr[test_index])
    # Ypred = perceptron4.predict(normalized_arr[test_index])
    f1_2 = f1_score(y[test_index], Ypred, average='binary')


print('The average F1:', np.mean(f1))
print('The average F1_2:', np.mean(f1_2))
print('The average maximum margin achieved in the training:', np.mean(maxmargin_train))



