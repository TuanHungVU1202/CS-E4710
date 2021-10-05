import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load the data, X is the whole matrix, while y is the response vector
X, y = load_diabetes(return_X_y=True)
# print(X.shape, y.shape)
print(X)
print(y)
# division into training and testing
np.random.seed(0)
order = np.random.permutation(len(y))
# test size = 50:50
# index from 0
# tst get all elements in array with index < 200
tst = np.sort(order[:200])
# tr get all elements in array with index >= 200
tr = np.sort(order[200:])

Xtr = X[tr, :]
Xtst = X[tst, :]
Ytr = y[tr]
Ytst = y[tst]

linReg = LinearRegression()
# training
linReg.fit(Xtr, Ytr)

y_pred = linReg.predict(Xtst)

# Ytst = y_actual = y_target
print(np.sqrt(mean_squared_error(Ytst, y_pred)))
