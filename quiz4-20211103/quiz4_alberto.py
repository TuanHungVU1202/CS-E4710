# thanks to https://gist.github.com/Canu2ESP/957a40bb5f452e6ed49941c1aa4c944f
## ####################################################
import numpy as np
from sklearn.datasets import load_breast_cancer
import scipy.stats

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
print(X.shape, y.shape)
## to convert the {0,1} output into {-1,+1}
y = 2 * y -1

mdata,ndim=X.shape                                   ## size of the data

iscale = 1   ## =0 no scaling, =1 scaling the by the maximum absolute value
if iscale == 1:
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

niter = 10 ## number of iteration

## initialize eta, lambda for the primal algorithm
eta=0.1              ##  step size
xlambda=0.01          ## balancing constant between loss and regularization
## set the penalty constant for the dual algorithm
C = 1000

# TODO: Stochastic gradient descent algorithm for soft-margin SVM
# Initializing the weight vector
w = np.zeros(ndim)

# Implementing the algo
for iterations in range(niter):
    for i in range(mdata):
        if y[i] * np.dot(w, X[i]) < 1:
            w = w - eta * (xlambda * w + (-y[i] * X[i]))
        else:
            w = w - eta * xlambda * w
w_SGD_SVM = w
print(w_SGD_SVM)

# TODO: Dual Soft-Margin SVM
# Initializing vectors
w = np.zeros(ndim)
alpha = np.zeros(mdata)
sum_alpha_j = 0

# Implementing the SDCA algo for SVM
for iterations in range(niter):
    for i in range(mdata):
        for j in range(mdata):
            if i == j:
                pass
            else:
                sum_alpha_j += alpha[j] * y[j] * np.dot(X[i], X[j])
        alpha[i] = (1 - y[i] * sum_alpha_j)/np.dot(X[i], X[i])
        alpha[i] = min(C/mdata, max(0, alpha[i]))
        sum_alpha_j = 0

for i in range(mdata):
    w += alpha[i] * y[i] * X[i]
w_DUAL_SVM = w
print(w_DUAL_SVM)

result = np.corrcoef(w_DUAL_SVM, w_SGD_SVM)
print(result)
pearson_correlation = scipy.stats.pearsonr(w_DUAL_SVM, w_SGD_SVM)
print(pearson_correlation)