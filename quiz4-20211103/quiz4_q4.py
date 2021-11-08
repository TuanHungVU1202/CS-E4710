import numpy as np
from scipy.stats import pearsonr
from sklearn.datasets import load_breast_cancer
from Softmargin_SVM_gd import Softmargin_SVM_gd
from Dual_softmargin_svm import Dual_softmargin_svm
from Dual_softmargin_svm2 import Dual_softmargin_svm2
# load the data
X, y = load_breast_cancer(return_X_y=True)  # X input, y output
print(X.shape, y.shape)
# to convert the {0,1} output into {-1,+1}
y = 2 * y - 1

mdata, ndim = X.shape  # size of the data

iscale = 1  # =0 no scaling, =1 scaling the by the maximum absolute value
if iscale == 1:
    X /= np.outer(np.ones(mdata), np.max(np.abs(X), 0))

niter = 10  # number of iteration

# initialize eta, lambda for the primal algorithm
eta = 0.1  # step size
xlambda = 0.01  # balancing constant between loss and regularization
# set the penalty constant for the dual algorithm
C = 1000

gd_softmargin_svm = Softmargin_SVM_gd()
w0 = np.zeros(X.shape[1])
# b0 = np.random.randn()
w_gd = gd_softmargin_svm.softmarginSVM_gd(X, y, w0, eta, niter, xlambda)
print(w_gd)

dual_softmargin_svm = Dual_softmargin_svm()

# test 2nd way
dual_softmargin_svm2 = Dual_softmargin_svm2()
w_dual, simu_primal_hist_sdca_cyclic = dual_softmargin_svm2.SDCA_SVM(X, y, X.shape[0],
                                                                T_0=50 * X.shape[0] // 2, lambd=xlambda / X.shape[0], nb_epochs=10,
                                                                iteration="cyclic", average=False)
print(w_dual)
# calculate Pearson's correlation
corr, _ = pearsonr(w_gd, w_dual)
result = np.corrcoef(w_gd, w_dual)
print(corr)


