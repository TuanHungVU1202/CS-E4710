import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""
More info about the attributes in the dataset:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
"""

X, y = load_breast_cancer(return_X_y=True)
print("data shapes:", X.shape, y.shape, np.unique(y))

# ------------------------------------------------------------------------------------
# add here transformations on X

# centering = StandardScaler(with_mean=True, with_std=False).fit(X)
# standardisation = StandardScaler(with_mean=True, with_std=True).fit(X)
# unit_range = preprocessing.MinMaxScaler()
# normalization = preprocessing.Normalizer().fit(X)

# X = centering.transform(X)
# X = standardisation.transform(X)
# X = unit_range.fit_transform(X)
# X = normalization.transform(X)
# ------------------------------------------------------------------------------------
# linear classification

# divide into training and testing
np.random.seed(42)
order = np.random.permutation(len(y))
tr = np.sort(order[:250])
tst = np.sort(order[250:])

svm = LinearSVC(fit_intercept=False, random_state=2)
svm.fit(X[tr, :], y[tr])
preds = svm.predict(X[tst, :])
print("SVM accuracy:", np.round(100 * accuracy_score(y[tst], preds), 1), "%")

# result
# No transformation: 89.7%
# Centering - with mean 90.9%
# Standardisation - StandardScaler - with std.dev - 94.7%
# Unit range - MinMaxScaler - 93.7%
# Normalization of feature vectors - 82.8%
