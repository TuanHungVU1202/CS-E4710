from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

names = [
    "Neural Net",
    "AdaBoost",
    "Gradient Boosting"
]

classifiers = [
    MLPClassifier(alpha=1, max_iter=100, random_state=1),
    AdaBoostClassifier(random_state=1),
    GradientBoostingClassifier(n_estimators=200, learning_rate=1, max_depth=1, random_state=1)
]

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1
)

score_dict = {}
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="binary")
    score_dict[name] = score
    # print(name + "= " + str(score) + "\n")

print(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
