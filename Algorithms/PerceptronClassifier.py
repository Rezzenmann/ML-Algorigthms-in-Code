import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score


class PerceptronScratch:
    def __init__(self, lrate=0.001, epochs=1000):
        self.lrate = lrate
        self.epochs = epochs
        self.weights = None
        self.intercept = None

    def _treshold_func(self, y):
        return np.where(y >= 0, 1, 0)

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((1,n))
        self.intercept = 0
        # let's transform all target values into 0 or 1
        y_ =  np.array([1 if i > 0 else 0 for i in y])

        for epoch in range(self.epochs):
            # for each iter we predict new values using linear function and then cast values to 0 or 1
            pred = self._treshold_func(np.dot(X, self.weights.T) + self.intercept)

            # weights updating
            w_delta = np.dot((y - pred).T, X)
            self.weights = self.weights + self.lrate * w_delta

    def predict(self, X):
        pred = self._treshold_func(np.dot(X, self.weights.T) + self.intercept)
        return pred


if __name__ == '__main__':
    data = load_breast_cancer()
    X, y = data.data.astype('float32'), data.target.reshape((len(data.target), 1))

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scratch_scores = []
    for train_indx, test_indx in sss.split(X, y):
        scratch_clf = PerceptronScratch(lrate=0.0001, epochs=10000)
        scratch_clf.fit(X[train_indx], y[train_indx])
        clf_pred = scratch_clf.predict(X[test_indx])
        scratch_scores.append(roc_auc_score(y[test_indx], clf_pred))

    print('Scratch', np.mean(scratch_scores), np.std(scratch_scores))
    print('Scratch', scratch_clf.weights)

    sklearn_scores = []
    for train_indx, test_indx in sss.split(X, y):
        sklearn_clf = Perceptron(n_jobs=-1, max_iter=10000)
        sklearn_clf.fit(X[train_indx], y[train_indx].ravel())
        clf_pred = sklearn_clf.predict(X[test_indx])
        sklearn_scores.append(roc_auc_score(y[test_indx].ravel(), clf_pred))

    print('Sklearn', np.mean(sklearn_scores), np.std(sklearn_scores))
    print('Sklearn', sklearn_clf.coef_)