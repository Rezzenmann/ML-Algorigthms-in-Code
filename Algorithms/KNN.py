import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import  f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from collections import Counter


class KNN:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = []

        for x_train in self.X_train:
            distances.append(self._euclidean_distance(x, x_train))
            
        tups = zip(distances, self.y_train)
        tups = sorted(tups)[:self.k]
        classes = [j for (i,j) in tups]

        return Counter(classes).most_common(1)[0][0]

    


if __name__ == '__main__':
    data = load_iris()
    X, y = data.data, data.target
    
    models = [KNN, KNeighborsClassifier]
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    model_scores = {}
    for model in models:
        scores = []
        for train_indx, test_indx in sss.split(X, y):
            clf = model(n_neighbors=5)
            clf.fit(X[train_indx], y[train_indx])
            clf_pred = clf.predict(X[test_indx])
            scores.append(f1_score(y[test_indx], clf_pred, average='weighted'))
        model_scores[model] = np.mean(scores)

    print(model_scores)
    