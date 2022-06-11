import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def train(self, X, y):
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

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=0)

    pure_knn_clf = KNN()
    pure_knn_clf.train(X_train, y_train)
    pure_knn_pred = pure_knn_clf.predict(X_test)
    print(classification_report(y_test, pure_knn_pred))

    sklearn_knn_clf = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn_clf.fit(X_train, y_train)
    sklearn_knn_pred = sklearn_knn_clf.predict(X_test)
    print(classification_report(y_test, sklearn_knn_pred))
    