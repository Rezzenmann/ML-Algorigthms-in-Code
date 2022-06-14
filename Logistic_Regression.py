import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


class LogisticRegressionScratch:
    
    def __init__(self, lrate=0.05, max_iter=100):
        self.lrate = lrate
        self.epochs = max_iter


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.weights = np.random.rand(self.X_train.shape[1], 1)
        self.intercept = np.random.rand(1)
        self.length = self.X_train.shape[0]

        loss = {}

        for _ in range(self.epochs):    
            # get prediction
            self.pred = self._sigmod( np.dot(self.X_train, self.weights)  + self.intercept)
            # log loss cost function

            cost = (-1/self.length)*(np.sum(np.dot(self.y_train,np.where(self.pred > 0, np.log(self.pred), 0).T)+\
                 np.dot((1-self.y_train),np.where(1-self.pred > 0, np.log(1-self.pred), 0).T)))

            loss[_] = [_, cost]
            dldw = ( 1/self.length) * (np.dot(self.X_train.T, (self.pred - self.y_train)))
            dldi =  ( 1/self.length) * np.sum((self.pred - self.y_train))
            self.weights = self.weights - self.lrate * dldw
            self.intercept = self.intercept - self.lrate * dldi
        
        
    def predict(self, X):
        return X @ self.weights  + self.intercept


    def _sigmod(self, pred):
        return 1 / (1 + np.exp(-pred))


if __name__ == '__main__':
    data = load_breast_cancer()
    X, y = data.data, data.target
    X = X.astype('float32')
    y = np.reshape(y,(len(y), 1))

    models = [LogisticRegression, LogisticRegressionScratch]
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    model_scores = {}
    for model in models:
        scores = []
        for train_indx, test_indx in sss.split(X, y):
            clf = model(max_iter=5000)
            clf.fit(X[train_indx], y[train_indx])
            clf_pred = clf.predict(X[test_indx])
            scores.append(roc_auc_score(y[test_indx], clf_pred))
        model_scores[model] = np.mean(scores)

    print(model_scores)


