import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes


class LinearRegressionOLS:
    ## equation of w  = (X.T @ X)**-1 @ X.T @ y

    def fit(self, X, y):

        # Initializing out base feature matrix and target vector
        # Also, we add a new column with only ones to the beginning of X matrix for the intercept

        self.X_train = np.column_stack((np.ones((len(X), 1)), X))
        self.y_train = y

        # Multiply X transpose with y matrix
        # Multiply both the matrices to find the intercept and the coefficient

        # According to the equation we need to find out an inverse matrix of X
        # We can do this by creating a GRAM matrix 
        X_train_gram = self.X_train.T @ self.X_train

        # Also adding here a little regularization to reduce posibility of error in case of collinearity
        X_train_gram = X_train_gram + 0.01 * np.eye(X_train_gram.shape[0])

        # Inverse of GRAM matrix
        X_train_inv = np.linalg.inv(X_train_gram)

        # Finding weights ( including intercept )
        weights = X_train_inv @ self.X_train.T @ self.y_train
        
        self.w0 = weights[0]
        self.w = weights[1:]


    def predict(self, X):
        pred = self.w0 + X @ self.w
        return pred


class LinearRegressionGradient:
    # init of basic constants
    def __init__(self, lrate=0.01, epochs=1000):
        self.lrate = lrate
        self.epochs = epochs

    # For training I've used MSE as the Loss function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # random values for 1st iter
        self.intercept = np.random.random(1)
        self.weights = np.random.random(self.X_train.shape[1])

        # iterating via epochs and constantly upgrading our values
        for _ in range(self.epochs):
            self.y_pred = self.X_train @ self.weights + self.intercept
            self.intercept = self.intercept -  self.lrate * self._dldi()
            self.weights = self.weights - self.lrate * self._dldw()

    
    # helper method for calculation derivative w.r.t. weights
    def _dldw(self):
        res = np.sum(-self.X_train.T @ (self.y_train - self.y_pred))
        return (2/self.X_train.shape[0]) * res

    # helper method for calculation derivative w.r.t. intercept
    def _dldi(self):
        res = np.sum(-(self.y_train - self.y_pred))
        return (2/self.X_train.shape[0]) * res


    def predict(self, X):
        pred =  X @ self.weights + self.intercept
        return pred


if __name__ == '__main__':
    data = load_diabetes()
    X, y = data.data, data.target

    models = [LinearRegressionOLS, LinearRegressionGradient, LinearRegression]
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    model_scores = {}
    for model in models:
        scores = []
        for train_indx, test_indx in kf.split(X, y):
            clf = model()
            clf.fit(X[train_indx], y[train_indx])
            clf_pred = clf.predict(X[test_indx])
            scores.append(mean_squared_error(y[test_indx], clf_pred))
        model_scores[model] = np.mean(scores)

    print(model_scores)