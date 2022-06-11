import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes


class LinearRegressionOLS:
    ## equation of w  = (X.T @ X)**-1 @ X.T @ y

    def train(self, X, y):

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
    def __init__(self, lrate=0.01, epochs=100):
        self.lrate = lrate
        self.epochs = epochs

    # For training I've used MSE as the Loss function
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # random values for 1st iter
        self.intercept = np.random.random(1)
        self.weights = np.random.random(self.X_train.shape[1])

        # iterating via epochs and constantly upgrading our values
        for _ in range(self.epochs):
            self.y_pred =  X @ self.weights + self.intercept
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

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=0)

    # Scratch version of Linear Regression using OLS

    lreg_scratch = LinearRegressionOLS()
    lreg_scratch.train(X_train, y_train)
    y_pred_scratch = lreg_scratch.predict(X_test)
    print('Scratch Version', round(mean_absolute_error(y_test, y_pred_scratch),3), round(mean_squared_error(y_test, y_pred_scratch),3))

    # Scratch version of Linear Regression using Gradient Descent
    # It needs to be tuned to get better results
    lreg_grad = LinearRegressionGradient()
    lreg_grad.train(X_train, y_train)
    y_pred_grad = lreg_grad.predict(X_test)
    print('Gradient Version', round(mean_absolute_error(y_test, y_pred_grad),3), round(mean_squared_error(y_test, y_pred_grad),3))


    # Sklearn's Linear Regression 
    lreg_sklearn = LinearRegression()
    lreg_sklearn.fit(X_train, y_train)
    y_pred_sklearn = lreg_sklearn.predict(X_test)
    print('Sklearn Version', round(mean_absolute_error(y_test, y_pred_sklearn),3), round(mean_squared_error(y_test, y_pred_sklearn),3))