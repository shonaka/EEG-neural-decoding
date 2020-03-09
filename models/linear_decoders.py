import sys
from sklearn import linear_model
from sklearn.svm import SVR
# Just for debugging, later delete
import pdb


# ========== Linear based Decoders ==========
class LinearRegression(object):
    """Defining a class for Linear Regression

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def __init__(self, args):
        """
        Defining a constructor
        """
        self.args = args
        return

    def train(self, trainX, trainY):
        """
        Train the Wiener Filter (Linear Regression).
        """
        self.model = linear_model.LinearRegression()
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        """
        Predict and return the output
        """
        predicted = self.model.predict(testX)
        return predicted


class RidgeRegression(object):
    """A class for Ridge Regression
    """

    def __init__(self, args):
        self.args = args
        return

    def train(self, trainX, trainY):
        self.model = linear_model.Ridge(alpha=self.args.rr_alpha)
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        predicted = self.model.predict(testX)
        return predicted


class Lasso(object):
    def __init__(self, args):
        self.args = args
        return

    def train(self, trainX, trainY):
        self.model = linear_model.Lasso()
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        predicted = self.model.predict(testX)
        return predicted


class ElasticNet(object):
    """Linear regression combined with L1 and L2 reguralization.

    Didn't work. Mostly because EEG are multilinearly correlated
    """

    def __init__(self):
        return

    def train(self, trainX, trainY):
        self.model = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.1)
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        predicted = self.model.predict(testX)
        return predicted
