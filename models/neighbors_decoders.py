import sys
from sklearn.neighbors import KNeighborsRegressor
# Just for debugging, later delete
import pdb


# ========== Linear based Decoders ==========
class kNNregression(object):
    """Defining a class for kNN Regression

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
        self.model = KNeighborsRegressor(n_neighbors=5)
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        """
        Predict and return the output
        """
        predicted = self.model.predict(testX)
        return predicted
