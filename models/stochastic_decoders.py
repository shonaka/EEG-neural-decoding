import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pdb


class GaussianProcess():
    def __init__(self, args):
        self.args = args
        return

    def train(self, trainX, trainY):
        # Instantiate a Gaussian Process Model
        kernel = RBF(1, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9)
        self.model.fit(trainX, trainY)

    def predict(self, testX):
        predicted = self.model.predict(testX)
        return predicted


class BayesRidge():
    def __init__(self, args):
        self.args = args
        return

    def train(self, trainX, trainY):
        models = []
        for i in range(self.args.num_chan_kin):
            model = BayesianRidge()
            model.fit(trainX, trainY[:, i])
            models.append(model)
        self.models = models

    def predict(self, testX):
        predicted = []
        for i in range(self.args.num_chan_kin):
            temp = self.models[i].predict(testX)
            predicted.append(temp)

        predicted = np.array(predicted).T
        return predicted
