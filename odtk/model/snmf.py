import numpy as np
from sklearn.decomposition import NMF
from odtk.model.superclass import *
from sklearn.ensemble import GradientBoostingRegressor


class nmf(NormalModel):

    def __init__(self,
                 train,
                 test,

                 header=0,

                 time_length=30,
                 resolution=15,

                 alpha=0.01,
                 beta=0.1):

        self.train = train
        self.test = test

        self.time_length = time_length
        self.resolution = resolution

        self.alpha = alpha
        self.beta = beta

        if isinstance(header, int):
            self.feature_col = header
        elif isinstance(header, str):
            self.feature_col = train.header[header]
        else:
            raise ValueError("The type of header is not int or str")


def run(self):
    ##########################Learning#################################
    X = []
    Y = []
    n = int(self.train.data.shape[0] / self.time_length) * self.time_length

    for i in range(0, n, self.resolution):
        X.append(self.train.data[i:i + self.time_length, self.feature_col])
        Y.append(np.mean(self.train.occupancy[i:i + self.time_length, 0]))

    model = NMF(n_components=self.resolution, init='random', random_state=0, alpha=self.alpha, l1_ratio=self.beta)

    W = model.fit_transform(X)
    H = model.components_

    gblsr = GradientBoostingRegressor(loss='ls', n_estimators=100).fit(W, Y)
    ####################################################################

    #############################Prediction#############################
    X_ = []
    n = int(self.test.data.shape[0] / self.time_length) * self.time_length

    for i in range(0, n, self.resolution):
        X_.append(self.test.data[i:i + self.time_length, self.feature_col])

    model_ = NMF(n_components=self.resolution, init='custom', random_state=0, alpha=self.alpha, l1_ratio=self.beta)

    W_ = model_.fit_transform(X_, H=H)

    predict_occupancy = gblsr.predict(W_)
    ####################################################################

    return predict_occupancy
