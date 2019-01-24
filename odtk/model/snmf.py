import numpy as np
from sklearn.decomposition import non_negative_factorization
from odtk.model.superclass import *
from sklearn.ensemble import GradientBoostingRegressor


class nmf(NormalModel):

    def __init__(self,
                 train,
                 test):

        self.train = train
        self.test = test
        
        self.alpha=0.9
        self.beta=0.8

        self.time_length = 60
        self.resolution = 3


    def run(self):
        ##########################Learning#################################
        X = []
        Y = []
        data = np.array(self.train.data).flatten()
        n = int(data.shape[0] / self.time_length) * self.time_length
        
        diff = self.time_length - self.resolution
        n = n-diff
        
        for i in range(0, n, self.resolution):
            X.append(data[i:i + self.time_length])
            Y.append(np.mean(self.train.occupancy[i:i + self.resolution]))

        W, H, _ = non_negative_factorization(X=X,
                                           n_components=self.resolution, 
                                           regularization='transformation',
                                           alpha=2*self.alpha+self.beta, 
                                           l1_ratio=self.beta / (2*self.alpha+self.beta))


        gblsr = GradientBoostingRegressor(loss='ls', n_estimators=500).fit(W, Y)
        ####################################################################

        #############################Prediction#############################
        X_ = []
        data = np.array(self.test.data).flatten()
        n = int(data.shape[0] / self.time_length) * self.time_length
        
        n = n-diff
        
        for i in range(0, n, self.resolution):
            X_.append(data[i:i + self.time_length])

        W_, _, _ = non_negative_factorization(X=X_,
                                              H=H,
                                              n_components=self.resolution, 
                                              regularization='transformation',
                                              alpha=2*self.alpha+self.beta, 
                                              l1_ratio=self.beta / (2*self.alpha+self.beta))

        Y_ = np.repeat(gblsr.predict(W_), self.resolution)
        Y_[Y_ < 0] = 0
        predict_occupancy = np.array(np.repeat(Y_, [1]*(len(Y_)-1)+[data.shape[0]-n+1]), int)
        ####################################################################

        return np.reshape(predict_occupancy, (-1,1))
