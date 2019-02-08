from odtk.model.superclass import *


class GPC(NormalModel):

    def __init__(self,
                 train,
                 test):

        self.train = train
        self.test = test

    def run(self):
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels \
            import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

        X = np.array(self.train.data)
        Y = np.array(self.train.occupancy).flatten()
        kernel = RBF() + RBF() * ExpSineSquared() + RationalQuadratic() + WhiteKernel()
        gp = GaussianProcessClassifier(kernel=kernel,
                                       optimizer='fmin_l_bfgs_b').fit(X, Y)

        predict_occupancy = gp.predict(np.array(self.test.data))
        return np.reshape(predict_occupancy, (-1, 1))
