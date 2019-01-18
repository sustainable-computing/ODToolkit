import numpy as np
from odtk.model.superclass import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared



class GPR(NormalModel):

    def __init__(self,
                 train,
                 test,

                 kernel=None,
                 auto_kernel=True,
                 normalize_y=True,

                 header=0):
                 
        self.train = train
        self.test = test
        
        self.normalize_y = normalize_y
        
        self.auto_kernel = auto_kernel
        
        if isinstance(header, int):
            self.feature_col = header
        elif isinstance(header, str):
            self.feature_col = train.header[header]
        else:
            raise ValueError("The type of header is not int or str")

        if kernel is None:
            # Kernel with parameters given in GPML book
            k1 = 66.0 ** 2 * RBF(length_scale=67.0)  # long term smooth rising trend
            k2 = 2.4 ** 2 * RBF(length_scale=90.0) \
                 * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
            # medium term irregularity
            k3 = 0.66 ** 2 \
                 * RationalQuadratic(length_scale=1.2, alpha=0.78)
            k4 = 0.18 ** 2 * RBF(length_scale=0.134) \
                 + WhiteKernel(noise_level=0.19 ** 2)  # noise terms
            self.kernel_gpml = k1 + k2 + k3 + k4
        else:
            self.kernel_gpml = kernel

    def run(self):
    
        X = self.train.data[:, self.feature_col]
        Y = self.train.occupancy[:, 0]

        optimizer = None
        if self.auto_kernel == True:
            optimizer = 'fmin_l_bfgs_b'

        gp = GaussianProcessRegressor(kernel=self.kernel_gpml, alpha=0,
                                      optimizer=optimizer, normalize_y=self.normalize_y).fit(X, Y)

        predict_occupancy = gp.predict(self.test.data[:, self.feature_col])
        return predict_occupancy
