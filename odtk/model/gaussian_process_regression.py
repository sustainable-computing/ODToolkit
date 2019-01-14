import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

from odtk.data.dataset import Dataset


def gaussian_process_regression(train,
                                test,

                                kernel=None,
                                normalize_y=True,

                                train_start=0,
                                train_end=-1,

                                test_start=0,
                                test_end=-1,

                                feature_col=0):
    if not isinstance(train, Dataset) or not isinstance(test, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

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
        kernel_gpml = k1 + k2 + k3 + k4
    else:
        kernel_gpml = kernel

    X = train.data[train_start:train_end, feature_col]
    Y = train.occupancy[train_start:train_end]

    gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                                  optimizer=None, normalize_y=normalize_y).fit(X, Y)

    predict_occupancy = gp.predict(test.data[test_start:test_end, feature_col])
    return predict_occupancy
