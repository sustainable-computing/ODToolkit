import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingRegressor

from odtk.data.dataset import Dataset


def nmf(train,
        test,
        
        time_length=30,
        resolution=15,
        
        alpha=0.01,
        beta=0.1,
        
        train_start=0,
        train_end=-1,
        
        test_start=0,
        test_end=-1,
        
        feature_col=0):

    if not isinstance(train, Dataset) or not isinstance(test, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")
        
    ##########################Learning#################################
    X = []
    Y = []
    for i in range(train_start, train_end, resolution):
        X.append(train.data[i:i+time_length, feature_col])
        Y.append(np.mean(train.data[i:i+time_length, feature_col]))
    
    model = NMF(n_components=resolution , init='random', random_state=0, alpha=alpha, l1_ratio=beta)
    
    W = model.fit_transform(X)
    H = model.components_
    
    gblsr = GradientBoostingRegressor(loss='ls', n_estimators=100).fit(W, Y)
    ####################################################################
    
    #############################Prediction#############################
    X_ = []
    for i in range(train_start, train_end, resolution):
        X_.append(train.data[i:i+time_length, feature_col])
        
    model_ = NMF(n_components=resolution , init='custom', random_state=0, alpha=alpha, l1_ratio=beta)
    
    W_ = model_.fit_transform(X_, H=H)
    
    predict_occupancy = gblsr.predict(W_)
    ####################################################################

    return predict_occupancy

