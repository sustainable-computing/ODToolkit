from sklearn.ensemble import RandomForestClassifier
from odtk.data.dataset import Dataset
from odtk.model.superclass import *


def random_forest(train,
                  test,
                  retrain=None,
                  estimators=200):

    if not isinstance(train, Dataset) or not isinstance(test, Dataset) or \
            retrain is not None and not isinstance(retrain, Dataset):
        raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

    classifier = RandomForestClassifier(n_estimators=estimators)

    classifier.fit(train.data, train.occupancy)

    if retrain is not None:
        classifier.fit(retrain.data, retrain.occupancy)

    predict_occupancy = classifier.predict(test.data)

    return predict_occupancy
