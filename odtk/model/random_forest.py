from sklearn.ensemble import RandomForestClassifier
# Have to include this
from odtk.model.superclass import *


class RandomForest(NormalModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, train, test):
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.estimator = 200

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = RandomForestClassifier(n_estimators=self.estimator)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        return predict_occupancy


class RandomForestDA(DomainAdaptiveModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, source, target_retrain, target_test):
        # all changeable parameters now store as an editable instance
        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.estimator = 200

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = RandomForestClassifier(n_estimators=self.estimator)

        classifier.fit(self.source.data, self.source.occupancy)

        if self.target_retrain is not None:
            classifier.fit(self.target_retrain.data, self.target_retrain.occupancy)

        predict_occupancy = classifier.predict(self.target_test.data)

        return predict_occupancy
