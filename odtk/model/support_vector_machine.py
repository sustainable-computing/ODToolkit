from sklearn import svm
# Have to include this
from odtk.model.superclass import *


class SVM(NormalModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, train, test):
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.gamma = 'auto',
        self.kernel = 'linear',
        self.penalty_error = 1

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = svm.SVC(kernel=self.kernel, C=self.penalty_error, gamma=self.gamma)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        return predict_occupancy


class SVMDA(DomainAdaptiveModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, source, target_retrain, target_test):
        # all changeable parameters now store as an editable instance
        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.gamma = 'auto',
        self.kernel = 'linear',
        self.penalty_error = 1

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = svm.SVC(kernel=self.kernel, C=self.penalty_error, gamma=self.gamma)

        classifier.fit(self.source.data, self.source.occupancy)

        if self.target_retrain is not None:
            classifier.fit(self.target_retrain.data, self.target_retrain.occupancy)

        predict_occupancy = classifier.predict(self.target_test.data)

        return predict_occupancy
