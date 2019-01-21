from sklearn.neural_network import MLPClassifier
from numpy import reshape
# Have to include this
from odtk.model.superclass import *


class NNv2(NormalModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, train, test):
        # all changeable parameters now store as an editable instance
        self.train = train
        self.train.remove_feature(self.train.header_info[self.train.time_column])
        self.test = test
        self.test.remove_feature(self.test.header_info[self.train.time_column])
        self.solver = 'adam'
        self.alpha = 0.0001
        self.batch_size = 'auto'
        self.activation = 'logistic'
        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = MLPClassifier(solver=self.solver,
                                   alpha=self.alpha,
                                   hidden_layer_sizes=(75,),
                                   batch_size=self.batch_size,
                                   activation=self.activation)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        if len(predict_occupancy.shape) == 1:
            predict_occupancy.shape += (1,)

        return predict_occupancy


class RandomForestDA(DomainAdaptiveModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, source, target_retrain, target_test):
        # all changeable parameters now store as an editable instance
        self.source = source
        self.source.remove_feature(self.source.header_info[self.source.time_column])
        self.target_retrain = target_retrain
        if self.target_retrain is not None:
            self.target_retrain.remove_feature(self.target_retrain.header_info[self.target_retrain.time_column])
        self.target_test = target_test
        self.target_test.remove_feature(self.target_test.header_info[self.target_test.time_column])
        self.estimator = 200

        if len(self.source.occupancy.shape) == 2 and self.source.occupancy.shape[1] == 1:
            self.source.change_occupancy(reshape(self.source.occupancy, (self.source.occupancy.shape[0],)))

        if len(self.target_retrain.occupancy.shape) == 2 and self.target_retrain.occupancy.shape[1] == 1:
            self.target_retrain.change_occupancy(
                reshape(self.target_retrain.occupancy, (self.target_retrain.occupancy.shape[0],)))

    # the model must have a method called run, and return the predicted result
    def run(self):
        classifier = RandomForestClassifier(n_estimators=self.estimator)

        classifier.fit(self.source.data, self.source.occupancy)

        if self.target_retrain is not None:
            classifier.fit(self.target_retrain.data, self.target_retrain.occupancy)

        predict_occupancy = classifier.predict(self.target_test.data)

        if len(predict_occupancy.shape) == 1:
            predict_occupancy.shape += (1,)

        return predict_occupancy
