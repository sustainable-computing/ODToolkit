# Have to include this
from odtk.model.superclass import *


class NNv2(NormalModel):
    # For NormalModel, require two parameters: train and test
    # For DomainAdaptiveModel, require three parameters: source, target_retrain and target_test
    def __init__(self, train, test):
        from numpy import reshape
        # all changeable parameters now store as an editable instance
        self.train = train
        self.test = test
        self.solver = 'adam'
        self.alpha = 0.0001
        self.batch_size = 'auto'
        self.activation = 'logistic'
        if len(self.train.occupancy.shape) == 2 and self.train.occupancy.shape[1] == 1:
            self.train.change_occupancy(reshape(self.train.occupancy, (self.train.occupancy.shape[0],)))

    # the model must have a method called run, and return the predicted result
    def run(self):
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(solver=self.solver,
                                   alpha=self.alpha,
                                   hidden_layer_sizes=(75,),
                                   batch_size=self.batch_size,
                                   activation=self.activation)

        classifier.fit(self.train.data, self.train.occupancy)

        predict_occupancy = classifier.predict(self.test.data)

        if len(predict_occupancy.shape) == 1:
            predict_occupancy.shape += (1,)

        print("NN done.")
        return predict_occupancy
