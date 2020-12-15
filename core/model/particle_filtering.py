#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .superclass import *


class PF(NormalModel):
    """
    Using `Particle Filtering <https://en.wikipedia.org/wiki/Particle_filter>`_
    model to predict the occupancy level

    This is a normal supervised learning model.

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: core.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: core.data.dataset.Dataset

    :parameter number_of_hidden_states: the number of maximum occupancy level
    :type number_of_hidden_states: int

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self,
                 train,
                 test):
        from numpy import amax

        self.train = train
        self.test = test
        self.number_of_hidden_states = int(amax(train.occupancy)) + 1

    def run(self):
        from . import hmm_core
        from numpy import reshape

        hmm = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)
        hmm.learn(hidden_seq=np.array(self.train.occupancy, int).flatten(),
                  emission_seq=self.train.data)

        predict_occupancy = hmm.pf_predict(emission_seq=np.array(self.test.data))

        return np.reshape(predict_occupancy, (-1, 1))


class PF_DA(DomainAdaptiveModel):
    """
    Using `Particle Filtering <https://en.wikipedia.org/wiki/Particle_filter>`_
    model to predict the occupancy level

    This is a domain-adaptive semi-supervised learning model.

    :parameter source: the source domain with full knowledge for training the model
    :type source: core.data.dataset.Dataset

    :parameter target_retrain: the labelled ground truth Dataset in the target domain for re-training the model
    :type target_retrain: ``None`` or core.data.dataset.Dataset

    :parameter target_test: the Dataset in the rest of the target domain for testing by using sensor data only
    :type target_test: core.data.dataset.Dataset

    :parameter number_of_hidden_states: the number of maximum occupancy level
    :type number_of_hidden_states: int

    :rtype: numpy.ndarray
    :return: Predicted occupancy level corresponding to the test Dataset
    """
    def __init__(self,
                 source,
                 target_retrain,
                 target_test):
        from numpy import amax

        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.number_of_hidden_states = int(amax(source.occupancy)) + 1

    def run(self):
        from . import hmm_core
        from numpy import reshape

        hmm_source = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)

        hmm_source.learn(hidden_seq=np.array(self.source.occupancy, int).flatten(),
                         emission_seq=self.source.data)

        hmm_target = hmm_core.HMM_Core(number_of_hidden_states=self.number_of_hidden_states)

        hmm_target.prior(hmm_source)

        hmm_target.learn(hidden_seq=np.array(self.target_retrain.occupancy, int).flatten(),
                         emission_seq=self.target_retrain.data)

        predict_occupancy = hmm_target.pf_predict(emission_seq=np.array(self.target_test.data))

        return reshape(predict_occupancy, (-1, 1))
