#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..data.dataset import Dataset


class NormalModel:
    """
    Use all normal supervised learning model to train and test the given Datasets

    :parameter train: the labelled ground truth Dataset for training the model
    :type train: odtoolkit.data.dataset.Dataset

    :parameter test: the Dataset for testing by using sensor data only
    :type test: odtoolkit.data.dataset.Dataset

    :parameter thread_num: the maximum number of threads can use to speed up
    :type thread_num: int

    :rtype: odtoolkit.model.superclass.NormalModel
    """
    def __init__(self,
                 train,
                 test,
                 thread_num=4):

        if not isinstance(train, Dataset) or not isinstance(test, Dataset):
            raise ValueError("Given train and test is not class odtoolkit.data.dataset.Dataset")

        self.train = train
        self.test = test
        self.models = {}
        self.thread_num = thread_num

    def get_all_model(self):
        """
        Get all subclasses

        :parameter: None

        :return: None
        """
        for model in NormalModel.__subclasses__():
            self.models[model.__name__] = model(self.train.copy(), self.test.copy())

    def add_model(self, list_of_model):
        """
        Add one or multiple models into the modelling queue

        :parameter list_of_model: one or multiple models that additionally add to the modelling queue
        :type list_of_model: str or list(str)

        :return: None
        """
        from collections import Iterable

        if not isinstance(list_of_model, Iterable) or isinstance(list_of_model, str):
            list_of_model = [list_of_model]
        else:
            list_of_model = list_of_model[:]

        for model in NormalModel.__subclasses__():
            if model.__name__ in list_of_model:
                self.models[model.__name__] = model(self.train.copy(), self.test.copy())
                list_of_model.remove(model.__name__)

        if len(list_of_model):
            raise NameError("Model {} is not defined in model library".format(list_of_model))

    def remove_model(self, list_of_model):
        """
        Remove one or multiple mdoels from the modelling queue

        :parameter list_of_model: one or multiple models that want to remove from the modelling queue
        :type list_of_model: str or list(str)

        :return: None
        """
        from collections import Iterable

        if not isinstance(list_of_model, Iterable) or isinstance(list_of_model, str):
            list_of_model = [list_of_model]
        else:
            list_of_model = list_of_model[:]

        i = 0
        while i < len(list_of_model):
            if self.models.pop(list_of_model[i], False):
                list_of_model.remove(list_of_model[i])
            else:
                i += 1

        if len(list_of_model):
            raise NameError("Model {} is not selected".format(list_of_model))

    def run_all_model(self):
        """
        Run all models that currently in the queue

        :parameter: None

        :rtype: dict(str, numpy.ndarray)
        :return: the predicted occupancy level data
        """
        from multiprocessing.pool import ThreadPool

        pool = ThreadPool(processes=self.thread_num)
        result = dict()
        for model in self.models.keys():
            result[model] = pool.apply_async(self.models[model].run)
        pool.close()
        pool.join()
        for key in result.keys():
            result[key] = result[key].get()

        return result


class DomainAdaptiveModel:
    """
    Use all domain-adaptive semi-supervised learning model to train and test the given Datasets

    :parameter source: the source domain with full knowledge for training the model
    :type source: odtoolkit.data.dataset.Dataset

    :parameter target_retrain: the labelled ground truth Dataset in the target domain for re-training the model
    :type target_retrain: ``None`` or odtoolkit.data.dataset.Dataset

    :parameter target_test: the Dataset in the rest of the target domain for testing by using sensor data only
    :type target_test: odtoolkit.data.dataset.Dataset

    :parameter thread_num: the maximum number of threads can use to speed up
    :type thread_num: int

    :rtype: odtoolkit.evaluation.superclass.DomainAdaptiveModel
    """
    def __init__(self,
                 source,
                 target_retrain,
                 target_test,
                 thread_num=4):
        if not isinstance(source, Dataset) or not isinstance(target_test, Dataset) or \
                target_retrain is not None and not isinstance(target_retrain, Dataset):
            raise ValueError("Given train and test is not class odtoolkit.data.dataset.Dataset")

        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.models = {}
        self.thread_num = thread_num

    def get_all_model(self):
        """
        Get all subclasses

        :parameter: None

        :return: None
        """
        for model in DomainAdaptiveModel.__subclasses__():
            self.models[model.__name__] = model(self.source.copy(), self.target_retrain.copy(), self.target_test.copy())

    def add_model(self, list_of_model):
        """
        Add one or multiple models into the modelling queue

        :parameter list_of_model: one or multiple models that additionally add to the modelling queue
        :type list_of_model: str or list(str)

        :return: None
        """
        from collections import Iterable

        if not isinstance(list_of_model, Iterable) or isinstance(list_of_model, str):
            list_of_model = [list_of_model]
        else:
            list_of_model = list_of_model[:]

        for model in DomainAdaptiveModel.__subclasses__():
            if model.__name__ in list_of_model:
                self.models[model.__name__] = model(self.source.copy(), self.target_retrain.copy(),
                                                    self.target_test.copy())
                list_of_model.remove(model.__name__)

        if len(list_of_model):
            raise NameError("Model {} is not defined in model library".format(list_of_model))

    def remove_model(self, list_of_model):
        """
        Remove one or multiple mdoels from the modelling queue

        :parameter list_of_model: one or multiple models that want to remove from the modelling queue
        :type list_of_model: str or list(str)

        :return: None
        """
        from collections import Iterable

        if not isinstance(list_of_model, Iterable) or isinstance(list_of_model, str):
            list_of_model = [list_of_model]
        else:
            list_of_model = list_of_model[:]

        i = 0
        while i < len(list_of_model):
            if self.models.pop(list_of_model[i], False):
                list_of_model.remove(list_of_model[i])
            else:
                i += 1

        if len(list_of_model):
            raise NameError("Model {} is not selected".format(list_of_model))

    def run_all_model(self):
        """
        Run all models that currently in the queue

        :parameter: None

        :rtype: dict(str, numpy.ndarray)
        :return: the predicted occupancy level data
        """
        from multiprocessing.pool import ThreadPool

        pool = ThreadPool(processes=self.thread_num)
        result = dict()
        for model in self.models.keys():
            result[model] = pool.apply_async(self.models[model].run)
        pool.close()
        pool.join()
        for key in result.keys():
            result[key] = result[key].get()

        return result
