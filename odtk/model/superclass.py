from odtk.data.dataset import Dataset
from multiprocessing.pool import ThreadPool
from collections import Iterable


class NormalModel:
    def __init__(self,
                 train,
                 test,
                 thread_num=4):

        if not isinstance(train, Dataset) or not isinstance(test, Dataset):
            raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

        self.train = train
        self.test = test
        self.models = {}
        self.thread_num = thread_num

    def get_all_model(self):
        for model in NormalModel.__subclasses__():
            self.models[model.__name__] = model(self.train.copy(), self.test.copy())

    def add_model(self, list_of_model):
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
    def __init__(self,
                 source,
                 target_retrain,
                 target_test,
                 thread_num=4):
        if not isinstance(source, Dataset) or not isinstance(target_test, Dataset) or \
                target_retrain is not None and not isinstance(target_retrain, Dataset):
            raise ValueError("Given train and test is not class odtk.data.dataset.Dataset")

        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.models = {}
        self.thread_num = thread_num

    def get_all_model(self):
        for model in DomainAdaptiveModel.__subclasses__():
            self.models[model.__name__] = model(self.source.copy(), self.target_retrain.copy(), self.target_test.copy())

    def add_model(self, list_of_model):
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
        pool = ThreadPool(processes=self.thread_num)
        result = dict()
        for model in self.models.keys():
            result[model] = pool.apply_async(self.models[model].run)
        pool.close()
        pool.join()
        for key in result.keys():
            result[key] = result[key].get()

        return result
