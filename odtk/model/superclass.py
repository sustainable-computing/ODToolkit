from odtk.data.dataset import Dataset
from multiprocessing.pool import ThreadPool


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
        self.get_all_model()
        self.thread_num = thread_num

    def get_all_model(self):
        for model in NormalModel.__subclasses__():
            self.models[model.__name__] = model(self.train, self.test)

    def run_all_model(self):
        pool = ThreadPool(processes=self.thread_num)
        result = dict()
        for model in self.models.keys():
            result[model] = pool.apply_async(self.models[model].run)
        pool.close()
        pool.join()
        for key in result.keys():
            result[key] = result[key].get()


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
        self.get_all_model()
        self.thread_num = thread_num

    def get_all_model(self):
        for model in DomainAdaptiveModel.__subclasses__():
            self.models[model.__name__] = model(self.source, self.target_retrain, self.target_test)

    def run_all_model(self):
        pool = ThreadPool(processes=self.thread_num)
        result = dict()
        for model in self.models.keys():
            result[model] = pool.apply_async(self.models[model].run)
        pool.close()
        pool.join()
        for key in result.keys():
            result[key] = result[key].get()
