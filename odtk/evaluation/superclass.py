from numpy import ndarray, round
from collections import Iterable


class BinaryEvaluation:
    def __init__(self,
                 predict,
                 truth):

        if not isinstance(predict, ndarray) or not isinstance(truth, ndarray):
            return

        if truth.shape != predict.shape:
            raise ValueError("Two array have different shape")

        if len(truth.shape) != 1 and truth.shape[1] != 1:
            truth = truth.argmax(axis=1)
            predict = predict.argmax(axis=1)

        predict = round(predict)

        self.predict = predict
        if len(self.predict.shape) == 1:
            self.predict.shape += (1,)
        self.truth = truth
        self.metrics = {}

    def get_all_metrics(self):
        for metric in BinaryEvaluation.__subclasses__():
            self.metrics[metric.__name__] = metric(self.predict.copy(), self.truth.copy())

    def add_metrics(self, list_of_metrics):
        if not isinstance(list_of_metrics, Iterable) or isinstance(list_of_metrics, str):
            list_of_metrics = [list_of_metrics]
        else:
            list_of_metrics = list_of_metrics[:]

        for metric in BinaryEvaluation.__subclasses__():
            if metric.__name__ in list_of_metrics:
                self.metrics[metric.__name__] = metric(self.predict, self.truth)
                list_of_metrics.remove(metric.__name__)

        if len(list_of_metrics):
            raise NameError("Metrics {} is not defined in evaluation library".format(list_of_metrics))

    def remove_metrics(self, list_of_metrics):
        if not isinstance(list_of_metrics, Iterable) or isinstance(list_of_metrics, str):
            list_of_metrics = [list_of_metrics]
        else:
            list_of_metrics = list_of_metrics[:]

        i = 0
        while i < len(list_of_metrics):
            if self.metrics.pop(list_of_metrics[i], False):
                list_of_metrics.remove(list_of_metrics[i])
            else:
                i += 1

        if len(list_of_metrics):
            raise NameError("Metrics {} is not selected".format(list_of_metrics))

    def run_all_metrics(self):
        result = dict()
        for metric in self.metrics.keys():
            result[metric] = self.metrics[metric].run()
        return result


class OccupancyEvaluation:
    def __init__(self,
                 predict,
                 truth):

        if not isinstance(predict, ndarray) or not isinstance(truth, ndarray):
            return

        if truth.shape != predict.shape:
            raise ValueError("Two array have different shape")

        if len(truth.shape) != 1 and truth.shape[1] != 1:
            truth = truth.argmax(axis=1)
            predict = predict.argmax(axis=1)

        predict = round(predict)

        self.predict = predict
        if len(self.predict.shape) == 1:
            self.predict.shape += (1,)
        self.truth = truth
        self.metrics = {}

    def get_all_metrics(self):
        for model in OccupancyEvaluation.__subclasses__():
            self.metrics[model.__name__] = model(self.predict.copy(), self.truth.copy())

    def add_metrics(self, list_of_metrics):
        if not isinstance(list_of_metrics, Iterable) or isinstance(list_of_metrics, str):
            list_of_metrics = [list_of_metrics]
        else:
            list_of_metrics = list_of_metrics[:]

        for metric in OccupancyEvaluation.__subclasses__():
            if metric.__name__ in list_of_metrics:
                self.metrics[metric.__name__] = metric(self.predict, self.truth)
                list_of_metrics.remove(metric.__name__)

        if len(list_of_metrics):
            raise NameError("Metrics {} is not defined in evaluation library".format(list_of_metrics))

    def remove_metrics(self, list_of_metrics):
        if not isinstance(list_of_metrics, Iterable) or isinstance(list_of_metrics, str):
            list_of_metrics = [list_of_metrics]
        else:
            list_of_metrics = list_of_metrics[:]

        i = 0
        while i < len(list_of_metrics):
            if self.metrics.pop(list_of_metrics[i], False):
                list_of_metrics.remove(list_of_metrics[i])
            else:
                i += 1

        if len(list_of_metrics):
            raise NameError("Metrics {} is not selected".format(list_of_metrics))

    def run_all_metrics(self):
        result = dict()
        for metric in self.metrics.keys():
            result[metric] = self.metrics[metric].run()
        return result
