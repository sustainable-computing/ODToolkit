#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BinaryEvaluation:
    """
    Use all binary occupancy evaluation metrics to evaluate the differences between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: core.evaluation.superclass.BinaryEvaluation
    """
    def __init__(self,
                 predict,
                 truth):
        from numpy import ndarray, round

        if not isinstance(predict, ndarray) or not isinstance(truth, ndarray):
            return

        if truth.shape != predict.shape:
            print(truth.shape, predict.shape)
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
        """
        Get all subclasses

        :parameter: None

        :return: None
        """
        for metric in BinaryEvaluation.__subclasses__():
            self.metrics[metric.__name__] = metric(self.predict.copy(), self.truth.copy())

    def add_metrics(self, list_of_metrics):
        """
        Add one or multiple metrics into the evaluation queue

        :parameter list_of_metrics: one or multiple metrics that additionally add to the evaluation queue.
        :type list_of_metrics: str or list(str)

        :return: None
        """
        from collections import Iterable

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
        """
        Remove one or multiple metrics from the evaluation queue

        :parameter list_of_metrics: one or multiple metrics that want to remove from the evaluation queue
        :type list_of_metrics: str or list(str)

        :return: None
        """
        from collections import Iterable

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
        """
        Run all metrics that currently in the queue

        :parameter: None

        :rtype: dict(str, float or int)
        :return: a dictionary map each metrics and their corresponding result
        """
        result = dict()
        for metric in self.metrics.keys():
            result[metric] = self.metrics[metric].run()
        return result


class OccupancyEvaluation:
    """
    Use all occupancy level estimation metrics to evaluate the differences between prediction and ground truth

    :parameter predict: the predicted values from occupancy estimation models
    :type predict: numpy.ndarray

    :parameter truth: the ground truth value from the Dataset
    :type truth: numpy.ndarray

    :rtype: core.evaluation.superclass.OccupancyEvaluation
    """
    def __init__(self,
                 predict,
                 truth):
        from numpy import ndarray, round

        if not isinstance(predict, ndarray) or not isinstance(truth, ndarray):
            return

        if truth.shape != predict.shape:
            print(truth.shape, predict.shape)
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
        """
        Get all subclasses

        :parameter: None

        :return: None
        """
        for model in OccupancyEvaluation.__subclasses__():
            self.metrics[model.__name__] = model(self.predict.copy(), self.truth.copy())

    def add_metrics(self, list_of_metrics):
        """
        Add one or multiple metrics into the evaluation queue

        :parameter list_of_metrics: one or multiple metrics that additionally add to the evaluation queue
        :type list_of_metrics: str or list(str)

        :return: None
        """
        from collections import Iterable

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
        """
        Remove one or multiple metrics from the evaluation queue

        :parameter list_of_metrics: one or multiple metrics that want to remove from the evaluation queue
        :type list_of_metrics: str or list(str)

        :return: None
        """
        from collections import Iterable

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
        """
        Run all metrics that currently in the queue

        :parameter: None

        :rtype: dict(str, float or int)
        :return: a dictionary map each metrics and their corresponding result
        """
        result = dict()
        for metric in self.metrics.keys():
            result[metric] = self.metrics[metric].run()
        return result


# core.evaluation.superclass.Result is for transform the evaluation result to a 3D array and select required data.


class Result:
    """
    Create a 3D array to fast select and reshape result

    :parameter: None

    :return: core.evaluation.superclass.Result
    """
    def __init__(self):
        self.result = None
        self.metrics = list()
        self.models = list()
        self.datasets = list()

    def set_result(self, result):
        """
        Initialize the data in self

        :parameter result: whole result from the experiment
        :type result: dict(str, dict(str, dict(str, float or int)))

        :return: None
        """
        from numpy import zeros, isnan

        self.datasets = list(result.keys())
        self.models = list(result[self.datasets[0]].keys())
        self.metrics = list(result[self.datasets[0]][self.models[0]].keys())

        self.result = zeros((len(self.datasets), len(self.models), len(self.metrics)), dtype=float)

        for i in range(len(self.datasets)):
            for j in range(len(self.models)):
                for k in range(len(self.metrics)):
                    try:
                        if isnan(result[self.datasets[i]][self.models[j]][self.metrics[k]]):
                            self.result[i][j][k] = 0
                        else:
                            self.result[i][j][k] = result[self.datasets[i]][self.models[j]][self.metrics[k]]
                    except KeyError:
                        continue

    def get_result(self, dataset=None, model=None, metric=None, fixed="auto"):
        """
        Shrink, select and reshape result by given require query

        :parameter dataset: one or multiple datasets that user want as result. If ``None`` then all datasets will
                            be selected
        :type dataset: str or None or list(str)

        :parameter model: one or multiple models that user want as result. If ``None`` then all models will
                          be selected
        :type model: str or None or list(str)

        :parameter metric: one or multiple metrics that user want as result. If ``None`` then all metrics will
                           be selected
        :type metric: str or None or list(str)

        :parameter fixed: find which asix only have one value in order to create 2D result. If ``'auto'`` then it will
                          automatically find the dimension with only one value. Value must be ``'auto'``, ``'dataset'``,
                          ``'model'``, or ``'metric'``
        :type fixed: str

        :rtype: numpy.ndarary
        :return: a 2D array contains the data for plotting
        """
        from numpy import ix_
        dimension = {"dataset": None, "model": None, "metric": None}

        for dim in dimension.keys():
            if eval(dim) is None:
                dimension[dim] = eval("list(range(len(self." + dim + "s)))")
            elif not isinstance(eval(dim), list):
                if fixed == "auto":
                    fixed = dim
                my_func = "[self." + dim + "s.index(" + dim + ")]"
                dimension[dim] = eval(my_func)
            else:
                my_func = "[self." + dim + "s.index(i) for i in " + dim + ']'
                dimension[dim] = eval(my_func, {"self": self, dim: eval(dim)})

        if fixed == "dataset":
            result = self.result[dimension["dataset"], :, :]
            result = result.reshape((result.shape[1], result.shape[2]))
            return [self.models[i] for i in dimension["model"]], \
                   [self.metrics[i] for i in dimension["metric"]], \
                   result[ix_(dimension["model"], dimension["metric"])]
        elif fixed == "model":
            result = self.result[:, dimension["model"], :]
            result = result.reshape((result.shape[0], result.shape[2]))
            return [self.datasets[i] for i in dimension["dataset"]], \
                   [self.metrics[i] for i in dimension["metric"]], \
                   result[ix_(dimension["dataset"], dimension["metric"])]
        elif fixed == "metric":
            result = self.result[:, :, dimension["metric"]]
            result = result.reshape((result.shape[0], result.shape[1]))
            return [self.datasets[i] for i in dimension["dataset"]], \
                   [self.models[i] for i in dimension["model"]], \
                   result[ix_(dimension["dataset"], dimension["model"])]

        raise ValueError("Target request not found")
