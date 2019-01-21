from numpy import ndarray, round


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

        self.get_all_metrics()

    def get_all_metrics(self):
        for model in BinaryEvaluation.__subclasses__():
            self.metrics[model.__name__] = model(self.predict.copy(), self.truth.copy())

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

        self.get_all_metrics()

    def get_all_metrics(self):
        for model in OccupancyEvaluation.__subclasses__():
            self.metrics[model.__name__] = model(self.predict.copy(), self.truth.copy())

    def run_all_metrics(self):
        result = dict()
        for metric in self.metrics.keys():
            result[metric] = self.metrics[metric].run()
        return result
