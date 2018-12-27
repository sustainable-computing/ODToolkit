from numpy import abs, round, sqrt


# Multiple different evaluation matrices, includes 11 different f-measure, RMSE, nRMSE, MAE, MAPE, MASE
#
# Parameters:
#   truth, estimation: two numpy.ndarray that are both one-hot or label encoded represents the occupancy level
#   truth: ground truth from the dataset
#   estimation: model result
# Return:
#   the computed score


# f_score have 11 mode, and one extra parameter
#   tolerance: maximum different number of occupancy that can omit. Only work for compute accuracy
#   mode includes: (https://en.wikipedia.org/wiki/F1_score)
#     true-positive
#     false-negative
#     false-positive
#     true-negative
#     recall
#     fall-out
#     miss-rate
#     selectivity
#     precision
#     accuracy: only this for non-binary evaluation
#     f1-score
def f_score(truth, estimation, tolerance=0, mode="f1-score"):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)

    occupied_index = truth > 0
    unoccupied_index = ~occupied_index
    tn = (estimation[unoccupied_index] == 0).sum()

    if mode == "accuracy":
        tp = (abs(estimation[occupied_index] - truth[occupied_index]) <= tolerance).sum()
        return (tn + tp) / truth.shape[0]

    tp = (estimation[occupied_index] > 0).sum()
    fn = (estimation[occupied_index] == 0).sum()
    fp = (estimation[unoccupied_index] > 0).sum()

    if mode == "true-positive":
        return tp
    elif mode == "false-negative":
        return fn
    elif mode == "false-positive":
        return fp
    elif mode == "true-negative":
        return tn
    elif mode == "recall":
        return tp / (tp + fn)
    elif mode == "fall-out":
        return fp / (fp + tn)
    elif mode == "miss-rate":
        return fn / (tp + fn)
    elif mode == "selectivity":
        return tn / (fp + tn)
    elif mode == "precision":
        return tp / (tp + fp)
    elif mode == "f1-score":
        return 2 * tp / (2 * tp + fn + fp)


def rmse(truth, estimation):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)
    return sqrt(((truth - estimation) ** 2).mean())


def nrmse(truth, estimation):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)
    return sqrt(((truth - estimation) ** 2).mean()) / (truth.max() - truth.min())


def mape(truth, estimation):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)
    occupied_index = truth > 0
    return abs(estimation[occupied_index] / truth[occupied_index]).mean()


def mase(truth, estimation):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)
    mae = abs(truth - estimation).mean()
    denominator = truth.shape[0] / (truth.shape[0] - 1)
    denominator *= abs(truth[1:] - truth[:-1]).mean()
    return mae / denominator


def mae(truth, estimation):
    if truth.shape != estimation.shape:
        raise ValueError("Two array have different shape")

    if len(truth.shape) != 1 and truth.shape[1] != 1:
        truth = truth.argmax(axis=1)
        estimation = estimation.argmax(axis=1)

    estimation = round(estimation)
    return abs(truth - estimation).mean()
