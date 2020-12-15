#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def remove_outlier(dataset, auto_fill=True, ratio=1.5):
    """
    Remove potential outliers using IQR, and fill with numpy.nan. Outliers are the value that
    less than value at the first quantile - ratio * IOR or
    greater than value at the third quantile + ratio * IOR
    in its corresponding column

    :parameter dataset: Dataset object that wants to remove outliers
    :type dataset: core.data.dataset.Dataset

    :parameter auto_fill: whether automatically fill the outliers or leave it nan
    :type auto_fill: bool

    :parameter ratio: IQR ratio in order to mark value as an outlier
    :type ratio: float

    :return: None
    """
    from ..data import Dataset
    from .fill import fill
    from numpy import percentile, isnan, nonzero, nan

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    data = dataset.data
    time_col = dataset.time_column_index

    for i in range(data.shape[1]):
        if i == time_col:
            continue
        column = nonzero(~isnan(data[:, i]))[0]
        q25, q75 = percentile(data[column, i], (25, 75))
        iqr = q75 - q25
        data[column[nonzero(data[column, i] < (q25 - ratio * iqr))[0]], i] = nan
        data[column[nonzero(data[column, i] > (q75 + ratio * iqr))[0]], i] = nan
        dataset.change_values(data)

    if auto_fill:
        fill(dataset)
