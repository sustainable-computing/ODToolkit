#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def dropout_rate(dataset, dataset_level=False):
    """
    Compute the dropout rate for a given dataset.
    Dropout rate is the percent of rows that is invalid

    :parameter dataset: Dataset object that want to compute the dropout rate.
                        The dropout rate is the percentage of data points missing in Dataset
    :type dataset: core.data.dataset.Dataset

    :parameter dataset_level: decide the result is separate for each room in room_list or
                              combine for the whole dataset together
    :type dataset_level: bool

    :rtype: str or dict(str, str)
    :return: the room name with its corresponding dropout rate
    """
    from core.data.dataset import Dataset
    from numpy import isnan

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")
    if dataset_level:
        data = dataset.data
        drop_row = ((~isnan(data)).sum(axis=1) != data.shape[1]).sum()
        return drop_row / data.shape[0]
    else:
        result = {}
        rooms = dataset.room_list

        for room in rooms:
            data = dataset[room][0]
            drop_row = ((~isnan(data)).sum(axis=1) != data.shape[1]).sum()
            result[room] = drop_row / data.shape[0]

        return result
