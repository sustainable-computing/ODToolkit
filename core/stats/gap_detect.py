#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def gap_detect(dataset, threshold, sensor_level=False):
    """
    Compute the gaps in the given dataset.
    Gap is a time sequence that two consecutive row have timestamp differences greater than threshold

    :parameter dataset: Dataset object that want to find the gaps
    :type dataset: core.data.dataset.Dataset

    :parameter threshold: the maximum time differences in seconds between two consecutive timestamp
                          to not mark them as a gap
    :type threshold: int

    :parameter sensor_level: decide the result is separate for each sensor in feature_list or
                             combine for the whole dataset together
    :type sensor_level: bool

    :rtype: dict(str, list(str)) or dict(str, dict(str, list(str)))
    :return: the room name corresponds to the name of sensor with its corresponding dropout rate
    """
    from core.data.dataset import Dataset
    from numpy import isnan, where
    from datetime import datetime

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    result = {}
    rooms = dataset.room_list
    time_col = dataset.time_column_index
    sensors = dataset.feature_list
    sensor_dict = dataset.feature_mapping

    for room in rooms:
        data = dataset[room][0]

        if sensor_level:
            result[room] = {}
            for sensor in sensors:
                if sensor == sensors[time_col]:
                    continue
                result[room][sensor] = []

                valid_data = data[(~isnan(data[:, [time_col, sensor_dict[sensor]]])).sum(axis=1) == 2, time_col]
                indices = where(valid_data[1:] - valid_data[:-1] >= threshold)[0]
                for period in indices:
                    result[room][sensor].append((str(datetime.fromtimestamp(valid_data[period])),
                                                 str(datetime.fromtimestamp(valid_data[period + 1])),
                                                 valid_data[period + 1] - valid_data[period]))

        else:
            result[room] = []
            valid_data = data[(~isnan(data)).sum(axis=1) == data.shape[1], time_col]
            indices = where(valid_data[1:] - valid_data[:-1] >= threshold)[0]
            for period in indices:
                result[room].append((str(datetime.fromtimestamp(valid_data[period])),
                                     str(datetime.fromtimestamp(valid_data[period + 1])),
                                     valid_data[period + 1] - valid_data[period]))

    return result
