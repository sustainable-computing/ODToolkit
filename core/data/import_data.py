#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def import_data(file_name, time_column_index=None, mode='csv', header=True, room_name=None, tz=0):
    """
    Load raw data from the disk.

    :type file_name: str
    :param file_name: the name of the raw data file

    :type time_column_index: int
    :param time_column_index: the column index for the timestamp in given raw data file

    :type mode: str
    :param mode: the format for raw data. Currently only support ``csv``

    :type header: bool
    :param header: indicate whether the raw data contains a header on the first row. If ``False``, then assign unique
                   index for each column

    :type room_name: str or None
    :param room_name: the name of the room. If ``None``, then assign unique number for the room

    :type tz: int
    :param tz: the time zone offset that need to fix in the raw data file

    :rtype: core.data.dataset.Dataset
    :return: The structured data set with one raw input data
    """
    from csv import reader
    from dateutil.parser import parse
    from numpy import nan, asarray
    from .dataset import Dataset

    if mode == 'csv':
        with open(file_name, 'r') as input_file:
            csv_reader = reader(input_file, delimiter=',')
            feature_name = []
            data = []
            if header:
                feature_name = next(csv_reader)[:-1]

            for line in csv_reader:
                if not len(line):
                    continue

                for i in range(len(line)):
                    if i == time_column_index:
                        line[i] = parse(line[i]).timestamp() + tz * 60 * 60
                    elif not len(line[i]):
                        line[i] = nan
                    else:
                        try:
                            line[i] = float(line[i])
                        except ValueError:
                            line[i] = nan

                data.append(line)
            data = asarray(data, dtype=float)

            if not len(feature_name):
                feature_name = list(range(data.shape[1]))

        dataset = Dataset()
        dataset.add_room(data[:, :-1], occupancy=data[:, -1], header=False, room_name=room_name)
        dataset.set_feature_name(feature_name)
        dataset.time_column_index = time_column_index
        return dataset
