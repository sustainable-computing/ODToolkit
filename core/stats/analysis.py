#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def analysis(dataset, threshold, save_to=None, print_out=False):
    """
    The full analysis for the given core.data.dataset.Dataset

    :parameter dataset: Dataset object that want to perform evaluation
    :type dataset: core.data.dataset.Dataset

    :parameter threshold: the maximum time differences in seconds between two consecutive timestamp
                          to not mark them as a gap
    :type threshold: int

    :parameter save_to: the file name of function's output result.
                        if None, then do not write analysis result to a file.
                        Otherwise, write analysis result to save_file
    :type save_to: str

    :parameter print_out: decide if analysis result should print to stdout or not
    :type print_out: bool

    :rtype: dict(str, result)
    :return: Analysis result in human readable format
    """
    from core.data.dataset import Dataset
    from .dropout_rate import dropout_rate
    from .frequency import frequency
    from .gap_detect import gap_detect
    from .occupancy_evaluation import occupancy_distribution_evaluation
    from .uptime import uptime
    from pprint import pprint

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class core.data.dataset.Dataset")

    result = dict()
    result["Name of features"] = dataset.feature_list
    result["Total number of features"] = len(result["Name of features"])
    result["Name of rooms"] = dataset.room_list
    result["Total number of rooms"] = len(result["Name of rooms"])
    result["Dataset is labelled"] = dataset.labelled
    result["Dataset label is binary"] = dataset.binary
    result["Maximum record interval (sec)"] = threshold

    rooms = dataset.room_list
    detail_room = dataset.room_mapping
    num_of_entries = {}
    for room in rooms:
        num_of_entries[room] = detail_room[room][1] - detail_room[room][0]

    result["Number of entries in each room_list"] = num_of_entries
    result["Total number of entries"] = sum(num_of_entries.values())

    dropout = dropout_rate(dataset, dataset_level=False)
    for room in dropout.keys():
        dropout[room] = "{:.5f}%".format(dropout[room] * 100)
    result["Dropout rate for each room_list (%)"] = dropout
    result["Total dropout rate (%)"] = "{:.5f}%".format(dropout_rate(dataset, dataset_level=True) * 100)

    result["Average frequency for each toom (sec)"] = frequency(dataset, dataset_level=False)
    result["Average frequency over dataset (sec)"] = frequency(dataset, dataset_level=True)

    gaps = gap_detect(dataset, threshold, sensor_level=False)
    for room in gaps.keys():
        for i in range(len(gaps[room])):
            gaps[room][i] = "From {} to {}".format(gaps[room][i][0], gaps[room][i][1])
    result["Missing interval for each room_list"] = gaps
    gaps = gap_detect(dataset, threshold, sensor_level=True)
    detail_uptime = uptime(dataset, result["Average frequency over dataset (sec)"], gaps=gaps)
    for room in gaps.keys():
        for sensor in gaps[room].keys():
            for i in range(len(gaps[room][sensor])):
                gaps[room][sensor][i] = "From {} to {}".format(gaps[room][sensor][i][0], gaps[room][sensor][i][1])
    result["Missing interval for each sensor in each room_list"] = gaps

    occupancy = occupancy_distribution_evaluation(dataset, dataset_level=False)
    summarize = {}
    for values in occupancy.values():
        for occ in values.keys():
            summarize[occ] = summarize.get(occ, 0) + values[occ][0]
            values[occ] = "{:.5f}%".format(100 * values[occ][1])
    for occ in summarize.keys():
        summarize[occ] = "{:.5f}%".format(100 * summarize[occ] / result["Total number of entries"])

    result["Occupancy distribution for each room_list"] = occupancy
    result["Occupancy distribution over dataset"] = summarize

    for room in detail_uptime.keys():
        for sensor in detail_uptime[room].keys():
            detail_uptime[room][sensor] = (detail_uptime[room][sensor][0],
                                           "{:.5f}%".format(100 * detail_uptime[room][sensor][2]))
    result["Sensor uptime for each room_list"] = detail_uptime

    if print_out:
        pprint(result, width=140)
    if save_to is not None:
        with open(save_to, 'w') as file:
            pprint(result, stream=file, width=140)

    return result
