from .__init__ import *


# The full analysis for the given odtk.data.dataset.Dataset()
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
#     threshold: maximum different second between two consecutive timestamp
#     save_to: if None, then do not write analysis result to a file. Otherwise, write analysis result to save_file
#     print_out: decide if analysis result print to stdout or not
#
# Return:
#     a dictionary contains all information
def analysis(dataset, threshold, save_to=None, print_out=False):
    from odtk.data.dataset import Dataset
    from pprint import pprint

    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = dict()
    result["Name of features"] = dataset.header
    result["Total number of features"] = len(result["Name of features"])
    result["Name of rooms"] = dataset.room
    result["Total number of rooms"] = len(result["Name of rooms"])
    result["Dataset is labelled"] = dataset.labelled
    result["Dataset label is binary"] = dataset.binary
    result["Maximum record interval (sec)"] = threshold

    rooms = dataset.room
    detail_room = dataset.room_info
    num_of_entries = {}
    for room in rooms:
        num_of_entries[room] = detail_room[room][1] - detail_room[room][0]

    result["Number of entries in each room"] = num_of_entries
    result["Total number of entries"] = sum(num_of_entries.values())

    dropout = dropout_rate(dataset, total=False)
    for room in dropout.keys():
        dropout[room] = "{:.5f}%".format(dropout[room] * 100)
    result["Dropout rate for each room (%)"] = dropout
    result["Total dropout rate (%)"] = "{:.5f}%".format(dropout_rate(dataset, total=True) * 100)

    result["Average frequency for each toom (sec)"] = frequency(dataset, total=False)
    result["Average frequency over dataset (sec)"] = frequency(dataset, total=True)

    gaps = gap_detect(dataset, threshold, detail=False)
    for room in gaps.keys():
        for i in range(len(gaps[room])):
            gaps[room][i] = "From {} to {}".format(gaps[room][i][0], gaps[room][i][1])
    result["Missing interval for each room"] = gaps
    gaps = gap_detect(dataset, threshold, detail=True)
    detail_uptime = uptime(dataset, result["Average frequency over dataset (sec)"], gaps=gaps)
    for room in gaps.keys():
        for sensor in gaps[room].keys():
            for i in range(len(gaps[room][sensor])):
                gaps[room][sensor][i] = "From {} to {}".format(gaps[room][sensor][i][0], gaps[room][sensor][i][1])
    result["Missing interval for each sensor in each room"] = gaps

    occupancy = occupancy_evaluation(dataset, total=False)
    summarize = {}
    for values in occupancy.values():
        for occ in values.keys():
            summarize[occ] = summarize.get(occ, 0) + values[occ][0]
            values[occ] = "{:.5f}%".format(100 * values[occ][1])
    for occ in summarize.keys():
        summarize[occ] = "{:.5f}%".format(100 * summarize[occ] / result["Total number of entries"])

    result["Occupancy distribution for each room"] = occupancy
    result["Occupancy distribution over dataset"] = summarize

    for room in detail_uptime.keys():
        for sensor in detail_uptime[room].keys():
            detail_uptime[room][sensor] = (detail_uptime[room][sensor][0],
                                           "{:.5f}%".format(100 * detail_uptime[room][sensor][2]))
    result["Sensor uptime for each room"] = detail_uptime

    if print_out:
        pprint(result, width=140)
    if save_to is not None:
        with open(save_to, 'w') as file:
            pprint(result, stream=file, width=140)

    return result
