from odtk.data.dataset import Dataset
from numpy import isnan, where
from datetime import datetime


# Compute the gaps in the given dataset
# Gap is a time sequence that two sample row have timestamp difference greater than threshold
#
# Parameters:
#   dataset: odtk.data.dataset.Dataset()
#   threshold: maximum different second between two consecutive row
#   detail: decide the result is separate for each sensor or combine the whole row together
# Return:
#   a dictionary contains all gaps for each room
def gap_detect(dataset, threshold, detail=False):
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset has to be class odtk.data.dataset.Dataset")

    result = {}
    rooms = dataset.room
    time_col = dataset.time_column
    sensors = dataset.header
    sensor_dict = dataset.header_info

    for room in rooms:
        data = dataset[room][0]

        if detail:
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
                print(valid_data[period])
                result[room].append((str(datetime.fromtimestamp(valid_data[period])),
                                     str(datetime.fromtimestamp(valid_data[period + 1])),
                                     valid_data[period + 1] - valid_data[period]))

    return result
