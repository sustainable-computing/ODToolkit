from csv import reader
# from os import path
# from inspect import stack, getmodule
from dateutil.parser import parse
from numpy import nan, asarray
from odtk.data.dataset import Dataset


def import_data(file_name, time_column=None, mode='csv', header=True):
    # relative_file = path.dirname(getmodule(stack()[1][0]).__file__) + '/' + file_name
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
                    if not len(line[i]):
                        line[i] = nan
                    elif i == time_column:
                        line[i] = parse(line[i]).timestamp()

                data.append(line)
            data = asarray(data, dtype=float)

            if not len(feature_name):
                feature_name = list(range(data.shape[1]))

        dataset = Dataset()
        dataset.add_room(data[:, :-1], occupancy=data[:, -1], header=False)
        dataset.set_header(feature_name)
        dataset.time_column = time_column
        return dataset
