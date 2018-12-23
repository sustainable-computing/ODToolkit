from odtk.data.read import *
from odtk.data.write import *
from odtk.data.import_data import *
import numpy as np
from collections import Iterable


class Dataset:
    def __init__(self):
        print("Class Dataset imported")
        self.__data = np.array([])
        # header: column, column: header
        self.__header = {}
        # room: [start_row, end_row], room_counter: room
        self.__room = {}
        self.iter_helper = 0

    @property
    def data(self):
        return self.__data.copy()

    @property
    def header(self):
        return self.__header.copy()

    @property
    def room(self):
        return self.__room.copy()

    def set_header(self, header):
        if not isinstance(header, Iterable):
            raise TypeError("Headers must iterable")
        if len(header) != self.__data.shape[1]:
            raise ValueError("Number of headers does not equal to the number of features")

        self.__header = {}
        for i in range(len(header)):
            self.__header[i] = header[i]
            self.__header[header[i]] = i

    def change_header(self, old, new):
        if old not in self.__header.keys():
            raise KeyError("The feature %s does not exist in the dataset!".format(old))
        if new in self.__header.keys():
            raise KeyError("The feature %s already exist in the dataset!".format(new))
        column = self.__header[old]
        self.__header[column] = new
        self.__header.pop(old)
        self.__header[new] = column

    # Can remove one or more feature
    def remove_feature(self, features, error=True):
        if not isinstance(features, Iterable):
            features = [features]

        column = list(range(self.__data.shape[1]))

        for feature in features:
            if feature not in self.__header.keys():
                if error:
                    raise KeyError("The feature %s does not exist in the dataset!".format(feature))
            else:
                column.remove(self.__header[feature])

        new_header = [self.__header[i] for i in column]

        self.__data = self.__data[:, column]
        self.set_header(new_header)

    # data is a float matrix of data. All time value need to be changed to its timestamp (datetime.timestamp())
    # if no header line, assume all data have same order as before.
    def add_room(self, data, room_name=None, header=True):
        if header:
            features = list(data[0])
            data = data[1:]
        else:
            features = list(range(len(list(data[0]))))

        try:
            data = np.asarray(data, dtype=float)
        except ValueError:
            raise ValueError("Data cannot convert to float or the shape of data is not a matrix")

        if len(features) != data.shape[1]:
            raise ValueError("Number of headers does not equal to the number of features")

        if room_name is None:
            room_name = len(self.__room)

        self.__room[len(self.__room) // 2] = room_name
        self.__room[room_name] = (self.__data.shape[0] + 1, self.__data.shape[0] + data.shape[0])

        if not self.__data.shape[0]:
            self.__data = data
            self.set_header(features)
        else:
            if header:
                target_column = []
                source_column = []
                rest_column = []
                for i in range(len(features)):
                    if features[i] in self.__header.keys():
                        target_column.append(self.__header[features[i]])
                        source_column.append(i)
                    else:
                        rest_column.append(i)
            else:
                target_column = list(range(self.__data.shape[1]))
                source_column = target_column
                rest_column = list(range(self.__data.shape[1], data.shape[1]))

            new_data = np.full([data.shape[0], self.__data.shape[1] + len(rest_column)], np.nan)
            new_data[:, target_column] = data[:, source_column]
            new_data[:, self.__data.shape[1]:] = data[:, rest_column]

            self.__data = np.concatenate((self.__data,
                                          np.full([self.__data.shape[0], len(rest_column)], np.nan)), axis=1)
            self.__data = np.concatenate((self.__data, new_data), axis=0)

            new_header = []
            i = 0
            while self.__header.get(i, False):
                new_header.append(self.__header[i])
                i += 1

            for name in rest_column:
                new_header.append(name)

            self.set_header(new_header)

    def __iter__(self):
        self.iter_helper = 0
        return self

    def __next__(self):
        room_name = self.__room.get(self.iter_helper, None)
        if room_name is None:
            raise StopIteration
        else:
            self.iter_helper += 1
            a, b = self.__room[room_name]
            return self.__data[a:b, :]

    def __getitem__(self, room_name):
        a, b = self.__room[room_name]
        return self.__data[a:b, :]

    def __len__(self):
        return len(self.__room)
