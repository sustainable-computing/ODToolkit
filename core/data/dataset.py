#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Core dataset format. The standard data structure for occupancy detection
#
# Accessible instances:
#     self.time_column_index
#     self.binary
#     self.labelled
#     self.data
#     self.occupancy
#     self.feature_mapping
#     self.feature_list
#     self.room_mapping
#     self.room_list
# Methods:
#     change_values
#     change_occupancy
#     change_room_mapping
#     change_feature_mapping
#     set_feature_name
#     change_feature_name
#     remove_feature
#     select_feature
#     add_room
#     pop_room
#     split
#     copy
# Built-in Methods:
#     __iter__
#     __next__
#     __getitem__
#     __len__
#     __add__
#     __sub__
#     __str__


class Dataset:
    """
    Core data set format. The standard data structure for occupancy and sensor data.

    .. note::
        All attributes are copies of the original values, therefore the changes will only be seen
        if user use methods to update values of ``self``.

    :var time_column_index: the timestamp column in ``self.data``
    :vartype time_column_index: int

    :var binary: indicate the occupancy data in ``self`` has binary encoding or not
    :vartype binary: bool

    :var labelled: indicate whether the occupancy data in ``self`` is available or not
    :vartype labelled: bool

    :parameter: None

    :rtype: core.data.dataset.Dataset
    """
    def __init__(self):
        from numpy import asarray
        self.__data = asarray([])
        self.__occupancy = asarray([])
        # feature_list: column, column: feature_list
        self.__feature_column_mapping = {}
        # room_list: [start_row, end_row], room_counter: room_list
        self.__room_mapping = {}
        self.iter_helper = 0
        self.time_column_index = None
        self.binary = True
        self.labelled = False

    @property
    def data(self):
        """
        :rtype: numpy.ndarray
        :return: a copy of the sensor data in numpy.ndarray
        """
        return self.__data.copy()

    @property
    def occupancy(self):
        """
        :rtype: numpy.ndarray
        :return: a copy of the occupancy data in numpy.ndarray
        """
        return self.__occupancy.copy()

    @property
    def feature_mapping(self):
        """
        :rtype: dict
        :return: a bidirectional dictionary map feature names with corresponding column index
        """
        return self.__feature_column_mapping.copy()

    @property
    def feature_list(self):
        """
        :rtype: list(str)
        :return: a list contains all feature names
        """
        return [self.__feature_column_mapping[i] for i in range(len(self.__feature_column_mapping) // 2)]

    @property
    def room_mapping(self):
        """
        :rtype: dict
        :return: a bidirectional dictionary map room names with corresponding row index tuple (start, end)
        """
        return self.__room_mapping.copy()

    @property
    def room_list(self):
        """
        :rtype: list(str)
        :return: a list contains all room names
        """
        return [self.__room_mapping[i] for i in range(len(self))]

    def change_values(self, data):
        """
        Replace the sensor data of ``self.data``.

        :parameter data: new sensor data have same shape with original sensor data
        :type data: numpy.ndarray

        :return: None
        """
        self.__data = data

    def change_occupancy(self, occupancy):
        """
        Replace the data of ``self.occupancy``.

        :parameter occupancy: new occupancy data have same number of rows with original occupancy data
        :type occupancy: numpy.ndarray

        :return: None
        """
        self.__occupancy = occupancy

    def change_room_mapping(self, room):
        """
        Replace the *room_mapping* within ``self``.

        :parameter room: new room mapping rule with bidirectional dict
        :type room: dict

        :return: None
        """
        self.__room_mapping = room

    def change_feature_mapping(self, feature_mapping):
        """
        Replace the *feature_mapping* within ``self``.

        :parameter feature_mapping: new feature mapping rule with bidirectional dict
        :type feature_mapping: dict

        :return: None
        """
        self.__feature_column_mapping = feature_mapping

    def set_feature_name(self, feature_list):
        """
        Replace all features' name in given order.

        :parameter feature_list: new feature name list have length same as number of columns of ``self.data``
        :type feature_list: list

        :return: None
        """
        from collections import Iterable

        if not isinstance(feature_list, Iterable):
            raise TypeError("Headers must iterable")
        if len(feature_list) != self.__data.shape[1]:
            raise ValueError("Number of headers does not equal to the number of features")
        feature_list = list(map(str, feature_list))

        self.__feature_column_mapping = {}
        for i in range(len(feature_list)):
            self.__feature_column_mapping[i] = feature_list[i]
            self.__feature_column_mapping[feature_list[i]] = i

    def change_feature_name(self, old, new):
        """
        Replace one feature's name.

        :parameter old: original name for the feature in ``self``
        :type old: str

        :parameter new: new name name for the feature in ``self``
        :type new: str

        :return: None
        """
        if old not in self.__feature_column_mapping.keys():
            raise KeyError("The feature {} does not exist in the dataset!".format(old))
        if new in self.__feature_column_mapping.keys():
            raise KeyError("The feature {} already exist in the dataset!".format(new))
        column = self.__feature_column_mapping[old]
        self.__feature_column_mapping[column] = new
        self.__feature_column_mapping.pop(old)
        self.__feature_column_mapping[new] = column

    # Can remove one or more feature
    def remove_feature(self, features, error=True):
        """
        Remove one or multiple features from the ``self.data``.

        :parameter features: one or multiple features that need to be removed
        :type features: str or list(str)

        :parameter error: whether throw an error if a name of feature is not available in ``self``
        :type error: bool

        :return: None
        """
        from collections import Iterable

        if not isinstance(features, Iterable) or isinstance(features, str):
            features = [features]

        column = list(range(self.__data.shape[1]))
        time_name = self.__feature_column_mapping[self.time_column_index]

        for feature in features:
            if feature not in self.__feature_column_mapping.keys():
                if error:
                    raise KeyError("The feature {} does not exist in the dataset!".format(feature))
            else:
                column.remove(self.__feature_column_mapping[feature])
                if self.__feature_column_mapping[feature] == self.time_column_index:
                    self.time_column_index = None

        new_header = [self.__feature_column_mapping[i] for i in column]
        if time_name in new_header:
            self.time_column_index = new_header.index(time_name)

        self.__data = self.__data[:, column]
        self.set_feature_name(new_header)

    # Can select one or more feature
    def select_feature(self, features, error=True):
        """
        Select one or multiple features from the ``self.data``, remove rest features.

        :parameter features: one or multiple features that need to be selected
        :type features: str or list(str)

        :parameter error: whether throw an error if any one of the name in parameter is not available in ``self``
        :type error: bool

        :return: None
        """
        from collections import Iterable

        if not isinstance(features, Iterable) or isinstance(features, str):
            features = [features]

        column = []
        time_name = self.__feature_column_mapping[self.time_column_index]

        for feature in features:
            if feature not in self.__feature_column_mapping.keys():
                if error:
                    raise KeyError("The feature {} does not exist in the dataset!".format(feature))
            else:
                column.append(self.__feature_column_mapping[feature])

        new_header = [self.__feature_column_mapping[i] for i in column]
        if self.time_column_index not in new_header:
            self.time_column_index = None

        new_header = [self.__feature_column_mapping[i] for i in column]
        if time_name in new_header:
            self.time_column_index = new_header.index(time_name)

        self.__data = self.__data[:, column]
        self.set_feature_name(new_header)

    # data is a float matrix of data. All time value need to be changed to its timestamp (datetime.timestamp())
    # if no feature_list line, assume all data have same order as before.
    def add_room(self, data, occupancy=None, room_name=None, header=True):
        """
        Add a new room to ``self``. ``self.data`` can automatically expand.

        :parameter data: sensor data from the new room
        :type data: numpy.ndarray

        :parameter occupancy: occupancy data from the new room. If ``None`` then fill with ``numpy.nan``
        :type occupancy: None or numpy.ndarray

        :parameter room_name: the name of the new room. If ``None`` then assign a unique index
        :type room_name: None or str

        :parameter header: Indicate whether the new room have a header on the first row
        :type header: bool

        :return: None
        """
        from numpy import asarray, unique, full, nan, concatenate

        if header:
            if isinstance(header, bool):
                features = list(data[0])
                data = data[1:]
            else:
                features = header
        else:
            features = list(range(len(list(data[0]))))

        try:
            data = asarray(data, dtype=float)
            if occupancy is not None:
                occupancy = asarray(occupancy, dtype=float)
                self.labelled = True
                if unique(occupancy).shape[0] > 2:
                    self.binary = False
            else:
                occupancy = full([data.shape[0], 1], nan)
        except ValueError:
            raise ValueError("Data cannot convert to float or the shape of data is not a matrix")

        if len(features) != data.shape[1]:
            raise ValueError("Number of headers does not equal to the number of features")
        if occupancy is not None and occupancy.shape[0] != data.shape[0]:
            raise ValueError("Number of ground truth does not equal to the number of entries")

        if room_name is None:
            room_name = len(self.__room_mapping)

        self.__room_mapping[len(self)] = str(room_name)
        self.__room_mapping[str(room_name)] = (self.__data.shape[0], self.__data.shape[0] + data.shape[0])

        if not self.__data.shape[0]:
            self.__data = data
            self.set_feature_name(features)
            if len(occupancy.shape) == 1:
                occupancy.shape += (1,)
            self.__occupancy = occupancy
        else:
            if header:
                target_column = []
                source_column = []
                rest_column = []
                for i in range(len(features)):
                    if features[i] in self.__feature_column_mapping.keys():
                        target_column.append(self.__feature_column_mapping[features[i]])
                        source_column.append(i)
                    else:
                        rest_column.append(i)
            else:
                target_column = list(range(self.__data.shape[1]))
                source_column = target_column
                rest_column = list(range(self.__data.shape[1], data.shape[1]))

            new_data = full([data.shape[0], self.__data.shape[1] + len(rest_column)], nan)
            new_data[:, target_column] = data[:, source_column]
            new_data[:, self.__data.shape[1]:] = data[:, rest_column]

            self.__data = concatenate((self.__data,
                                       full([self.__data.shape[0], len(rest_column)], nan)), axis=1)
            self.__data = concatenate((self.__data, new_data), axis=0)

            new_header = []
            i = 0
            while self.__feature_column_mapping.get(i, False):
                new_header.append(self.__feature_column_mapping[i])
                i += 1

            for name in rest_column:
                new_header.append(name)

            self.set_feature_name(new_header)

            if len(occupancy.shape) == 1:
                occupancy.shape += (1,)
            self.__occupancy = concatenate((self.__occupancy, occupancy), axis=0)

    def pop_room(self, room_name):
        """
        Remove a room from ``self``.

        :parameter room_name: name of the room need to be removed
        :type room_name: str

        :rtype: core.data.dataset.Dataset
        :return: removed Dataset
        """
        from numpy import delete, unique, isnan

        if room_name not in self.__room_mapping.keys():
            raise KeyError("This dataset do not contain room_list {}".format(room_name))

        a, b = self.__room_mapping[room_name]
        pop_data, pop_occupancy = self[room_name]

        new_dataset = Dataset()
        new_dataset.add_room(pop_data, pop_occupancy, room_name=room_name, header=self.feature_list)

        self.__data = delete(self.__data, range(a, b), axis=0)
        self.__occupancy = delete(self.__occupancy, range(a, b), axis=0)

        unique_entry = unique(self.__occupancy)
        if unique_entry.shape[0] <= 2:
            self.binary = True
            if unique_entry.shape[0] == 1 and isnan(unique_entry[0]):
                self.labelled = False

        remove_col = b - a
        found = False
        for i in range(len(self) - 1):
            if self.__room_mapping[i] == room_name:
                found = True

            if found:
                new_a, new_b = self.__room_mapping[self.__room_mapping[i + 1]]
                self.__room_mapping[self.__room_mapping[i]] = (new_a - remove_col, new_b - remove_col)
                self.__room_mapping[i] = self.__room_mapping[i + 1]

        self.__room_mapping.pop(room_name)
        self.__room_mapping.pop(len(self) - 1)

        return new_dataset

    def split(self, percentage):
        """
        Separate ``self`` into two smaller ``core.data.dataset.Dataset`` objects by given split point.

        :parameter percentage: percentage of the row in the first part
        :type percentage: float

        :return: None
        """
        front_dataset = Dataset()
        back_dataset = Dataset()
        front_dataset.time_column_index = self.time_column_index
        front_dataset.binary = self.binary
        front_dataset.labelled = self.labelled
        back_dataset.time_column_index = self.time_column_index
        back_dataset.binary = self.binary
        back_dataset.labelled = self.labelled

        split_point = round(percentage * self.__data.shape[0])

        for room in self.room_list:
            room_data, room_occupancy = self[room]
            if self.__room_mapping[room][1] <= split_point:
                front_dataset.add_room(room_data, room_occupancy, room_name=room, header=self.feature_list)
            elif self.__room_mapping[room][0] > split_point:
                back_dataset.add_room(room_data, room_occupancy, room_name=room, header=self.feature_list)
            else:
                start_pos = self.__room_mapping[room][0]
                mid_pos = split_point - start_pos
                front_room_data = room_data[:mid_pos, :]
                front_room_occupancy = room_occupancy[:mid_pos, :]
                back_room_data = room_data[mid_pos:, :]
                back_room_occupancy = room_occupancy[mid_pos:, :]
                front_dataset.add_room(front_room_data, front_room_occupancy,
                                       room_name="Partially " + str(room), header=self.feature_list)
                back_dataset.add_room(back_room_data, back_room_occupancy,
                                      room_name="Partially " + str(room), header=self.feature_list)

        return front_dataset, back_dataset

    def copy(self):
        """
        Make a copy of ``self``.

        :parameter: None

        :rtype: core.data.dataset.Dataset
        :return: A same copy of ``self``, with different addresses for all values
        """
        duplicate = Dataset()
        duplicate.change_values(self.__data.copy())
        duplicate.change_occupancy(self.__occupancy.copy())
        duplicate.change_feature_mapping(self.__feature_column_mapping.copy())
        duplicate.change_room_mapping(self.__room_mapping.copy())
        duplicate.time_column_index = self.time_column_index
        duplicate.binary = self.binary
        duplicate.labelled = self.labelled
        return duplicate

    def __iter__(self):
        self.iter_helper = 0
        return self

    def __next__(self):
        room_name = self.__room_mapping.get(self.iter_helper, None)
        if room_name is None:
            raise StopIteration
        else:
            self.iter_helper += 1
            a, b = self.__room_mapping[room_name]
            return self.__data[a:b, :], self.__occupancy[a:b, :]

    def __getitem__(self, room_name):
        a, b = self.__room_mapping[room_name]
        return self.__data[a:b, :], self.__occupancy[a:b, :]

    def __len__(self):
        return len(self.__room_mapping) // 2

    def __add__(self, other):
        if not isinstance(other, Dataset):
            raise TypeError("Dataset need to add with Dataset")
        rooms = other.room_list
        header = other.feature_list
        for room in rooms:
            data, occupancy = other[room]
            while room in self.__room_mapping.values():
                room = str(int(room) + 1)
            self.add_room(data, occupancy=occupancy, room_name=room, header=header)
        return self

    def __sub__(self, other):
        if not isinstance(other, Dataset):
            raise TypeError("Dataset need to sub with Dataset")
        rooms = other.room_list
        for room in rooms:
            if room in self.room_list:
                self.remove_room(room)
        return self

    def __str__(self):
        return str(self.__dict__)
