from numpy import asarray, concatenate, nan, full, unique, delete, isnan
from collections import Iterable


class Dataset:
    def __init__(self):
        self.__data = asarray([])
        self.__occupancy = asarray([])
        # header: column, column: header
        self.__header = {}
        # room: [start_row, end_row], room_counter: room
        self.__room = {}
        self.iter_helper = 0
        self.time_column = None
        self.binary = True
        self.labelled = False

    @property
    def data(self):
        return self.__data.copy()

    @property
    def occupancy(self):
        return self.__occupancy.copy()

    @property
    def header_info(self):
        return self.__header.copy()

    @property
    def header(self):
        return [self.__header[i] for i in range(len(self.__header) // 2)]

    @property
    def room_info(self):
        return self.__room.copy()

    @property
    def room(self):
        return [self.__room[i] for i in range(len(self))]

    def change_values(self, data):
        self.__data = data

    def change_occupancy(self, occupancy):
        self.__occupancy = occupancy

    def change_room_info(self, room):
        self.__room = room

    def change_whole_header(self, header):
        self.__header = header

    def set_header(self, header):
        if not isinstance(header, Iterable):
            raise TypeError("Headers must iterable")
        if len(header) != self.__data.shape[1]:
            raise ValueError("Number of headers does not equal to the number of features")
        header = list(map(str, header))

        self.__header = {}
        for i in range(len(header)):
            self.__header[i] = header[i]
            self.__header[header[i]] = i

    def change_header(self, old, new):
        if old not in self.__header.keys():
            raise KeyError("The feature {} does not exist in the dataset!".format(old))
        if new in self.__header.keys():
            raise KeyError("The feature {} already exist in the dataset!".format(new))
        column = self.__header[old]
        self.__header[column] = new
        self.__header.pop(old)
        self.__header[new] = column

    # Can remove one or more feature
    def remove_feature(self, features, error=True):
        if not isinstance(features, Iterable) or isinstance(features, str):
            features = [features]

        column = list(range(self.__data.shape[1]))
        time_name = self.__header[self.time_column]

        for feature in features:
            if feature not in self.__header.keys():
                if error:
                    raise KeyError("The feature {} does not exist in the dataset!".format(feature))
            else:
                column.remove(self.__header[feature])
                if self.__header[feature] == self.time_column:
                    self.time_column = None

        new_header = [self.__header[i] for i in column]
        if time_name in new_header:
            self.time_column = new_header.index(time_name)

        self.__data = self.__data[:, column]
        self.set_header(new_header)

    # Can select one or more feature
    def select_feature(self, features, error=True):
        if not isinstance(features, Iterable) or isinstance(features, str):
            features = [features]

        column = []
        time_name = self.__header[self.time_column]

        for feature in features:
            if feature not in self.__header.keys():
                if error:
                    raise KeyError("The feature {} does not exist in the dataset!".format(feature))
            else:
                column.append(self.__header[feature])

        new_header = [self.__header[i] for i in column]
        if self.time_column not in new_header:
            self.time_column = None

        new_header = [self.__header[i] for i in column]
        if time_name in new_header:
            self.time_column = new_header.index(time_name)

        self.__data = self.__data[:, column]
        self.set_header(new_header)

    # data is a float matrix of data. All time value need to be changed to its timestamp (datetime.timestamp())
    # if no header line, assume all data have same order as before.
    def add_room(self, data, occupancy=None, room_name=None, header=True):
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
            room_name = len(self.__room)

        self.__room[len(self)] = str(room_name)
        self.__room[str(room_name)] = (self.__data.shape[0], self.__data.shape[0] + data.shape[0])

        if not self.__data.shape[0]:
            self.__data = data
            self.set_header(features)
            if len(occupancy.shape) == 1:
                occupancy.shape += (1,)
            self.__occupancy = occupancy
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

            new_data = full([data.shape[0], self.__data.shape[1] + len(rest_column)], nan)
            new_data[:, target_column] = data[:, source_column]
            new_data[:, self.__data.shape[1]:] = data[:, rest_column]

            self.__data = concatenate((self.__data,
                                       full([self.__data.shape[0], len(rest_column)], nan)), axis=1)
            self.__data = concatenate((self.__data, new_data), axis=0)

            new_header = []
            i = 0
            while self.__header.get(i, False):
                new_header.append(self.__header[i])
                i += 1

            for name in rest_column:
                new_header.append(name)

            self.set_header(new_header)

            if len(occupancy.shape) == 1:
                occupancy.shape += (1,)
            self.__occupancy = concatenate((self.__occupancy, occupancy), axis=0)

    def pop_room(self, room_name):
        if room_name not in self.__room.keys():
            raise KeyError("This dataset do not contain room {}".format(room_name))

        a, b = self.__room[room_name]
        pop_data, pop_occupancy = self[room_name]

        new_dataset = Dataset()
        new_dataset.add_room(pop_data, pop_occupancy, room_name=room_name, header=self.header)

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
            if self.__room[i] == room_name:
                found = True

            if found:
                new_a, new_b = self.__room[self.__room[i + 1]]
                self.__room[self.__room[i]] = (new_a - remove_col, new_b - remove_col)
                self.__room[i] = self.__room[i + 1]

        self.__room.pop(room_name)
        self.__room.pop(len(self) - 1)

        return new_dataset

    def split(self, percentage, random_sequence=False):
        front_dataset = Dataset()
        back_dataset = Dataset()
        front_dataset.time_column = self.time_column
        front_dataset.binary = self.binary
        front_dataset.labelled = self.labelled
        back_dataset.time_column = self.time_column
        back_dataset.binary = self.binary
        back_dataset.labelled = self.labelled


        split_point = round(percentage * self.__data.shape[0])

        for room in self.room:
            room_data, room_occupancy = self[room]
            if self.__room[room][1] <= split_point:
                front_dataset.add_room(room_data, room_occupancy, room_name=room, header=self.header)
            elif self.__room[room][0] > split_point:
                back_dataset.add_room(room_data, room_occupancy, room_name=room, header=self.header)
            else:
                start_pos = self.__room[room][0]
                mid_pos = split_point - start_pos
                front_room_data = room_data[:mid_pos, :]
                front_room_occupancy = room_occupancy[:mid_pos, :]
                back_room_data = room_data[mid_pos:, :]
                back_room_occupancy = room_occupancy[mid_pos:, :]
                front_dataset.add_room(front_room_data, front_room_occupancy,
                                       room_name="Partially " + str(room), header=self.header)
                back_dataset.add_room(back_room_data, back_room_occupancy,
                                      room_name="Partially " + str(room), header=self.header)

        return front_dataset, back_dataset

    def copy(self):
        duplicate = Dataset()
        duplicate.change_values(self.__data.copy())
        duplicate.change_occupancy(self.__occupancy.copy())
        # header: column, column: header
        duplicate.change_whole_header(self.__header.copy())
        # room: [start_row, end_row], room_counter: room
        duplicate.change_room_info(self.__room.copy())
        duplicate.time_column = self.time_column
        duplicate.binary = self.binary
        duplicate.labelled = self.labelled
        return duplicate

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
            return self.__data[a:b, :], self.__occupancy[a:b, :]

    def __getitem__(self, room_name):
        a, b = self.__room[room_name]
        return self.__data[a:b, :], self.__occupancy[a:b, :]

    def __len__(self):
        return len(self.__room) // 2

    def __add__(self, other):
        if not isinstance(other, Dataset):
            raise TypeError("Dataset need to add with Dataset")
        rooms = other.room
        header = other.header
        for room in rooms:
            data, occupancy = other[room]
            while room in self.__room.values():
                room = str(int(room) + 1)
            self.add_room(data, occupancy=occupancy, room_name=room, header=header)
        return self

    def __sub__(self, other):
        if not isinstance(other, Dataset):
            raise TypeError("Dataset need to sub with Dataset")
        rooms = other.room
        for room in rooms:
            if room in self.room:
                self.remove_room(room)
        return self

    def __str__(self):
        return str(self.__dict__)
