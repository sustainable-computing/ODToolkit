# Load raw data from disk
#
# Parameters:
#     file_name: The name of the raw data file
#     time_column: The column index for the timestamp in given raw data file
#     mode: The raw data format, support 'csv'
#     header: Indicate whether the raw data include header at the first line
#     room_name: The name of room for the raw data file
#     tz: The time zone offset that need to fix for the raw data file
# Return:
#     odtk.data.dataset.Dataset object contains one room data


def import_data(file_name, time_column=None, mode='csv', header=True, room_name=None, tz=0):
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
                    if i == time_column:
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
        dataset.set_header(feature_name)
        dataset.time_column = time_column
        return dataset
