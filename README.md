# ODTK: Occupancy Detection Toolkit
Occupancy Detection Toolkit

// Return class odtk.data.dataset.Dataset
odtk.data.import_data(file_name, time_column=None, mode='csv', header=True)
odtk.data.load_sample() # Not finish
odtk.data.read(file_name)
odtk.data.write(dataset, file_name)
odtk.analyzer.analyze(dataset, threshold, save_file, print_out=False)
odtk.analyzer.dropout_rate(dataset, total=False)
odtk.analyzer.frequency(dataset, total=True)
odtk.analyzer.gap_detect(dataset, threshold, detail=False)
odtk.analyzer.occupancy_evaluation(dataset, total=True)
odtk.analyzer.uptime(dataset, frequency, gaps=None)
odtk.modifier.auto_clean(dataset, target_frequency)
odtk.modifier.change_to_one_hot(dataset)
odtk.modifier.change_to_label(dataset)
odtk.modifier.change_to_binary(dataset)
odtk.modifier.clean(dataset, auto_fill=True)
odtk.modifier.downsample(dataset, target_frequency, algorithm="mean")
odtk.modifier.fill(dataset)
odtk.modifier.regulate() # Not finish
odtk.modifier.upsample(dataset, target_frequency, algorithm="linear")

odtk.data.dataset.Dataset:
    Protected instance:
        __data      -> numpy.ndarray
        __occupancy -> numpy.ndarray
        __header    -> dictionary -> header: column, column: header
        __room      -> dictionary -> room: [start_row, end_row], room_counter: room
        time_column -> int
        binary      -> bool
        labelled    -> bool
    property:
        data
        occupancy
        header
        header_info
        room
        room_info
    methods:
        change_values(self, data)
        change_occupancy(self, occupancy)
        change_room_info(self, room)
        change_header(self, old, new)
        add_room(self, data, occupancy=None, room_name=None, header=True)
        remove_feature(self, features, error=True)
        remove_room(self, room_name)
        set_header(self, header)
    rewrite methods:
        __add__(self, other)
        __sub__(self, other)
        __getitem__(self, room_name)
        __iter__(self)
        __len__(self)
        __next__(self)
        __str__(self)