# ODTK: Occupancy Detection Toolkit

Occupancy Detection Toolkit

odtk.data
---
1. odtk.data.import_data(file_name, time_column=None, mode='csv', header=True)
*// This will return class odtk.data.dataset.Dataset*
2. **odtk.data.load_sample() # Not finish**
3. odtk.data.read(file_name)
4. odtk.data.write(dataset, file_name)

odtk.analyzer
---
1. odtk.analyzer.analyze(dataset, threshold, save_file, print_out=False)
2. odtk.analyzer.dropout_rate(dataset, total=False)
3. odtk.analyzer.frequency(dataset, total=True)
4. odtk.analyzer.gap_detect(dataset, threshold, detail=False)
5. odtk.analyzer.occupancy_evaluation(dataset, total=True)
6. odtk.analyzer.uptime(dataset, frequency, gaps=None)

odtk.modifier
---
1. odtk.modifier.auto_clean(dataset, target_frequency)
2. odtk.modifier.change_to_one_hot(dataset)
3. odtk.modifier.change_to_label(dataset)
4. odtk.modifier.change_to_binary(dataset)
5. odtk.modifier.clean(dataset, auto_fill=True)
6. odtk.modifier.downsample(dataset, target_frequency, algorithm="mean")
7. odtk.modifier.fill(dataset)
8. **odtk.modifier.regulate() # Not finish**
9. odtk.modifier.upsample(dataset, target_frequency, algorithm="linear")

## odtk classes - odtk.data.dataset.Dataset
```python
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
```
