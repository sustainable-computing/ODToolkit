Please go to https://odtoolkit.github.io for documentation

# ODTK: Occupancy Detection Toolkit

Occupancy Detection Toolkit

Required package
---
pandas (python3 -m pip install pandas)

numpy (python3 -m pip install numpy)

Levenshtein (python3 -m pip install python-levenshtein)

odtk.data
---
1. odtk.data.import_data(file_name, time_column=None, mode='csv', header=True)
*// This will return class odtk.data.dataset.Dataset*
2. **odtk.data.load_sample() # Not finish**
3. odtk.data.read(file_name)
4. odtk.data.save_dataset(dataset, file_name)

odtk.analyzer
---
1. analyze(dataset, threshold, save_file, print_out=False)
2. dropout_rate(dataset, total=False)
3. frequency(dataset, total=True)
4. gap_detect(dataset, threshold, detail=False)
5. occupancy_evaluation(dataset, total=True)
6. uptime(dataset, frequency, gaps=None)

odtk.modifier
---
1. auto_clean(dataset, target_frequency)
2. change_to_one_hot(dataset)
3. change_to_label(dataset)
4. change_to_binary(dataset)
5. clean(dataset, auto_fill=True)
6. downsample(dataset, target_frequency, algorithm="mean")
7. fill(dataset)
8. regulate(dataset or list)
9. upsample(dataset, target_frequency, algorithm="linear")

odtk.analyzer.evaluation (or in short odtk.analyzer)
---
1. f_score(truth, estimation, tolerance=0, mode="f1-score")
2. rmse(truth, estimation)
3. nrmse(truth, estimation)
4. mape(truth, estimation)
5. mase(truth, estimation)
6. mae(truth, estimation)

odtk.model
---
1. support_vector_machine()
2. random_forest()

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
