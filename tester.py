import odtk

dataset = odtk.data.import_data("datatest.csv", time_column=1)
print(type(dataset))
print(odtk.analyzer.occupancy_evaluation(dataset, total=True))