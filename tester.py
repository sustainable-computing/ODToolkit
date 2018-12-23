import odtk

dataset = odtk.data.import_data("datatest.csv", time_column=1)
print(type(dataset))
print(odtk.analyzer.gap_detect(dataset, 65, detail=False))
print(odtk.analyzer.uptime(dataset, 65))