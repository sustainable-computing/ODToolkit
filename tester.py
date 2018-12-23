import odtk

dataset = odtk.data.import_data("datatest.csv", time_column=1)
print(type(dataset))
odtk.analyzer.analyze(dataset, 65, "result.txt")