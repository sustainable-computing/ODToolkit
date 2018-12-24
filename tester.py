import odtk

dataset = odtk.data.import_data("datatest.csv", time_column=1)
print(type(dataset))
odtk.modifier.regulate(dataset)
# print(dataset)
print(dataset.header)