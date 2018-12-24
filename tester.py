import odtk
import numpy as np

dataset = odtk.data.import_data("datatest.csv", time_column=1)
print(type(dataset))
# odtk.analyzer.analyze(dataset, 65, "result.txt")
odtk.modifier.downsample(dataset, 120)
odtk.modifier.change_to_binary(dataset)
# print(dataset)