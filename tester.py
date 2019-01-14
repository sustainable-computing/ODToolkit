import odtk
import numpy as np

# dataset = odtk.data.import_data("datatest.csv", time_column=1)
# print(type(dataset))
# odtk.modifier.regulate(dataset)
# # print(dataset)
# print(dataset.header)

all = odtk.model.NormalModel(1, 2)
all.models["Test3"].b = 7
all.run_all_model()
