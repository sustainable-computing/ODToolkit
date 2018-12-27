import odtk
import numpy as np

# dataset = odtk.data.import_data("datatest.csv", time_column=1)
# print(type(dataset))
# odtk.modifier.regulate(dataset)
# # print(dataset)
# print(dataset.header)

truth = np.array([1, 2, 3, 4, 5, 6, 0, 0, 9])
truth.shape += (1,)
estimation = np.array([1, 1, 1, 1, 5, 5, 0, 5, 10])
estimation.shape += (1,)


print(np.concatenate((truth, estimation), axis=1))
print(odtk.analyzer.mae(truth, estimation))
