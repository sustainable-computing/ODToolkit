import odtk
import numpy as np
from tqdm import tqdm
from pprint import pprint
# for i in tqdm(range(1, 25)):
#     odtk.data.write(
#         odtk.data.import_data("./odtk/data/sample_csv/lbl/data" + str(i) + ".csv", 0, room_name="data_" + str(i)),
#         "data" + str(i))

# all = odtk.data.load_sample("lbl-data1")
# for i in tqdm(range(2, 25)):
#     all += odtk.data.load_sample("lbl-data" + str(i))
# odtk.data.write(all, "all")

# odtk.analyzer.analyze(odtk.data.load_sample("lbl-all"), 61*15, save_file="result.txt")

data = dict()
data["datatest"] = odtk.data.load_sample("umons-datatest")
data["datatest2"] = odtk.data.load_sample("umons-datatest2")
data["datatraining"] = odtk.data.load_sample("umons-datatraining")
# source, target = data.split(0.8)

# models = odtk.model.NormalModel(data, data)
# models.get_all_model()
# print(models.models)

# all_metrics = odtk.evaluation.OccupancyEvaluation(data.occupancy, data.occupancy)
# all_metrics.get_all_metrics()
# print(all_metrics.metrics)

# print(odtk.easy_experiment(source, target, models=["RandomForest"], binary_evaluation=False))


pprint(odtk.easy_set_experiment(data, models=["RandomForest"]))

for i in range(0):
    umons = odtk.data.load_sample("umons-all")
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()

    print("Full")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["Light"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("No light")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["CO2"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("No CO2")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["Humidity"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("No Humidity")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["Temperature"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("No Temperature")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["CO2"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("No HumidityRatio")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["Light", "Humidity", "Temperature", "HumidityRatio"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("Only CO2")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))

    umons = odtk.data.load_sample("umons-all")
    umons.remove_feature(["CO2", "Humidity", "Temperature", "HumidityRatio"])
    train, test = umons.split(0.8)
    result = odtk.model.random_forest.RandomForest(train, test).run()
    print("Only Light")
    print(odtk.analyzer.evaluation.rmse(test.occupancy, result))
    print(odtk.analyzer.evaluation.mase(test.occupancy, result))
    print(odtk.analyzer.evaluation.f_score(test.occupancy, result))
