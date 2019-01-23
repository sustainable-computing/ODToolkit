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
#
# data = dict()
# data["datatest"] = odtk.data.load_sample("umons-datatest")
# source, target = data.split(0.8)

# models = odtk.model.NormalModel(data, data)
# models.get_all_model()
# print(models.models)

# all_metrics = odtk.evaluation.OccupancyEvaluation(data.occupancy, data.occupancy)
# all_metrics.get_all_metrics()
# print(all_metrics.metrics)

# print(odtk.easy_experiment(source, target, models=["RandomForest"], binary_evaluation=False))

# a6hWPkq5032WDXxVGbEN

# result = odtk.easy_set_experiment(data, models=["RNN", "NNv2"])
# pprint(result)
#
# a = odtk.evaluation.Result()
# a.set_result(result)
#
# odtk.plot.plot_one(a, dataset="datatest")

data = odtk.data.load_sample(["aifb-all"])
pprint(odtk.analyzer.analyze(data["aifb-all"], 60))
for name in data:
    data[name].remove_feature([data[name].header_info[data[name].time_column]])
odtk.plot.plot_feature(data["aifb-all"])
# data["sdu-all"].remove_feature([data["sdu-all"].header_info[data["sdu-all"].time_column]])
# data["sdu-binary"] = data["sdu-all"].copy()
# odtk.modifier.change_to_binary(data["sdu-binary"])
result = odtk.easy_set_experiment(data, models=["RandomForest"], split_percentage=0.8)
pprint(result)
# a = odtk.evaluation.Result()
# a.set_result(result)
# odtk.plot.plot_one(a, model="SVM")

# umons = odtk.data.load_sample("umons-all")
# odtk.plot.plot_feature(umons)
# odtk.plot.plot_feature(data["umons"])
# odtk.plot.plot_occupancy(data, total=False, binary=False)

a = odtk.evaluation.Result()
a.set_result(result)
odtk.plot.plot_one(a, model="RandomForest",
                   # legend=["Missrate", "Fallout"],
                   threshold=">0"
                   )
