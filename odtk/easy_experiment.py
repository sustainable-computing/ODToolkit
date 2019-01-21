from odtk.model.superclass import *
from odtk.evaluation.superclass import *


def easy_experiment(source,
                    target_test,
                    target_retrain=None,
                    domain_adaptive=False,
                    models="all",
                    binary_evaluation=True,
                    evaluation_metrics="all",
                    thread_num=4):
    if domain_adaptive and target_retrain is None:
        raise ValueError("Domain Adaptive model must have target_retrain dataset")

    if domain_adaptive:
        model = DomainAdaptiveModel(source, target_retrain, target_test, thread_num=thread_num)
    else:
        model = NormalModel(source, target_test, thread_num=thread_num)

    if model == "all":
        model.get_all_model()
    else:
        model.add_model(models)

    results = model.run_all_model()

    total_result = dict()

    for model_result in results.keys():
        total_result[model_result] = dict()

        if binary_evaluation:
            metrics = BinaryEvaluation(results[model_result], target_test.occupancy)
        else:
            metrics = OccupancyEvaluation(results[model_result], target_test.occupancy)

        if evaluation_metrics == "all":
            metrics.get_all_metrics()
        else:
            metrics.add_metrics(evaluation_metrics)

        total_result[model_result] = metrics.run_all_metrics()

    return total_result
