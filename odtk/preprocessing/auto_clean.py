# The full preprocessing for the given odtk.data.dataset.Dataset()
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
#     target_frequency: sampling frequency that the dataset want to becomes
# Return:
#     No return


def auto_clean(dataset, target_frequency):
    from .downsample import downsample
    from .fill import fill
    from .ontology import ontology
    from .upsample import upsample
    from .outlier import remove_outlier
    from ..stats import frequency

    remove_outlier(dataset)
    overall_frequency = frequency(dataset, dataset_level=True)

    if overall_frequency > target_frequency:
        upsample(dataset, target_frequency)
    else:
        downsample(dataset, target_frequency)

    fill(dataset)

    ontology(dataset)
