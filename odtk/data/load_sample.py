# Load a odtk.data.dataset.Dataset object from the sample folder
#
# Parameters:
#     sample_name: Name of the sample. Use '-' represent folder relation.
# Return:
#     odtk.data.dataset.Dataset


def load_sample(sample_name):
    from .io import read
    from os.path import abspath, join, basename, isfile
    from os import listdir

    if isinstance(sample_name, str):
        sample_dir = join(abspath(__file__).rstrip(basename(__file__)), "sample_csv")
        if sample_name == "all":
            all_data = dict()
            for name in listdir(sample_dir):
                if isfile(join(sample_dir, name)):
                    all_data[name] = read(join(sample_dir, name))
                else:
                    all_data[name] = read(join(sample_dir, name, "all"))
            return all_data
        sample_name = sample_name.split('-')
        directory = join(sample_dir, *sample_name)
        if isfile(directory):
            return read(directory)
        else:
            raise FileNotFoundError("Dataset not found in built-in library")

    result = {}
    for required_name in sample_name:
        result[required_name] = load_sample(required_name)
    return result
