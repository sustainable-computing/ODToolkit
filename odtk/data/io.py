# Save a odtk.data.dataset.Dataset object to local disk
#
# Parameters:
#     dataset: odtk.data.dataset.Dataset()
#     file_name: The name of target file
# Return:
#     No return


def save_dataset(dataset, file_name):
    from pickle import dump

    with open(file_name, 'wb') as file:
        dump(dataset, file)


# Load a odtk.data.dataset.Dataset object from local disk
#
# Parameters:
#     file_name: The name of target file
# Return:
#     odtk.data.dataset.Dataset


def read_dataset(file_name):
    from pickle import load

    with open(file_name, 'rb') as file:
        dataset = load(file)
    return dataset
