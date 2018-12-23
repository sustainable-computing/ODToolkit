from pickle import load


def read(file_name):
    with open(file_name, 'rb') as file:
        dataset = load(file)
    return dataset
