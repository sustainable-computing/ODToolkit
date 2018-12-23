from pickle import dump


def write(dataset, file_name):
    with open(file_name, 'wb') as file:
        dump(dataset, file)
