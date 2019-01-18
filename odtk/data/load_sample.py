from odtk.data.read import read


def load_sample(sample_name):
    sample_name = sample_name.split('-')
    if sample_name[0] == "umons":
        if sample_name[1] == "datatest":
            return read("./odtk/data/sample_csv/umons/datatest")
        elif sample_name[1] == "datatest2":
            return read("./odtk/data/sample_csv/umons/datatest2")
        elif sample_name[1] == "datatraining":
            return read("./odtk/data/sample_csv/umons/datatraining")
        elif sample_name[1] == "all":
            return read("./odtk/data/sample_csv/umons/umons")

    print("Data not defined")
