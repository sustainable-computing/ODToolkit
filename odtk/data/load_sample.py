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

    if sample_name[0] == "sdu":
        if sample_name[1] == "508":
            return read("./odtk/data/sample_csv/sdu/508")
        elif sample_name[1] == "511":
            return read("./odtk/data/sample_csv/sdu/511")
        elif sample_name[1] == "601":
            return read("./odtk/data/sample_csv/sdu/601")
        elif sample_name[1] == "604":
            return read("./odtk/data/sample_csv/sdu/604")
        elif sample_name[1] == "all":
            return read("./odtk/data/sample_csv/sdu/sdu")

    if sample_name[0] == "aifb":
        return read("./odtk/data/sample_csv/aifb/aifb")

    print("Data not defined")
