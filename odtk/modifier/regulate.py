from odtk.data.dataset import Dataset
from Levenshtein import jaro


# Parameters:
#   dataset: odtk.data.dataset.Dataset() or header list
# Return:
#   Edited header list
def regulate(dataset):
    dictionary = {"co2": ["co2", "carbon dioxide"],
                  "humidity": ["humidity", "humidness", "wetness", "moisture"],
                  "temperature": ["temp", "temperature", "indoor temperature", "environment temperature"],
                  "out-temperature": ["outside temperature", "outside temp", "outdoor temp", "outdoor temperature"],
                  "damper": ["damper", "damper position"],
                  "voc": ["voc", "volative organic compounds"],
                  "air": ["air", "air velocity", "wind"],
                  "cloud": ["cloud", "cloud coverage", "cloud ratio"]}

    print(dictionary.keys())

    header = dataset
    return_list = True
    if isinstance(dataset, Dataset):
        header = dataset.header
        return_list = False
    if not isinstance(header, list):
        raise TypeError("Cannot recognize the header")

    new_header = list()

    for word in header:
        maximum_score = 0
        similar_word = ''
        for target_word in dictionary.keys():
            target_word_score = 0
            for possible_word in dictionary[target_word]:
                score = jaro(possible_word, word.lower())
                if target_word_score < score:
                    target_word_score = score
            if target_word_score > maximum_score:
                maximum_score = target_word_score
                similar_word = target_word

        if maximum_score > 0.9:
            new_header.append(similar_word)
        else:
            new_header.append(word)

    if not return_list:
        dataset.set_header(new_header)
    return new_header
