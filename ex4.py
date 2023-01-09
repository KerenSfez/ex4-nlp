from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx


training_set = []
test_set = []


def load_data():
    data = dependency_treebank.parsed_sents()
    return data[:int(0.9*len(data))], data[int(0.9*len(data)):]


