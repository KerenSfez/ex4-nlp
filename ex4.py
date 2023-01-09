from nltk.corpus import dependency_treebank
import numpy as np
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

TOP_TAG = "TOP"


class Item:
    def __init__(self, content, index, num_occurrence):
        self.content = content
        self.index = index
        self.num_occurrence = num_occurrence

    def add_occurrence(self):
        self.num_occurrence += 1


class Arc:
    def __init__(self, start_node, end_node, weight):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

    def get_word(self, node):
        return node["word"]

    def get_tag(self, node):
        return node["tag"]


def load_data():
    data = dependency_treebank.parsed_sents()
    return data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]


def init_words_and_tags(training_set):
    words, tags = dict(), dict()
    for tree in training_set:
        for node in tree.nodes.values():
            if node["word"] not in words:
                words[node["word"]] = Item(node["word"], node["address"], 1)
            else:
                words[node["word"]].add_occurrence()

            if node["tag"] not in tags:
                tags[node["tag"]] = Item(node["tag"], node["address"], 1)
            else:
                tags[node["tag"]].add_occurrence()
    return words, tags

def are_same_node(node1, node2):
    return node1["address"] == node2["address"]


def get_arcs(tree, words_dict, tags_dict):
    arcs = []
    for end_node in tree.nodes.values:
        if end_node["word"] in words_dict:
            for start_node in tree.nodes.values:
                if start_node["word"] in words_dict and start_node["tag"] != TOP_TAG and not are_same_node(start_node, end_node):
                    new_arc = Arc(start_node["address"], end_node["address"], get_score())


def get_tree_weight(tree, words_dict, tags_dict, weights, learning_rate):
    arcs = get_arcs(tree, words_dict, tags_dict)
    arcs_MST = min_spanning_arborescence_nx(arcs, None)


def get_total_weights(words_dict, tags_dict, weights, learning_rate):
    for tree in training_set:
        weights += get_tree_weight(tree, words_dict, tags_dict, weights, learning_rate)
    return weights

def perceptron_per_iteration(training_set, words_dict, tags_dict, weights, learning_rate):
    np.random.shuffle(training_set)
    return get_total_weights(words_dict, tags_dict, weights, learning_rate)


def perceptron(training_set, words_dict, tags_dict, num_iteration=2, learning_rate=1):
    training_set_len = len(training_set)
    perceptron_len = pow(len(words_dict), 2) + pow(len(tags_dict), 2)
    weights = np.zeros(perceptron_len)

    for i in range(num_iteration):
        weights = perceptron_per_iteration(training_set, words_dict, tags_dict, weights, learning_rate)

    return weights / (training_set_len * num_iteration)



if __name__ == '__main__':
    training_set, test_set = load_data()
    words_dict, tags_dict = init_words_and_tags(training_set)
    perceptron(training_set, words_dict, tags_dict, 2, 1)
