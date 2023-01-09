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
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
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


def get_weight_features(node1, node2, words_dict, tags_dict):
    word_feature, tag_feature = None, None

    if node1["word"] in words_dict and node2["word"] in words_dict:
        word_feature = words_dict[node1["word"]].index * len(words_dict) + words_dict[node2["word"]].index

    elif node1["tag"] in tags_dict and node2["tag"] in tags_dict:
        tag_feature = tags_dict[node1["tag"]].index * len(tags_dict) + tags_dict[node2["tag"]].index + pow(len(words_dict), 2)

    return word_feature, tag_feature


def get_score(node1, node2, words_dict, tags_dict, weights):
    word_feature, tag_feature = get_weight_features(node1, node2, words_dict, tags_dict)

    if not word_feature or not tag_feature:
        return 0

    return -(weights[word_feature] + weights[tag_feature])


def get_arcs(tree, words_dict, tags_dict, weights):
    arcs = []
    for tail in tree.nodes.values():
        if tail["word"] in words_dict:
            for head in tree.nodes.values():
                if head["word"] in words_dict and head["tag"] != TOP_TAG and not are_same_node(head, tail):
                    arcs.append(Arc(head["address"], tail["address"], get_score(head, tail, words_dict, tags_dict, weights)))
    return arcs


def get_Tree_Vector(tree, words_dict, tags_dict, vector_len):
    tree_vector = np.zeros(vector_len)

    for node in tree.nodes.values():
        if node["address"]:
            word_index, tag_index = get_weight_features(tree.nodes[node['head']], node, words_dict, tags_dict)
            if word_index and tag_index:
                tree_vector[word_index] += 1
                tree_vector[tag_index] += 1
    return tree_vector


def get_MST_vector(tree, arcs_MST, words_dict, tags_dict, vector_len):
    MST_vector = np.zeros(vector_len)

    for arc in arcs_MST.values():
        word_index, tag_index = get_weight_features(tree.nodes[arc.head], tree.nodes[arc.tail], words_dict, tags_dict)
        if word_index and tag_index:
            MST_vector[word_index] += 1
            MST_vector[tag_index] += 1

    return MST_vector


def get_tree_weight(tree, words_dict, tags_dict, weights, learning_rate):
    arcs = get_arcs(tree, words_dict, tags_dict, weights)
    arcs_MST = min_spanning_arborescence_nx(arcs, None)
    vector_len = pow(len(words_dict), 2) + pow(len(tags_dict), 2)
    tree_vector = get_Tree_Vector(tree, words_dict, tags_dict, vector_len)
    MST_vector = get_MST_vector(tree, arcs_MST, words_dict, tags_dict, vector_len)
    return tree_vector - MST_vector * learning_rate


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


def get_dependency_tree(test_set, words_dict, tags_dict, perceptron_result):
    return [min_spanning_arborescence_nx(get_arcs(tree, words_dict, tags_dict, perceptron_result), 0).values() for tree in test_set]


def get_likeliness(arcs, tree):
    num_commons = 0
    for node in tree.nodes.values():
        for child in node['deps']['']:
            parent = node['address']
            if (parent, child) in arcs:
                num_commons += 1
    return num_commons


def get_number_shared_edges_bw(test_set, words_dict, tags_dict, perceptron_result):
    number_shared_edges = 0
    for tree in test_set:
        arcs = get_dependency_tree(test_set, words_dict, tags_dict, perceptron_result)
        number_shared_edges += (get_likeliness(arcs, tree) / (len(tree.nodes) - 1))
    return number_shared_edges


def evaluate(test_set, words_dict, tags_dict, perceptron_result):
    number_shared_edges = get_number_shared_edges_bw(test_set, words_dict, tags_dict, perceptron_result)
    return number_shared_edges / len(test_set)




if __name__ == '__main__':
    training_set, test_set = load_data()
    words_dict, tags_dict = init_words_and_tags(training_set)

    perceptron_result = perceptron(training_set, words_dict, tags_dict, 2, 1)
    score = evaluate(test_set, words_dict, tags_dict, perceptron_result)
    print('SCORE: ', score)