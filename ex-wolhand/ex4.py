import numpy as np
from nltk.corpus import dependency_treebank

import Chu_Liu_Edmonds_algorithm


class Perceptron:
    def __init__(self, length, words, tags, training_set, learning_rate, num_iteration):
        self.length = length
        self.weight = np.zeros(length)
        self.averaged_weight = np.zeros(length)
        self.words_index = words
        self.tags_index = tags
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def iteration(self):
        counter = 0
        np.random.shuffle(self.training_set)
        for nodes in training_set:
            print('Tree :', counter)
            self.weight += self.calculate_weight(nodes)
            self.averaged_weight += self.weight
            counter += 1

    def MST_vec(self,tree,arcs):
        vec=np.zeros(self.length)
        for arc in arcs.values():
           vec = self.add_to_vector(vec,tree.nodes[arc.head]['word'],
                                      tree.nodes[arc.tail]['word'],tree.nodes[arc.head]['tag'],
                                      tree.nodes[arc.tail]['tag'])
        return vec

    def tree_vec(self,tree):
        vec = np.zeros(self.length)
        for i in tree.nodes.values():
            if i["address"]:
                vec = self.add_to_vector(vec,tree.nodes[i['head']]['word'],
                                          i['word'],tree.nodes[i['head']]['tag'],
                                         i['tag'])
        return vec

    def add_to_vector(self,vector, w1, w2, t1, t2):
        index_word = self.arc_word_index_weight(w1, w2)
        index_tag = self.arc_tag_index_weight(t1, t2)
        if index_word is not None and index_tag is not None:
            vector[index_word] += 1
            vector[index_tag] += 1
        return vector

    def calculate_weight(self, tree):
        result_arc = Chu_Liu_Edmonds_algorithm.min_spanning_arborescence_nx(self.treeArcs(tree), 0)
        final_res = self.tree_vec(tree) - self.MST_vec(tree, result_arc)
        return final_res * self.learning_rate

    def calculate_arc_weight_sum(self, node1, node2):
        word_arc_index = self.arc_word_index_weight(node1['word'], node2['word'])
        tag_arc_index = self.arc_tag_index_weight(node1['tag'], node2['tag'])
        if word_arc_index is None or tag_arc_index is None:
            return 0
        return self.weight[word_arc_index] + self.weight[tag_arc_index]

    def arc_word_index_weight(self, word1, word2):
        if self.tags_index is None and word1 not in self.words_index or word2 not in self.words_index:
            return None
        return self.words_index[word1] * len(self.words_index) + self.words_index[word2]

    def arc_tag_index_weight(self, tag1, tag2):
        if self.tags_index != None and tag1 not in self.tags_index or tag2 not in self.tags_index:
            return None
        else:
            return len(self.words_index) ** 2 + self.tags_index[tag1] * len(self.tags_index.keys()) +\
                   self.tags_index[tag2]

    def treeArcs(self, tree):
        arcs = []
        for node in tree.nodes.values():
            if node['word'] in self.words_index:
                for node2 in tree.nodes.values():
                    if node2["tag"] != 'TOP' and node2['word'] in self.words_index \
                            and node['address'] != node2['address']:
                        arcs.append(Arc(node2['address'], node['address'], -self.calculate_arc_weight_sum(node2, node)))
        return arcs

    def result(self):
        self.weight = self.averaged_weight / (len(self.training_set) * self.num_iteration)
        return self.weight


class Arc:
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight


def load_parsed_sentences():
    data = dependency_treebank.parsed_sents()
    ninety_percent = int(len(data) * 0.9)
    test_set = data[ninety_percent:]
    training_set = data[:ninety_percent]
    return training_set, test_set


def dictionary_index(dict):
    return_dict = {}
    for index, word in enumerate(dict):
        return_dict[word] = index
    return return_dict


def get_dict_word_and_tag(training_set):
    dict_words = {}
    dict_tags = {}
    for graph in training_set:
        for n in graph.nodes.values():
            if n["word"] not in dict_words:
                dict_words[n["word"]] = 0
            dict_words[n["word"]] += 1
            if n["tag"] not in dict_tags:
                dict_tags[n["tag"]] = 0
            dict_tags[n["tag"]] += 1
    dict_words_index = dictionary_index(dict_words)
    dict_tags_index = dictionary_index(dict_tags)
    return dict_words, dict_words_index, dict_tags, dict_tags_index


def perceptron_algorithm(training_set, dict_words_index, dict_tags_index, num_iterations, learning_rate, length):
    perceptron = Perceptron(length, dict_words_index, dict_tags_index, training_set, learning_rate, num_iterations)
    for index in range(num_iterations):
        perceptron.iteration()
        print("Iteration : ",index)
    return perceptron.result(),perceptron


def compute_on_test(test_set, perceptron):
    result = 0
    for tree in test_set:
        arcs = set()
        for arc in Chu_Liu_Edmonds_algorithm.min_spanning_arborescence_nx(perceptron.treeArcs(tree),0).values():
            arcs.add((arc.head, arc.tail))
        result += find_equivalences(arcs,tree) / (len(tree.nodes)-1)
    return result / len(test_set)

def find_equivalences(arc_set,tree):
    equivalence = 0
    for node in tree.nodes.values():
        equivalence += sum(1 for index in node["deps"][""] if (node["address"], index) in arc_set)
    return equivalence


if __name__ == '__main__':
    training_set, test_set = load_parsed_sentences()
    dict_words, dict_words_index, dict_tags, dict_tags_index = get_dict_word_and_tag(training_set)
    len_for_perceptron = len(dict_tags.keys()) ** 2 + len(dict_words.keys()) ** 2
    perceptron_result, perceptron = perceptron_algorithm(training_set, dict_words_index, dict_tags_index, 2, 1,
                                                        len_for_perceptron)
    accuracy = compute_on_test(test_set, perceptron)
    print(accuracy)
