from collections import Counter
from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence
from Chu_Liu_Edmonds_algorithm import *
import random
import numpy as np
from tqdm import tqdm
from nltk.corpus import dependency_treebank

class _Counter(Counter):
    def __mul__(self, other):
        if type(other) is _Counter:
            return _Counter({k: self[k] * other[k] for k in self.keys() & other.keys()})
        elif type(other) is float or type(other) is int:
            return _Counter({k: self[k]*other for k in self.keys()})

    def __rmul__(self, other):
        return self.__mul__(other)


class MSTParser:

    class Arc:
        """ Represent an edge between 2 words """
        def __init__(self, head, tail, hword, htag, tword, ttag, weight=1):
            self.head = head
            self.tail = tail
            self.weight = weight
            self.hword = hword
            self.htag = htag
            self.tword = tword
            self.ttag = ttag

        def __eq__(self, other):
            return self.head == other.head and self.tail == other.tail

    def __init__(self):
        self.weights = _Counter()
        self.words, self.tags = set(), set()

    def init_w_t(self, X):
        """
        Initiliaze the Words vocabulary set and the Tags set
        :param X: list of nltk DependencyGraph
        """
        self.words.add("ROOT")
        for sent in X:
            for n in range(len(sent.nodes)):
                node = sent.nodes[n]
                if node["address"] != 0:
                    self.words.add(node["word"])
                    self.tags.add(node["tag"])

    def _get_all_arcs(self, sent):
        """
        get all possible arc of a given sentence
        :param sent: nltk DependencyGraph
        :return: an array of Arc object
        """
        arcs = []
        for add1 in range(len(sent.nodes)):
            for add2 in range(1, len(sent.nodes)):
                if add1 != add2:
                    if add1 == 0:
                        hword, htag = "ROOT", "ROOT"
                    else:
                        hword, htag = sent.nodes[add1]["word"], sent.nodes[add1]["tag"]

                    arc = self.Arc(add1, add2, hword, htag, sent.nodes[add2]["word"], sent.nodes[add2]["tag"])
                    arc.weight = self._get_arc_score(arc)
                    arcs.append(arc)
        return arcs

    def _get_arc_score(self, arc: Arc):
        """
        get the score of a given arc (edge) The score is the dot product between the arc feature function and the weights
        :param arc: An Arc object
        :return: score
        """
        mul = self._get_arc_features(arc) * self.weights
        return -sum(mul.values())

    def _get_gold_standard_graph(self, sent):
        """
        return an array of Arc object which represent the gold standard graph
        :param sent: nltk Dependency Graph
        :return: array of Arc object
        """
        nodelist = list(range(1, len(sent.nodes)))
        edgelist = [self.Arc(h, t, sent.nodes[h]["word"], sent.nodes[h]["tag"], sent.nodes[t]["word"],
                             sent.nodes[t]["tag"]) for h in nodelist for t in sent.nodes[h]["deps"][""]]
        edgelist.append(self.Arc(0, sent.root["address"], "ROOT", "ROOT", sent.root["word"], sent.root["tag"]))
        return edgelist

    def _get_arc_features(self, arc: Arc):
        """
        2 types of features: words bigram and PosTag
        each feature which appear in the dictionary are the feature of the given arc
        :param arc: Arc object
        :return: _Counter dictionnary with features name as key and 1 as values
        """
        feat1 = arc.hword + arc.tword
        feat2 = arc.htag + arc.ttag
        return _Counter({feat1: 1, feat2: 1})


    def _get_graph_features(self, arcs):
        """
        return the sum
        :param sent:
        :return:
        """
        f = _Counter()
        for a in arcs:
            f += self._get_arc_features(a)
        return f

    def _min_spanning_arborescence_nx(self, arcs, sink):
        """
        Wrapper for the networkX min_spanning_tree to follow the original API
        :param arcs: list of Arc tuples
        :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
         the algorithm.
        """
        G = DiGraph()
        for arc in arcs:
            G.add_edge(arc.head, arc.tail, weight=arc.weight)
        ARB = minimum_spanning_arborescence(G)
        # result = {}
        result = []
        headtail2arc = {(a.head, a.tail): a for a in arcs}
        for edge in ARB.edges:
            # tail = edge[1]
            # result[tail] = headtail2arc[(edge[0], edge[1])]
            result.append(headtail2arc[(edge[0], edge[1])])
        return result

    def _perceptron(self, X, lr=1, n_iter=2):
        for i in range(n_iter):
            random.shuffle(X)
            for sent in tqdm(X):
                arcs_mst = self._min_spanning_arborescence_nx(self._get_all_arcs(sent), None)
                diff = self._get_graph_features(self._get_gold_standard_graph(sent))
                diff.subtract(self._get_graph_features(arcs_mst))
                self.weights.update(diff)

        self.weights = self.weights * (1/(n_iter * len(X)))

    def fit(self, X, lr=1, n_iter=2):
        """
        fit the model to the given trainning set using perceptron algorithm
        :param X: list of nltk DependencyGraph
        :param lr: learning rate
        :param n_iter: number of iteration
        """
        self.init_w_t(X)
        self._perceptron(X, lr, n_iter)

    def predict(self, X):
        """
        predict Dependency tree for all sentence in x
        :param X: list of Dependency graph from nltk
        :return: list of list of arcs
        """
        all_arcs = []
        for sent in X:
            arcs_mst = min_spanning_arborescence_nx(self._get_all_arcs(sent), None)
            all_arcs.append(arcs_mst)
        return all_arcs

    def _accuracy(self, pred, gt):
        """
        :param pred: prediction tree - list of arc object
        :param gt: ground truth tree - list of arcs object
        :return: the accuracy
        """

        correct_edges = 0
        for arc_pred in pred:
            for gt_arc in gt:
                if arc_pred == gt_arc:
                    correct_edges += 1
        return correct_edges/len(gt)

    def score(self, X):
        """
        :param X: list of nltk dependency Graph
        :return: the mean accuracy of the model
        """
        predict = self.predict(X)
        gt = [self._get_gold_standard_graph(sent) for sent in X]
        all_accuracy = []
        for arcs_pred, arcs_gt in zip(predict, gt):
            all_accuracy.append(self._accuracy(arcs_pred, arcs_gt))
        return np.mean(all_accuracy)

