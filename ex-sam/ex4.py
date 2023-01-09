import numpy as np
from nltk.corpus import dependency_treebank

from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx


class Arc:
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight


def getData():
    """ Loads the data. 90% training set, 10% test set. """

    data = dependency_treebank.parsed_sents()

    return data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]


def getWT(data):
    """
    Gets the words and tags.

    :param data: The data.
    :return: The words and tags.
    """

    return {n["word"]
            for t in data
            for n in t.nodes.values()}, \
           {n["tag"]
            for t in data
            for n in t.nodes.values()}


def getArcWeightIndex(h, t, wIndex, tIndex=None):
    """
    Gets the arc weight index, if in the train set.

    :param h: The head.
    :param t: The tail.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :return: The arc weight index.
    """

    if tIndex is None and h in wIndex and t in wIndex:
        return wIndex[h] * len(wIndex) + wIndex[t]

    if tIndex is not None and h in tIndex and t in tIndex:
        return len(wIndex) ** 2 + tIndex[h] * len(tIndex) + tIndex[t]

    else:
        return False


def getFeatureVec(n1, n2, wDic, tDic):
    """
    Gets the feature vector.

    :param n1: Node 1.
    :param n2: Node 2.
    :param wDic: Words dictionary.
    :param tDic: Tags dictionary.
    :return: The feature vector.
    """

    featVec = np.zeros(len(wDic) ** 2 + len(tDic) ** 2)

    if n1['word'] in wDic.keys() and n2 in wDic.keys():
        featVec[wDic[n1['word']] * len(wDic) + wDic[n2['word']]] = 1

    if n1['tag'] in tDic.keys() and n2 in tDic.keys():
        featVec[len(wDic) ** 2 + wDic[n1['tag']] * len(tDic) +
                wDic[n2['tag']]] = 1

    return featVec


def getScore(h, t, weight, wIndex, tIndex):
    """
    Gets score.

    :param h: The head.
    :param t: The tail.
    :param weight: The weight.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :return: The score.
    """

    index1 = getArcWeightIndex(h['word'], t['word'], wIndex)
    index2 = getArcWeightIndex(h['tag'], t['tag'], wIndex, tIndex)

    if index1 is False or index2 is False:
        return 0

    else:
        return weight[index1] + weight[index2]


def checkNodes(wIndex, tIndex, arcs, n1, values, weight):
    """
    Checks the current nodes.

    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :param arcs: The tree arcs.
    :param n1: The node.
    :param values: The values,
    :param weight: The weight.
    """
    for n2 in values:
        word = n2['word']
        address1 = n1['address']
        address2 = n2['address']
        tag = n2['tag']
        if word in wIndex and address1 != address2 and tag != 'TOP':

            arc = Arc(address2, address1,
                      -getScore(n2, n1, weight, wIndex, tIndex))
            arcs.append(arc)


def computeTreeArcs(tree, weight, wIndex, tIndex):
    """
    Computes the tree arcs.

    :param tree: The tree.
    :param weight: The weight.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :return: The tree arcs.
    """

    treeArcs = list()
    values = tree.nodes.values()

    for node in values:
        word = node['word']
        if word in wIndex:
            checkNodes(wIndex, tIndex, treeArcs, node, values, weight)

    return treeArcs


def getTreeVector(tree, wIndex, tIndex):
    """
    Get the corresponding vector for the tree.

    :param tree: The tree.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :return: The corresponding vector for the tree.
    """

    length = len(wIndex) ** 2 + len(tIndex) ** 2
    treeVec = np.zeros(length)
    values = tree.nodes.values()

    for value in values:
        if value['address']:
            wordIndex = getArcWeightIndex(tree.nodes[value['head']]['word'],
                                          value['word'], wIndex)
            tagIndex = getArcWeightIndex(tree.nodes[value['head']]['tag'],
                                         value['tag'], wIndex, tIndex)

            if wordIndex is False or tagIndex is False:
                continue

            else:
                treeVec[wordIndex] += 1
                treeVec[tagIndex] += 1

    return treeVec


def getMSTVector(t, arcs, wIndex, tIndex):
    """
    Gets the MST Vectors.

    :param t: The tree.
    :param arcs: The MST arcs.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :return: The MST vector using the Chui Lui Edmond algorithm
    """

    length = len(wIndex) ** 2 + len(tIndex) ** 2
    MSTVector = np.zeros(length)
    arcsValues = arcs.values()

    for arc in arcsValues:
        wordIndex = getArcWeightIndex(t.nodes[arc.head]['word'],
                                      t.nodes[arc.tail]['word'], wIndex)
        tagIndex = getArcWeightIndex(t.nodes[arc.head]['tag'],
                                     t.nodes[arc.tail]['tag'], wIndex, tIndex)

        if wordIndex is False or tagIndex is False:
            continue

        else:
            MSTVector[wordIndex] += 1
            MSTVector[tagIndex] += 1

    return MSTVector


def computeWeights(tree, wIndex, tIndex, learningRate, weights):
    arcs = min_spanning_arborescence_nx(
        computeTreeArcs(tree, weights, wIndex, tIndex), 0)

    return (getTreeVector(tree, wIndex, tIndex) -
            getMSTVector(tree, arcs, wIndex, tIndex)) * learningRate


def computeIteration(trainingSet, wIndex, tIndex, learningRate, weights,
                     resultWeights):
    np.random.shuffle(trainingSet)
    for tree in trainingSet:
        weights += computeWeights(tree, wIndex, tIndex, learningRate, weights)
        resultWeights += weights


def computePerceptron(trainingSet, wIndex, tIndex, iterationsNum=2,
                      learningRate=1):
    """
    Uses the Chui Lui Edmond algorithm to train the model and computes the
    averaged perceptron algorithm.

    :param trainingSet: The training set.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :param iterationsNum: Number of iterations.
    :param learningRate: The learning rate.
    :return:The averaged perceptron result.
    """

    length = len(wIndex) ** 2 + len(tIndex) ** 2

    weights, resultWeights = np.zeros(length), np.zeros(length)

    i = 0
    while i < iterationsNum:
        computeIteration(trainingSet, wIndex, tIndex, learningRate, weights,
                         resultWeights)

        i += 1

    return resultWeights / (len(trainingSet) * iterationsNum)


def computeScoreResult(testingSet, wIndex, tIndex, weight):
    """
    Computes the attachment score for the learned w, averaged over all
    sentences in the test set.

    :param testingSet: The testing set.
    :param wIndex: The words to their indexes.
    :param tIndex: The tags to their indexes.
    :param weight: the weight.
    :return: The attachment score for the learned w, averaged over all
    sentences in the test set.
    """

    total = 0

    for t in testingSet:
        arcSet = {(arc.head, arc.tail)
                  for arc in
                  min_spanning_arborescence_nx(computeTreeArcs(t, weight,
                                                               wIndex, tIndex),
                                               0).values()}

        total += getSimilitude(arcSet, t) / (len(t.nodes) - 1)

    return total / len(testingSet)


def getSimilitude(arcSet, t):
    similitude = 0
    values = t.nodes.values()

    for node in values:
        deps = node['deps']['']

        for childIndex in deps:
            address = node['address']

            if (address, childIndex) in arcSet:
                similitude += 1

    return similitude


if __name__ == '__main__':
    train, test = getData()
    words, tags = getWT(train)

    word_to_index = {word: i for i, word in enumerate(words)}
    tag_to_index = {tag: i for i, tag in enumerate(tags)}
    weight = computePerceptron(train, word_to_index, tag_to_index,
                               iterationsNum=2, learningRate=1)

    acc = computeScoreResult(test, word_to_index, tag_to_index, weight)
    print(acc)
