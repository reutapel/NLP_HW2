import logging
import os
import numpy as np
import time
from chu_liu import Digraph
import pickle
from scipy.sparse import csr_matrix
from collections import defaultdict
from copy import copy


# TODO - Understand when should the perceptron stop updating weights
# todo: change to csr_matrix
# todo: take out the full_graph
class StructPerceptron:
    """
    this class implements the structured perceptron, calculates the best weights
    using Chu-Liu-Edmonds algorithm to find the Maximum Spanning Tree
    """

    def __init__(self, model, directory, mode='train'):
        """
        :param model: an object of the model, which will create for a given edge its feature vector
        :type model: parser_model.ParserModel
        :param str directory: the path to save files in, for the current run
        :param str mode: indicates class mode:
        * train mode ('train')
        * test mode ('test')
        * competition mode ('comp')
        """
        self.directory = os.path.join(directory, 'weights')
        # constant which represent the 'root' node in the data
        self._ROOT = 0
        self.model = model
        # number of features
        self.feature_vec_len = model.feature_vec_len
        self._global_gold_tree = model.gold_tree
        # the feature vector of a complete sentence in the training
        self.features_vector_train = model.gold_tree_features_vector['train']
        self.current_weight_vec_iter = 0
        # self.current_weight_vec = csr_matrix((1, self.feature_vec_len), dtype=int)
        self.current_weight_vec = np.zeros(self.feature_vec_len, dtype=int)
        self.current_sentence = 0
        self.scores = defaultdict(dict)    # type: defaultdict[int,dict[(int,int),int]]
        # full graph contains a full graph + root per sentence {sentence_id: {parent_node: [child_nods]}}
        self.full_graph = {}
        self.sets_of_nodes = {}  # dict that contains set of nodes per sentence: {sentence_id: set_of_nodes}
        # mode of the class
        self._mode = mode
        self.gold_tree = None
        self.inference_mode(mode)

    def inference_mode(self, mode='train', weight_vec=None):
        """
        based on whether we use this class for train or test,
        this method sets the gold tree source and the mode of using the model functions,
        and than creates a new full_graph dictionary out of the relevant gold tree
        if the mode of the class is not train, we need also to set the scores dict

        :param weight_vec:
        :param str mode: indicates class mode:
        * train mode ('train')
        * test mode ('test')
        * competition mode ('comp')
        :return: None
        """
        self._mode = mode
        self.gold_tree = self._global_gold_tree[mode]
        self.scores = defaultdict(dict)
        self.full_graph = self.model.full_graph[self._mode]
        # if the mode is 'test' or 'comp' we need a new scores dict
        if self._mode != 'train':
            if weight_vec is not None:
                self.current_weight_vec = copy(weight_vec)
            sentence_count = range(len(self.full_graph.keys()))
            for sentence_idx in sentence_count:
                self.calculate_new_scores(sentence_idx)

    def perceptron(self, num_of_iter):
        """
        this method implements the pseudo-code of the perceptron

        :param num_of_iter: N from the pseudo-code
        :return: the final weight vector
        :rtype: csr_matrix[int]
        """
        for i in range(num_of_iter):
            print('{}: Starting Iteration #{}'.format(time.asctime(time.localtime(time.time())), i + 1))
            logging.info('{}: Starting Iteration #{}'.format(time.asctime(time.localtime(time.time())), i + 1))
            for t in range(len(self.gold_tree)):
                self.calculate_new_scores(t)
                if t % 100 == 0:
                    print('{}: Working on sentence #{}'.format(time.asctime(time.localtime(time.time())), t + 1))
                    logging.info('{}: Working on sentence #{}'.format(time.asctime(time.localtime(time.time())), t + 1))
                self.current_sentence = t
                pred_tree = self.calculate_mst(t)
                if not GraphUtil.identical_dependency_tree(pred_tree, self.gold_tree[t]):
                    curr_feature_vec = self.features_vector_train[t]
                    new_feature_vec = self.model.create_global_feature_vector(pred_tree, t, mode=self._mode)
                    new_weight_vec = self.current_weight_vec + curr_feature_vec - new_feature_vec
                    self.current_weight_vec_iter += 1
                    self.current_weight_vec = new_weight_vec
            if i+1 in [20, 50, 80, 100]:
                with open(os.path.join(self.directory, 'final_weight_vec_{}.pkl'.format(i + 1)), 'wb') as f:
                    pickle.dump(self.current_weight_vec, f)
        print("{}: the number of weight updates in this training:{}".format(time.asctime(time.localtime(time.time()))
                                                                            , self.current_weight_vec_iter))
        logging.info("{}: the number of weight updates in this training:{}"
                     .format(time.asctime(time.localtime(time.time())), self.current_weight_vec_iter))
        with open(os.path.join(self.directory, 'final_weight_vec_{}.pkl'.format(num_of_iter)), 'wb') as f:
            pickle.dump(self.current_weight_vec, f)
        return self.current_weight_vec

    def calculate_mst(self, t):
        """
        this method calculates the Max Spanning Tree using the chu-liu implementation,
        using a full graph of the sentence.

        :param int t: the index of the sentence
        :return: A predicted tree from the algorithm
        :rtype: dict[int, List[int]]
        :raise AssertionError: with argument of the defected predicated tree
        """
        if t % 100 == 0:
            print('{}: Start calculating mst for sentence #{}, on {} mode'
                  .format(time.asctime(time.localtime(time.time())), t + 1, self._mode))
            logging.info('{}: Start calculating mst for sentence #{}, on {} mode'
                         .format(time.asctime(time.localtime(time.time())), t + 1, self._mode))
        if self.current_sentence != t:
            self.current_sentence = t
        full_tree = self.full_graph[t]
        digraph = Digraph(full_tree, get_score=self.edge_score)
        new_graph = digraph.mst()
        pred_tree = new_graph.successors
        if t % 100 == 0:
            print('{}: Finished calculating mst for sentence #{}, on {} mode'
                  .format(time.asctime(time.localtime(time.time())), t + 1, self._mode))
            logging.info('{}: Finished calculating mst for sentence #{}, on {} mode'
                         .format(time.asctime(time.localtime(time.time())), t + 1, self._mode))
        return pred_tree

    def edge_score(self, source, target):
        """
        this method return a score of likelihood , for a pair of source and target
        s[sentence_index][(source,target)]

        :param source: a source node
        :param target: a target node
        :return: score value
        """
        return self.scores[self.current_sentence][(source, target)]

    def calculate_new_scores(self, sentence_idx):
        """
        this method update self.scores dict with the scores of likelihood for each edge in each sentence

        :cvar self.scores[sentence_index][(source, target)]: = feature_vec[sentence_index](source, target)*weight_vec^T
        :return: None
        """

        feature_vecs = self.model.full_graph_features_vector[self._mode]
        for edge, features_indexes in feature_vecs[sentence_idx].items():
            relevant_weight = [self.current_weight_vec[value] for value in features_indexes]
            self.scores[sentence_idx][edge] = int(sum(relevant_weight))

    def check_valid_tree(self, pred_tree, t):
        """
        check weather the tree returned from the Chu-Liu-Edmonds algorithm is a valid tree
        Valid tree means that each node have exactly one incoming edge, and the root is connected

        :param pred_tree: the predicted tree from the algorithm
        :param t: the sentence index
        :return: True if the tree is valid
        :rtype: bool
        """
        set_of_nodes = self.sets_of_nodes[t]
        for node in set_of_nodes:
            if sum(node in targets for targets in pred_tree.values()) != 1:
                return False
        return True


class GraphUtil:
    _ROOT = 0

    @staticmethod
    def create_full_graph(gold_tree, edges_existed_on_train=None, pos_edges_existed_on_train=None, pos_dict=None):
        """
        this method will create for a given gold dependency tree,
        a fully connected graph from each sentence of it

        :param dict[(int,int),str] pos_dict: the part of speech of given word in position (sentence_index,word_index)
        :param dict[int,list[int]] edges_existed_on_train: dictionary of edges found in train data
        :param dict[str,list[str]] pos_edges_existed_on_train: dict of Part Of Speech edges existed on the train data
        :param gold_tree: the original gold_tree of the sentence
        :type gold_tree: dict[int,dict[int,list[int]]]
        :return: set of nodes per graph and full graph
        :rtype: (dict[int,set[int]], dict[int,dict[int,list[int]]])
        """
        sets_of_nodes = defaultdict(set)
        full_graph = defaultdict(dict)
        for idx, sentence in gold_tree.items():
            set_of_nodes = set()
            for source, targets in sentence.items():
                set_of_nodes.add(source)
                set_of_nodes = set_of_nodes.union(set(targets))
            if GraphUtil._ROOT in set_of_nodes:
                set_of_nodes.remove(GraphUtil._ROOT)
            sets_of_nodes[idx] = set_of_nodes
            graph = defaultdict(list)
            pos_per_sentence = None
            if pos_edges_existed_on_train is not None:
                pos_per_sentence = GraphUtil.get_valid_pos_edges(idx, pos_dict, pos_edges_existed_on_train)
            for node in set_of_nodes:
                targets = set_of_nodes.difference({node})
                if edges_existed_on_train is not None:
                    targets = targets.intersection(set(edges_existed_on_train[node]))
                if pos_per_sentence is not None:
                    targets = targets.intersection(set(pos_per_sentence[node]))
                targets = list(targets)
                graph[node] = targets
            graph[GraphUtil._ROOT] = list(set_of_nodes)
            full_graph[idx] = graph
        return sets_of_nodes, full_graph

    @staticmethod
    def get_valid_pos_edges(sentence_index, pos_dict, pos_edges_existed_on_train):
        """
        this method calculates, for a sentence, set of possible edges based on the POS edges from the train data

        :param int sentence_index: the sentence index
        :param dict[(int,int),str] pos_dict: the part of speech of given word in position (sentence_index,word_index)
        :param pos_edges_existed_on_train: dictionary of Part Of Speech edges existed on the train data
        :type pos_edges_existed_on_train: dict[str,list[str]]
        :return: dictionary of possible edges, based on POS edges from the train
        :rtype: defaultdict[str,list[int]]
        """
        # create tuples of (POS, word_index) for the sentence
        pos_sentence_tuples = [(pos, w_id) for (s_id, w_id), pos in pos_dict.items() if sentence_index == s_id]
        # for each POS, the indexes of words with such POS
        pos_index_mapping = defaultdict(list)  # type: defaultdict[str, list[int]]
        for pos, w_id in pos_sentence_tuples:
            pos_index_mapping[pos].append(w_id)
        # for each POS, the list of targets, based on the edges from the train data
        pos_targets = defaultdict(set)  # type: defaultdict[str, set[int]]
        for pos in pos_index_mapping.keys():
            for target in pos_edges_existed_on_train[pos]:
                if target in pos_index_mapping:
                    pos_targets[pos] = pos_targets[pos].union(set(pos_index_mapping[target]))
        # map between the POS indexes location to the list of targets for such POS
        pos_per_sentence = defaultdict(list)  # type: defaultdict[str, list[int]]
        for pos in pos_targets.keys():
            for source in pos_index_mapping[pos]:
                pos_per_sentence[source] = list(pos_targets[pos].difference({source}))
        return pos_per_sentence

    @staticmethod
    def identical_dependency_tree(pred_tree, gold_tree):
        """
        this method evaluate whether two dependency trees are identical

        :param pred_tree: the predicted tree from the algorithm
        :param gold_tree: the gold labeled tree (graph with the correct edges)
        :return:
        """
        if set(gold_tree.keys()) != set(pred_tree.keys()):
            return False
        for gold_source, gold_targets in gold_tree.items():
            pred_source = gold_source
            pred_targets = pred_tree[pred_source]
            if set(pred_targets) != set(gold_targets):
                return False
        return True
