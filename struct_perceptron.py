import numpy as np

from chu_liu import Digraph


# TODO - Understand when should the perceptron stop updating weights
class StructPerceptron:
    """
    this class implements the structured perceptron, calculates the best weights
    using Chu-Liu-Edmonds algorithm to find the Maximum Spanning Tree
    """
    def __init__(self, model, mode='train'):
        """
        :param model: an object of the model, which will create for a given edge its feature vector
        :type model: parser_model.ParserModel
        """
        # constant which represent the 'root' node in the data
        self._ROOT = 0
        self.model = model
        # number of features
        self.feature_vec_len = model.feature_vec_len
        self.train_gold_tree = model.train_gold_tree
        self.test_gold_tree = model.test_gold_tree
        self.comp_gold_tree = model.comp_gold_tree
        # the feature vector of a complete sentence in the training
        self.features_vector_train = model.features_vector_train
        self.weight_matrix = []
        self.current_weight_vec_iter = 0
        self.current_weight_vec = np.zeros(shape=self.feature_vec_len)
        self.current_sentence = 0
        # full graph contains a full graph + root per sentence {sentence_id: {parent_node: [child_nods]}}
        self.full_graph = {}
        self.sets_of_nodes = {}  # dict that contains set of nodes per sentence: {sentence_id: set_of_nodes}
        # mode of the class
        self._mode = mode
        self.gold_tree = None
        self.inference_mode(mode)

    def inference_mode(self, mode='train'):
        """
        based on whether we use this class for train or test,
        this method sets the gold tree source and the mode of using the model functions,
        and than creates a new full_graph dictionary out of the relevant gold tree

        :param str mode:
        indicates class mode:
          * train mode ('train')
          * test mode ('test')
          * competition mode ('comp')
        :return: None
        """
        if mode == 'train':
            self._mode = mode
            self.gold_tree = self.train_gold_tree
        elif mode == 'test':
            self._mode = mode
            self.gold_tree = self.test_gold_tree
        else:  # mode == 'comp'
            self._mode = mode
            self.gold_tree = self.comp_gold_tree
        self.sets_of_nodes = {} # clear the set of nodes
        self.full_graph = {}
        self.create_full_graph()

    def perceptron(self, num_of_iter):
        """
        this method implements the pseudo-code of the perceptron

        :param num_of_iter: N from the pseudo-code
        :return: the final weight vector
        :rtype: np.ndarray
        """
        for i in range(num_of_iter):
            for t in range(len(self.gold_tree)):
                self.current_sentence = t
                try:
                    pred_tree = self.calculate_mst(t)
                except AssertionError as err:
                    pred_tree = err.args
                    print("The algorithm returned a bad tree, update is skipped. \n tree: {}".format(pred_tree))
                else:
                    if not self.identical_dependency_tree(pred_tree, self.gold_tree[t]):
                        curr_feature_vec = self.features_vector_train[t]
                        new_feature_vec = self.model.create_global_feature_vector(pred_tree, t, mode=self._mode)
                        new_weight_vec = np.zeros(shape=self.feature_vec_len)
                        new_weight_vec = self.current_weight_vec + curr_feature_vec - new_feature_vec
                        self.weight_matrix.append(new_weight_vec)
                        self.current_weight_vec_iter += 1
                        self.current_weight_vec = new_weight_vec
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
        if self.current_sentence != t:
            self.current_sentence = t
        pred_tree = self.full_graph.get(t)
        digraph = Digraph(pred_tree, get_score=self.edge_score)
        new_graph = digraph.mst()
        pred_tree = new_graph.successors
        assert self.check_valid_tree(pred_tree, t), pred_tree
        return pred_tree

    def create_full_graph(self):
        """
        this method will create for a given gold dependency tree,
        a fully connected graph from each sentence of it

        :return: a fully connected tree
        """
        for idx, sentence in self.gold_tree.items():
            set_of_nodes = set()
            for source, targets in sentence.items():
                set_of_nodes.add(source)
                set_of_nodes.union(set(targets))
            if self._ROOT in set_of_nodes:
                set_of_nodes.remove(self._ROOT)
            self.sets_of_nodes.update({idx: set_of_nodes})
            graph = {}
            for node in set_of_nodes:
                targets = list(set_of_nodes.difference({node}))
                graph.update({node: targets})
            graph.update({self._ROOT: list(set_of_nodes)})
            self.full_graph.update({idx: graph})
        return

    def edge_score(self, source, target):
        """
        this method return a score of likelihood , for a pair of source and target
        s(source,target) = weight_vec * feature_vec(source, target)

        :param source: a source node
        :param target: a target node
        :return: score value
        """
        feature_vec = self.model.get_local_feature_vec(self.current_sentence, source, target, mode=self._mode)
        return self.current_weight_vec.dot(feature_vec)

    def identical_dependency_tree(self, pred_tree, gold_tree):
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

    def check_valid_tree(self, pred_tree, t):
        """
        check weather the tree returned from the Chu-Liu-Edmonds algorithm is a valid tree
        Valid tree means that each node have exactly one incoming edge, and the root is connected

        :param pred_tree: the predicted tree from the algorithm
        :param t: the sentence index
        :return: True if the tree is valid
        :rtype: bool
        """
        if self._mode != 'train' and len(pred_tree[self._ROOT]) != 1:
            return False
        set_of_nodes = self.sets_of_nodes[t]
        for node in set_of_nodes:
            if sum(node in targets for targets in pred_tree.values()) != 1:
                return False
        return True
