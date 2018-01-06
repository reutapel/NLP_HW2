import numpy as np

from chu_liu import Digraph

# TODO - Understand when should the perceptron stop updating weights
class StructPerceptron:
    """
    this class implements the structured perceptron, calculates the best weights
    using Chu-Liu-Edmonds algorithm to find the Maximum Spanning Tree
    """
    # todo: set a score function to send to chu-liu
    # todo: implement the pseudo-code
    # todo: change 'root' tag to 0
    # todo: try and catch
    # todo: extract to external function chu-liu uses (lines 39:46), calculate_mst()
    def __init__(self, gold_tree, model):
        """
        :param model: an object of the model, which will create for a given edge its feature vector
        :param gold_tree: a representation of the gold dependency tree, which is represented by a list of tuples
        in a form of (x_i,y_i) where x_i is the head node and y_i the target node
        """
        self.model = model
        self.feature_vec_len = model.feature_vec_len
        self.gold_tree = model.train_gold_tree
        self.features_vector_train = model.features_vector_train
        self.weight_matrix = []
        self.current_weight_vec_iter = 0
        self.current_weight_vec = np.empty(self.feature_vec_len)
        self.full_graph = {}
        self.current_sentence = 0

    def perceptron(self, num_of_iter):
        """
        this method implements the pseudo-code of the perceptron
        :param num_of_iter: N from the pseudo-code
        :return:
        """
        for i in range(num_of_iter):
            for t in range(len(self.gold_tree)):
                self.current_sentence = t
                pred_tree = self.full_graph.get(t)
                if pred_tree is None:
                    self.create_full_graph()
                    pred_tree = self.full_graph.get(t)
                digraph = Digraph(pred_tree, get_score=self.edge_score)
                new_graph = digraph.mst()
                pred_tree = new_graph.successors
                assert self.check_valid_tree(pred_tree, t)
                if not self.identical_dependency_tree(pred_tree, self.gold_tree[t]):
                    # todo: collab with Reut on the exact functions
                    curr_feature_vec = self.features_vector_train[t]  # todo: Reut changed it to the correct dictionary
                    new_feature_vec = self.model.create_global_feature_vector(pred_tree, t, 'train')
                    new_weight_vec = np.empty(self.feature_vec_len)  # todo: check if this is faster
                    new_weight_vec = self.current_weight_vec + curr_feature_vec - new_feature_vec
                    self.weight_matrix.append(new_weight_vec)
                    self.current_weight_vec_iter += 1
                    self.current_weight_vec = new_weight_vec
        return self.current_weight_vec

    def create_full_graph(self):
        """
        this method will create for a given gold dependency tree,
        a fully connected graph from each sentence of it
        :return: a fully connected tree
        """
        for idx, sentence in enumerate(self.gold_tree.values()):
            set_of_nodes = set()
            for source, targets in sentence.items():
                set_of_nodes.add(source)
                set_of_nodes.union(set(targets))
            if 'root' in set_of_nodes:
                set_of_nodes.remove('root')
            graph = {}
            for node in set_of_nodes:
                targets = list(set_of_nodes.difference({node}))
                graph.update({node: targets})
            graph.update({'root': list(set_of_nodes)})
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
        feature_vec = self.model.get_local_feature_vec(self.current_sentence, source, target, 'train')
        return self.current_weight_vec.dot(feature_vec)

    def identical_dependency_tree(self, pred_tree, gold_tree):
        """
        this method evaluate whether two dependency trees are identical
        :param pred_tree:
        :param gold_tree:
        :return:
        """
        if set(gold_tree.keys()) != set(pred_tree.keys()):
            return False
        for gold_source, gold_tragets in gold_tree.items():
            pred_source = gold_source
            pred_targets = pred_tree[pred_source]
            if set(pred_targets) != set(gold_tragets):
                    return False
        return True

    def check_valid_tree(self, pred_tree, t):
        gold_tree = self.gold_tree[t]
        if len(pred_tree['root']) != 1:
            return False
        set_of_nodes = set()
        for source, targets in gold_tree.items():
            set_of_nodes.add(source)
            set_of_nodes.union(set(targets))
        if 'root' in set_of_nodes:
            set_of_nodes.remove('root')
        for node in set_of_nodes:
            if sum(node in targets for targets in pred_tree.values()) != 1:
                return False
        return True
