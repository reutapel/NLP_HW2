
class StructPerceptron:
    """
    this class implements the structured perceptron, calculates the best weights
    using Chu-Liu-Edmonds algorithm to find the Maximum Spanning Tree
    """
    # todo: set a score function to send to chu-liu
    # todo: implement the pseudo-code

    def __init__(self, gold_tree, model):
        """
        :param model: an object of the model, which will create for a given edge its feature vector
        :param gold_tree: a representation of the gold dependency tree, which is represented by a list of tuples
        in a form of (x_i,y_i) where x_i is the head node and y_i the target node
        """
        self.model = model
        self.gold_tree = gold_tree
        self.current_weight_vec = None

    def perceptron(self, num_of_iter):
        """
        this method implements the pseudo-code of the perceptron
        :param num_of_iter: N from the pseudo-code
        :return:
        """
        pass

    def create_full_graph(self):
        """
        this method will create for a given sentence,
        a fully connected tree (plus root) from it
        :return: a fully connected tree
        """
        pass

    def edge_score(self, source, target):
        """
        this method return a score of likelihood , for a pair of source and target
        s(source,target) = weight_vec * feature_vec(source, target)
        :param source: a source node
        :param target: a target node
        :return: score value
        """
        pass


