import pandas as pd
import time
import logging
import csv
import os
import numpy as np
from collections import defaultdict


class ParserModel:
    """
    This class is a model of a parser tree that build a feature vectors for train and test data
    based on a given train data
    """

    def __init__(self, directory, train_file_name, test_file_name, features_combination):
        """
        :param train_file_name: the name of the train file
        :param test_file_name: the name of the test file
        :param features_combination: is a list of the features we want to use
        """

        # Columns names of the data files
        self.file_columns_names = ['token_counter', 'token', '-_1', 'token_POS', '-_2', '-_3', 'token_head',
                                   'dependency_label', '-_4', '-_5']

        # features dict folder
        self.features_path_string = ''
        for feature in features_combination:
            self.features_path_string += feature + '_'
        self.dict_path = os.path.join(directory + 'dict/', self.features_path_string)

        self.feature_vec_len = 0
        """create train data and gold trees dict"""
        self.train_data = pd.read_table(train_file_name, header=None, names=self.file_columns_names)
        # add relevant columns:
        self.train_data = self.train_data.assign(sentence_index=0)
        self.train_data = self.train_data.assign(head_word='')
        self.train_data = self.train_data.assign(head_POS='')
        # will be in the format: {{token: {(sentence_index, token_counter), token)},
        #                          token_POS: {(sentence_index, token_counter), token_POS)}
        #                           we will use it for the head_word and head_POS assignment
        self.train_token_POS_dict = dict()
        # a dictionary where keys are the sentence index and the values are dictionaries where their keys are
        # the head node and the value ia a list of its target nodes, i.e all edges of the sentence
        # The format: {sentence_index: {source_node: [target_nodes]}}
        self.train_gold_tree = self.create_gold_tree_dictionary('train')

        """create test data and gold trees dict"""
        self.test_data = pd.read_table(test_file_name, header=None, names=self.file_columns_names)
        # add relevant columns:
        self.test_data = self.test_data.assign(sentence_index=0)
        self.test_data = self.test_data.assign(head_word='')
        self.test_data = self.test_data.assign(head_POS='')
        # will be in the format: {{token: {(sentence_index, token_counter), token)},
        #                          token_POS: {(sentence_index, token_counter), token_POS)}
        #                           we will use it for the head_word and head_POS assignment
        self.test_token_POS_dict = dict()
        # a dictionary where keys are the sentence index and the values are dictionaries where their keys are
        # the head node and the value ia a list of its target nodes, i.e all edges of the sentence
        # The format: {sentence_index: {source_node: [target_nodes]}}
        self.test_gold_tree = self.create_gold_tree_dictionary('test')

        self.features_combination = features_combination

        """Define relevant dictionaries"""
        # dictionary for each feature type, will include the number of instances from each feature
        self.feature_1 = {}
        self.feature_2 = {}
        self.feature_3 = {}
        self.feature_4 = {}
        self.feature_5 = {}
        self.feature_6 = {}
        self.feature_8 = {}
        self.feature_10 = {}
        self.feature_13 = {}

        self.features_dicts = {
            '1': [self.feature_1, 'p-word, p-pos'],
            '2': [self.feature_2, 'p-word'],
            '3': [self.feature_3, 'p-pos'],
            '4': [self.feature_4, 'c-word, c-pos'],
            '5': [self.feature_5, 'c-word'],
            '6': [self.feature_6, 'c-pos'],
            '8': [self.feature_8, 'p-pos, c-word, c-pos'],
            '10': [self.feature_10, 'p-word, p-pos, c-pos'],
            '13': [self.feature_13, 'p-pos, c-pos']
        }

        # the dictionary that will hold all indexes for all the instances of the features
        self.features_vector = {}

        # mainly for debugging and statistics
        self.features_vector_mapping = {}

        # feature vectors for the gold trees
        # The format is: {sentence_index: feature_vector}
        self.features_vector_train = defaultdict(list)
        self.features_vector_test = defaultdict(list)

        """Create the features vectors"""
        # build the type of features
        self.build_features_from_train()
        # build the features_vector
        self.build_features_vector()
        # build the feature vector for each tree gold of the train and the test data
        self.create_feature_vector('train')
        self.create_feature_vector('test')

    def create_gold_tree_dictionary(self, file):
        """
        This method create a dictionary with all the gold trees of a given file.
        :param file will be the file we want to create a dictionary based on it (train or test file)
        :return: a dictionary where keys are the sentence index and the values are dictionaries where their keys are
        the head node and the value ia a list of its target nodes, i.e all edges of the sentence
        """

        print('{}: Start building gold tree from {}'.format(time.asctime(time.localtime(time.time())), file))
        logging.info('{}: Start building gold tree from {}'.format(time.asctime(time.localtime(time.time())), file))

        if file == 'train':
            data = self.train_data
        elif file == 'test':
            data = self.test_data
        else:
            print('Data is not train and not test: cant create gold tree')
            return dict()

        gold_tree = dict()
        sentence_index = -1
        sentence_dict = dict()
        for index, row in data.iterrows():
            if row['token_counter'] == 1:
                # if this is the first word in the sentence: insert the list of the previous sen. to the gold tree dict
                if sentence_index == -1:  # if this is the first sentence - there is no previous sentence
                    sentence_index += 1
                else:  # if this is not the first sentence - insert the list of the previous sen. to the gold tree dict
                    gold_tree[sentence_index] = sentence_dict
                    sentence_index += 1
                    sentence_dict = dict()
            # add the edge: {head: target}
            if row['token_head'] in sentence_dict.keys():
                sentence_dict[row['token_head']].append(row['token_counter'])
            else:
                sentence_dict[row['token_head']] = [row['token_counter']]

            # update the sentence_index in the relevant data dataframe
            data.set_value(index, 'sentence_index', sentence_index)
        gold_tree[sentence_index] = sentence_dict
        sentence_index += 1

        # after we have the sentence index- we will create the data_token_pos_dict
        data_token_pos_dict = data[['token', 'token_POS', 'sentence_index', 'token_counter']]\
            .set_index(['sentence_index', 'token_counter']).to_dict()
        # add the root
        for root_index in range(sentence_index + 1):
            data_token_pos_dict['token'][(root_index, 0)] = 'root'
            data_token_pos_dict['token_POS'][(root_index, 0)] = 'root'

        if file == 'train':
            self.train_token_POS_dict = data_token_pos_dict
        elif file == 'test':
            self.test_token_POS_dict = data_token_pos_dict

        # add the head word and POS for each target
        for index, row in data.iterrows():
            token_head = row['token_head']
            curr_sentence_index = row['sentence_index']
            data.set_value(index, 'head_word', data_token_pos_dict['token'][(curr_sentence_index, token_head)])
            data.set_value(index, 'head_POS', data_token_pos_dict['token_POS'][(curr_sentence_index, token_head)])

        print('{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), file))
        logging.info('{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), file))

        return gold_tree

    def build_features_from_train(self):
        """
        This method build a features from the train data based on the features we want to use
        and update the features dictionaries
        :return:
        """
        for index, row in self.train_data.iterrows():
            # Define for features creation
            p_word = row['head_word']
            p_pos = row['head_POS']
            c_word = row['token']
            c_pos = row['token_POS']

            # build feature_1 of p-word, p-pos
            self.update_feature_dict('1', p_word=p_word, p_pos=p_pos)
            # build feature_2 of p-word
            self.update_feature_dict('2', p_word=p_word)
            # build feature_3 of p_pos
            self.update_feature_dict('3', p_pos=p_pos)
            # build feature_4 of c-word, c-pos
            self.update_feature_dict('4', c_word=c_word, c_pos=c_pos)
            # build feature_5 of c-word
            self.update_feature_dict('5', c_word=c_word)
            # build feature_6 of c-pos
            self.update_feature_dict('6', c_pos=c_pos)
            # build feature_8 of p-pos, c-word, c-pos
            self.update_feature_dict('8', p_pos=p_pos, c_word=c_word, c_pos=c_pos)
            # build feature_10 of p-word, p-pos, c-pos
            self.update_feature_dict('10', p_word=p_word, p_pos=p_pos, c_pos=c_pos)
            # build feature_13 of p-pos, c-pos
            self.update_feature_dict('13', p_pos=p_pos, c_pos=c_pos)

        # save all features dicts to csv
        for feature in self.features_dicts.keys():
            self.save_dictionary(feature)

        return

    def update_feature_dict(self, feature_number, p_word=None, p_pos=None, c_word=None, c_pos=None):
        """
        This method update the relevant feature dictionary
        :param feature_number: the number of the feature
        :param p_word: the word of the parent
        :param p_pos: the POS of the parent
        :param c_word: the word of the child
        :param c_pos: the POS of the child
        :return: no return, just update the object's features' dictionaries
        """

        # Create the list of relevant feature components
        option_for_features_list = [p_word, p_pos, c_word, c_pos]
        option_for_features_list = [x for x in option_for_features_list if x is not None]
        if feature_number in self.features_combination:
            # get relevant feature
            feature_dict = self.features_dicts[feature_number][0]
            # build feature name : key of the dictionary
            feature_key = 'f' + feature_number
            for option_feature in option_for_features_list:
                feature_key += '_' + option_feature

            if feature_key in feature_dict:
                feature_dict[feature_key] += 1
            else:
                feature_dict[feature_key] = 1

        return

    def save_dictionary(self, feature_number):
        """
        This method save the relevant feature dictionary
        :param feature_number: the number of the feature
        :return:
        """
        if feature_number in self.features_combination:
            w = csv.writer(open(self.dict_path + 'feature_' + feature_number + '.csv', 'w'))
            feature_dict = self.features_dicts[feature_number][0]
            for key, val in feature_dict.items():
                w.writerow([key, val])
            print('{}: Finished saving feature {}'.format(time.asctime(time.localtime(time.time())), feature_number))
            logging.info('{}: Finished saving feature {}'.format(time.asctime(time.localtime(time.time())),
                                                                 feature_number))

    def build_features_vector(self):
        """
        This method build a feature vector with all of the features we created based on the train data
        :return:
        """
        start_time = time.time()
        print('{}: Start building feature vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Start building feature vector'.format(time.asctime(time.localtime(time.time()))))

        features_vector_idx = 0

        for feature_number in self.features_combination:
            feature_instances = 0
            feature_dict = self.features_dicts[feature_number][0]
            feature_description = self.features_dicts[feature_number][1]
            # create first type of feature in features_vector which is word tag instances
            for feature_key in feature_dict.keys():
                self.features_vector[feature_key] = features_vector_idx
                self.features_vector_mapping[features_vector_idx] = feature_key
                features_vector_idx += 1
                feature_instances += 1
            print('{}: Size of feature_{} - {} instances is: {}'.format(time.asctime(time.localtime(time.time())),
                                                                        feature_number, feature_description,
                                                                        feature_instances))
            logging.info('{}: Size of feature_{} - {} instances is: {}'
                         .format(time.asctime(time.localtime(time.time())), feature_number, feature_description,
                                 feature_instances))

        print('{}: Finished building features vector in : {}'.format(time.asctime(time.localtime(time.time())),
              time.time() - start_time))
        logging.info('{}: Finished building features vector in : {}'.format(time.asctime(time.localtime(time.time())),
                     time.time() - start_time))

        print('{}: Saving dictionaries'.format(time.asctime(time.localtime(time.time()))))
        print('{}: Saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'features_vector' + '.csv', "w"))
        for key, val in self.features_vector.items():
            w.writerow([key, val])
        print('{}: Finished saving features_vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Finished saving features_vector'.format(time.asctime(time.localtime(time.time()))))

        print('{}: Saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        w = csv.writer(open(self.dict_path + 'features_vector_mapping' + '.csv', "w"))
        for key, val in self.features_vector_mapping.items():
            w.writerow([key, val])
        print('{}: finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))

        self.feature_vec_len = features_vector_idx

        return

    def calculate_local_feature_vec_per_feature(self, indexes_vector, feature_number, p_word=None, p_pos=None,
                                                c_word=None, c_pos=None):
        """
        This method create a feature vector per feature number for a given edge and a given feature number
        :param indexes_vector: the indexes_vector that we are calculating for the edge
        :param feature_number: the number of the feature we working on
        :param p_word: the word of the parent
        :param p_pos: the POS of the parent
        :param c_word: the word of the child
        :param c_pos: the POS of the child
        :return: no return, the indexes_vector is updated
        """
        option_for_features_list = [p_word, p_pos, c_word, c_pos]
        option_for_features_list = [x for x in option_for_features_list if x is not None]
        if feature_number in self.features_combination:
            # build feature
            feature_dict = self.features_dicts[feature_number][0]
            # build feature name
            feature_key = 'f' + feature_number
            for option_feature in option_for_features_list:
                feature_key += '_' + option_feature

            if feature_key in feature_dict:
                feature_idx = self.features_vector[feature_key]
                indexes_vector[feature_idx] = 1

        return

    def get_local_feature_vec(self, sentence_index, source, target, data):
        """
        This method create a feature vector for a given edge
        :param sentence_index: the index of the sentence
        :param source: the token_counter of the parent
        :param target: the token_counter of the child
        :param data: the data type: train or test
        :return: the vector feature of the given edge
        """

        indexes_vector = np.zeros(shape=self.feature_vec_len, dtype=int)

        if data == 'train':
            data_token_pos_dict = self.train_token_POS_dict
        elif data == 'test':
            data_token_pos_dict = self.test_token_POS_dict
        else:
            print('Data is not train and not test: cant create gold tree')
            return indexes_vector

        p_word = data_token_pos_dict['token'][(sentence_index, source)]
        p_pos = data_token_pos_dict['token_POS'][(sentence_index, source)]
        c_word = data_token_pos_dict['token'][(sentence_index, target)]
        c_pos = data_token_pos_dict['token_POS'][(sentence_index, target)]

        # build feature_1 of p-word, p-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '1', p_word=p_word, p_pos=p_pos)
        # build feature_2 of p-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '2', p_word=p_word)
        # build feature_3 of p_pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '3', p_pos=p_pos)
        # build feature_4 of c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '4', c_word=c_word, c_pos=c_pos)
        # build feature_5 of c-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '5', c_word=c_word)
        # build feature_6 of c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '6', c_pos=c_pos)
        # build feature_8 of p-pos, c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '8', p_pos=p_pos, c_word=c_word, c_pos=c_pos)
        # build feature_10 of p-word, p-pos, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '10', p_word=p_word, p_pos=p_pos, c_pos=c_pos)
        # build feature_13 of p-pos, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '13', p_pos=p_pos, c_pos=c_pos)

        return indexes_vector

    def create_global_feature_vector(self, tree, sentence_index, data):
        """
        create a global feature vector for a given tree
        :param tree: a dict of that maps all the edges in the tree in the format:
        {source_node: a list of its target nodes]}. For example: {1: [2], 2: [1, 3], 3: [1]}
        :param sentence_index: the number of the sentence in the train data
        :param data: the data type: train or test
        :return:
        """

        tree_indexes_vector = np.zeros(shape=self.feature_vec_len, dtype=int)
        for edge in tree.items():
            source = edge[0]
            target_nodes_list = edge[1]
            for target in target_nodes_list:
                edge_indexes_vector = self.get_local_feature_vec(sentence_index, source, target, data)
                tree_indexes_vector = np.add(edge_indexes_vector, tree_indexes_vector)

        return tree_indexes_vector

    def create_feature_vector(self, data):
        """
        create feature vectors for the tree gold of each of the sentences in the train data
        :param data: the data type: train or test
        :return: no return, just save the dictionary with the feature vector for each sentence
        """

        if data == 'train':
            features_vector = self.features_vector_train
            gold_tree = self.train_gold_tree

        elif data == 'test':
            features_vector = self.features_vector_test
            gold_tree = self.test_gold_tree
        else:
            print('Data is not train and not test: cant create gold tree')
            return defaultdict(list)

        start_time = time.time()
        print('{}: Starting building feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))
        logging.info('{}: Starting building feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))

        for sentence_index, sentence_tree in gold_tree.items():
            features_vector[sentence_index] = self.create_global_feature_vector(sentence_tree, sentence_index, data)

        print('{}: Finished building feature vectors {} in : {}'.
              format(time.asctime(time.localtime(time.time())), data, time.time() - start_time))
        logging.info('{}: Finished building feature vectors {} in : {}'.
                     format(time.asctime(time.localtime(time.time())), data, time.time() - start_time))

        print('{}: Saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))
        logging.info('{}: Saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))
        w = csv.writer(open(self.dict_path + 'features_vector_' + data + '.csv', "w"))
        for key, val in features_vector.items():
            w.writerow([key, val])

        print('{}: Finished saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))
        logging.info('{}: Finished saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), data))

        return


if __name__ == '__main__':
    all_start_time = time.time()
    curr_directory = '/Users/reutapel/Documents/Technion/Msc/NLP/hw2/NLP_HW2/'
    train_file = curr_directory + 'HW2-files/train.labeled'
    test_file = curr_directory + 'HW2-files/test.labeled'

    features = ['1', '2', '3', '4', '5', '6', '8', '10', '13']
    model_obj = ParserModel(curr_directory, train_file, test_file, features)

    run_time_cv = (time.time() - all_start_time) / 60.0
    print('{}: Finished all parser model creation in : {}'.
          format(time.asctime(time.localtime(time.time())), run_time_cv))
