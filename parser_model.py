import pandas as pd
import time
import logging
import csv
import os
import numpy as np
from collections import defaultdict
from copy import copy
from struct_perceptron import GraphUtil


class ParserModel:
    """
    This class is a model of a parser tree that build a feature vectors for train and test data
    based on a given train data
    """

    def __init__(self, directory, train_file_name, test_file_name, comp_file_name, features_combination,
                 use_edges_existed_on_train=True, use_pos_edges_existed_on_train=True, train_index=None,
                 test_index=None):
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
        self.dict_path = os.path.join(directory, 'dict', self.features_path_string)

        self.feature_vec_len = 0
        # declare gold tree dict:
        # for each mode: a dictionary where keys are the sentence index and the values are dictionaries where their
        # keys are the head node and the value ia a list of its target nodes, i.e all edges of the sentence
        # The format: {{mode: sentence_index: {source_node: [target_nodes]}}}
        self.gold_tree = {'train': defaultdict(list),
                          'test': defaultdict(list),
                          'comp': defaultdict(list)}

        # will be in the format: {mode: {token: {(sentence_index, token_counter), token)},
        #                          token_POS: {(sentence_index, token_counter), token_POS)}}
        #                           we will use it for the head_word and head_POS assignment
        self.token_POS_dict = {'train': defaultdict(list),
                               'test': defaultdict(list),
                               'comp': defaultdict(list)}

        # Dictionary of edges found in train data - the indexes of the nodes
        # The format will be: {source: list of targets}
        self.edges_existed_on_train = None
        self.use_edges_existed_on_train = use_edges_existed_on_train

        # Dictionary of edges found in train data - the POS of the nodes
        # The format will be: {source POS: list of targets POS}
        self.pos_edges_existed_on_train = None
        self.use_pos_edges_existed_on_train = use_pos_edges_existed_on_train

        """create train data and gold trees dict"""
        self.train_data = pd.read_table(train_file_name, header=None, names=self.file_columns_names)
        # add relevant columns:
        self.train_data = self.train_data.assign(sentence_index=0)
        self.train_data = self.train_data.assign(head_word='')
        self.train_data = self.train_data.assign(head_POS='')
        self.create_gold_tree_dictionary('train')
        if train_index is not None:  # if we use CV
            # choose the relevant sentences
            self.test_data = self.train_data.loc[self.train_data['sentence_index'].isin(test_index)]
            self.train_data = self.train_data.loc[self.train_data['sentence_index'].isin(train_index)]
            # modify the token_POS_dict and gold_tree to the new train data
            self.gold_tree['train'] = {key: self.gold_tree['train'][key] for key in train_index}
            # self.token_POS_dict['train'] = {key: self.token_POS_dict['train'][key] for key in train_index}

        self.create_edges_existed_on_train()

        """create test data and gold trees dict"""
        if train_index is None:  # if no CV
            self.test_data = pd.read_table(test_file_name, header=None, names=self.file_columns_names)
            # add relevant columns:
            self.test_data = self.test_data.assign(sentence_index=0)
            self.test_data = self.test_data.assign(head_word='')
            self.test_data = self.test_data.assign(head_POS='')
            # self.create_gold_tree_dictionary('test')

        """create comp data and gold trees dict"""
        self.comp_data = pd.read_table(comp_file_name, header=None, names=self.file_columns_names)
        self.comp_data = self.comp_data.assign(sentence_index=0)
        # self.create_gold_tree_dictionary('comp')

        self.features_combination = features_combination

        """Define relevant dictionaries"""
        # dictionary for each feature type, will include the number of instances from each feature
        self.feature_1 = dict()
        self.feature_2 = dict()
        self.feature_3 = dict()
        self.feature_4 = dict()
        self.feature_5 = dict()
        self.feature_6 = dict()
        self.feature_7 = dict()
        self.feature_8 = dict()
        self.feature_9 = dict()
        self.feature_10 = dict()
        self.feature_11 = dict()
        self.feature_12 = dict()
        self.feature_13 = dict()
        self.feature_14 = dict()
        self.feature_15 = dict()
        self.feature_16 = dict()
        self.feature_17 = dict()
        self.feature_18 = dict()
        self.feature_19 = dict()
        self.feature_20 = dict()
        self.feature_21 = dict()
        self.feature_22 = dict()
        self.feature_23 = dict()
        self.feature_24 = dict()
        self.feature_25 = dict()
        self.feature_26 = dict()
        self.feature_27 = dict()
        self.feature_28 = dict()
        self.feature_29 = dict()
        self.feature_30 = dict()
        self.feature_31 = dict()

        # a dictionary with the dictionary and the description of each feature
        self.features_dicts = self.define_features_dicts()

        # the dictionary that will hold all indexes for all the instances of the features
        self.features_vector = {}

        # mainly for debugging and statistics
        self.features_vector_mapping = {}

        # feature vectors for the gold trees
        # The format is: {mode: {sentence_index: feature_vector}}
        self.gold_tree_features_vector = {'train': defaultdict(list),
                                          'test': defaultdict(list)}

        # feature vectors for the full graphs
        # The format is: {mode: {sentence_index: {(head, target): feature_vector}}}
        self.full_graph_features_vector = {'train': defaultdict(dict),
                                           'test': defaultdict(dict),
                                           'comp': defaultdict(dict)}

        # full graph of the train
        self.full_graph = {'train': defaultdict(dict),
                           'test': defaultdict(dict),
                           'comp': defaultdict(dict)}

        # create object of the GraphUtils
        self.graph_utils = GraphUtil()

        """Create the features vectors"""
        # build the type of features
        self.build_features_from_train()
        # build the features_vector
        self.build_features_vector()
        # build the feature vector for each tree gold of the train and the test data
        self.create_gold_tree_feature_vector('train')
        # self.create_gold_tree_feature_vector('test')
        # build the feature vectors for each full graph in test and train
        self.create_full_feature_vector('train')
        # self.create_full_feature_vector('test')
        # self.create_full_feature_vector('comp')

    def define_features_dicts(self):
        """
        This method define the features_dict which will be a dictionary with the dictionary
        and the description of each feature.
        Need to change for different tasks
        :return: features_dicts
        """
        features_dicts = {
            '1': [self.feature_1, 'p-word, p-pos'],
            '2': [self.feature_2, 'p-word'],
            '3': [self.feature_3, 'p-pos'],
            '4': [self.feature_4, 'c-word, c-pos'],
            '5': [self.feature_5, 'c-word'],
            '6': [self.feature_6, 'c-pos'],
            '7': [self.feature_7, 'p-word, p-pos, c-word, c-pos'],
            '8': [self.feature_8, 'p-pos, c-word, c-pos'],
            '9': [self.feature_9, 'p-word, c-word, c-pos'],
            '10': [self.feature_10, 'p-word, p-pos, c-pos'],
            '11': [self.feature_11, 'p-word, p-pos, c-word'],
            '12': [self.feature_12, 'p-word, c-word'],
            '13': [self.feature_13, 'p-pos, c-pos'],
            '14': [self.feature_14, 'p-pos, p-pos+1, c-pos-1, c-pos'],
            '15': [self.feature_15, 'p-pos-1, p-pos, c-pos-1, c-pos'],
            '16': [self.feature_16, 'p-pos, p-pos+1, c-pos, c-pos+1'],
            '17': [self.feature_17, 'p-pos-1, p-pos, c-pos, c-pos+1'],
            '18': [self.feature_18, 'p-pos, b-pos, c-pos'],
            '19': [self.feature_19, 'p-pos, c-pos, distance(p,c)'],
            '20': [self.feature_20, 'p-word, c-word, distance(p,c)'],
            '21': [self.feature_21, 'p-pos, c-word, c-pos, is_parent_before'],
            '22': [self.feature_22, 'p-word, c-word, c-pos, is_parent_before'],
            '23': [self.feature_23, 'p-word, p-pos, c-pos, is_parent_before'],
            '24': [self.feature_24, 'p-word, p-pos, c-word, is_parent_before'],
            '25': [self.feature_25, 'p-word, c-word, distance(p,c), is_parent_before'],
            '26': [self.feature_26, 'p-pos, c-pos, distance(p,c), is_parent_before'],
            '27': [self.feature_27, 'p-pos, c-pos, g-pos, is_parent_before, grand_parent_before'],
            '28': [self.feature_28, 'p-word, c-word, g-word, is_parent_before, grand_parent_before'],
            '29': [self.feature_29, 'p-word, c-word, bro-word, is_parent_before, brother_before'],
            '30': [self.feature_30, 'p-pos, c-pos, bro-pos, is_parent_before, brother_before'],
            '31': [self.feature_31, 'distance(p,c), is_parent_before']
        }

        return features_dicts

    def create_gold_tree_dictionary(self, mode):
        """
        This method create a dictionary with all the gold trees of a given file.
        :param mode will be the data we want to create a dictionary based on it (train, test of comp file)
        :return: a dictionary where keys are the sentence index and the values are dictionaries where their keys are
        the head node and the value ia a list of its target nodes, i.e all edges of the sentence
        --> no return - insert into self.gold_tree[mode]
        """

        print('{}: Start building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Start building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))

        if mode == 'train':
            data = self.train_data
        elif mode == 'test':
            data = self.test_data
        elif mode == 'comp':
            data = self.comp_data
        else:
            print('Data is not train, test or comp: cant create gold tree')
            return dict()

        sentence_index = -1
        sentence_dict = defaultdict(list)
        for index, row in data.iterrows():
            if row['token_counter'] == 1:
                # if this is the first word in the sentence: insert the list of the previous sen. to the gold tree dict
                if sentence_index == -1:  # if this is the first sentence - there is no previous sentence
                    sentence_index += 1
                else:  # if this is not the first sentence - insert the list of the previous sen. to the gold tree dict
                    self.gold_tree[mode][sentence_index] = sentence_dict
                    sentence_index += 1
                    sentence_dict = defaultdict(list)

            if mode != 'comp':
                # for each node add the sentence[token_counter]
                if row['token_counter'] not in sentence_dict.keys():
                    sentence_dict[row['token_counter']] = []
                # add the edge: {head: target}
                sentence_dict[row['token_head']].append(row['token_counter'])

            else:
                # if this is comp: append the node to the root's list
                sentence_dict[0].append(row['token_counter'])

            # update the sentence_index in the relevant data dataframe
            data.set_value(index, 'sentence_index', sentence_index)

        # for the last sentence
        self.gold_tree[mode][sentence_index] = sentence_dict
        sentence_index += 1

        # after we have the sentence index- we will create the data_token_pos_dict
        self.token_POS_dict[mode] = data[['token', 'token_POS', 'sentence_index', 'token_counter', 'token_head']]\
            .set_index(['sentence_index', 'token_counter']).to_dict()

        # add the root
        for root_index in range(sentence_index):
            self.token_POS_dict[mode]['token'][(root_index, 0)] = 'root'
            self.token_POS_dict[mode]['token_POS'][(root_index, 0)] = 'root'
            self.token_POS_dict[mode]['token_head'][(root_index, 0)] = -1

        if mode == 'comp':  # if this is comp - the next lines are not relevant
            print('{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))
            logging.info(
                '{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))

            return

        # add the head word and POS for each target
        for index, row in data.iterrows():
            token_head = row['token_head']
            curr_sentence_index = row['sentence_index']
            data.set_value(index, 'head_word', self.token_POS_dict[mode]['token'][(curr_sentence_index, token_head)])
            data.set_value(index, 'head_POS', self.token_POS_dict[mode]['token_POS'][(curr_sentence_index, token_head)])

        print('{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Finish building gold tree from {}'.format(time.asctime(time.localtime(time.time())), mode))

        return

    def create_edges_existed_on_train(self):
        """
        This method build edges_existed_on_train and pos_edges_existed_on_train which are all the possible edges that
        have been seen in the train data - their indexes and their POS
        :return: no return - build the class variables
        """

        start_time = time.time()
        print('{}: Start creating edges_existed_on_train={} and pos_edges_existed_on_train={}'.
              format(time.asctime(time.localtime(time.time())), self.use_edges_existed_on_train,
                            self.use_pos_edges_existed_on_train))
        logging.info('{}: Start creating edges_existed_on_train={} and pos_edges_existed_on_train={}'.
                     format(time.asctime(time.localtime(time.time())), self.use_edges_existed_on_train,
                            self.use_pos_edges_existed_on_train))
        if self.use_edges_existed_on_train:
            self.edges_existed_on_train = defaultdict(list)
            source_target_df = copy(self.train_data[['token_head', 'token_counter']])
            source_target_list_group_by = source_target_df.groupby('token_head', as_index=False)
            for source in source_target_list_group_by.groups.keys():
                target_df = source_target_list_group_by.get_group(source)['token_counter']
                target_list = list(set(target_df))
                self.edges_existed_on_train[source] = target_list

        if self.use_pos_edges_existed_on_train:
            self.pos_edges_existed_on_train = defaultdict(list)
            source_target_pos_df = copy(self.train_data[['head_POS', 'token_POS']])
            source_target_list_pos_df = source_target_pos_df.groupby('head_POS', as_index=False)
            for source in source_target_list_pos_df.groups.keys():
                target_df = source_target_list_pos_df.get_group(source)['token_POS']
                target_list = list(set(target_df))
                self.pos_edges_existed_on_train[source] = target_list

        print('{}: Finish creating edges_existed_on_train and pos_edges_existed_on_train in {} seconds'
              .format(time.asctime(time.localtime(time.time())), time.time() - start_time))
        logging.info('{}: Finish creating edges_existed_on_train and pos_edges_existed_on_train in {} seconds'.
                     format(time.asctime(time.localtime(time.time())), time.time() - start_time))

        return

    def build_features_from_train(self):
        """
        This method build a features from the train data based on the features we want to use
        and update the features dictionaries
        :return:
        """
        for index, row in self.train_data.iterrows():
            # Define for features creation
            p_word = copy(row['head_word'])
            p_pos = copy(row['head_POS'])
            c_word = copy(row['token'])
            c_pos = copy(row['token_POS'])
            sentence_index = copy(row['sentence_index'])
            # find p_pos_minus_1 and p_pos_plus_1
            parent_index = copy(row['token_head'])
            if parent_index == 1:
                p_pos_minus_1 = 'root'
            elif parent_index == 0:
                p_pos_minus_1 = 'before_root'
            else:
                p_pos_minus_1 = copy(self.token_POS_dict['train']['token_POS'][(sentence_index, parent_index - 1)])

            if (sentence_index, parent_index + 1) in self.token_POS_dict['train']['token_POS'].keys():
                p_pos_plus_1 = copy(self.token_POS_dict['train']['token_POS'][(sentence_index, parent_index + 1)])
            else:
                p_pos_plus_1 = 'end'

            # find c_pos_minus_1 and c_pos_plus_1
            child_index = copy(row['token_counter'])
            if child_index == 1:
                c_pos_minus_1 = 'root'
            elif child_index == 0:
                p_pos_minus_1 = 'before_root'
            else:
                c_pos_minus_1 = copy(self.token_POS_dict['train']['token_POS'][(sentence_index, child_index - 1)])

            if (sentence_index, child_index + 1) in self.token_POS_dict['train']['token_POS'].keys():
                c_pos_plus_1 = copy(self.token_POS_dict['train']['token_POS'][(sentence_index, child_index + 1)])
            else:
                c_pos_plus_1 = 'end'

            # create a list of all POS between the parent and the child
            pos_between = list()
            if parent_index > child_index:
                index_list = range(child_index + 1, parent_index)
                parent_before = False
            else:
                index_list = range(parent_index + 1, child_index)
                parent_before = True
            for index_between in index_list:
                pos_between.append(copy(self.token_POS_dict['train']['token_POS'][(sentence_index, index_between)]))

            p_c_distance = abs(parent_index - child_index)

            # get the grandparent index, pos and word
            if parent_index == 0:
                g_pos = 'before_root'
                g_word = 'before_root'
                grand_parent_before = True
            else:
                g_index = copy(self.token_POS_dict['train']['token_head'][(sentence_index, parent_index)])
                g_pos = copy(self.token_POS_dict['train']['token_POS'][(sentence_index, g_index)])
                g_word = copy(self.token_POS_dict['train']['token'][(sentence_index, g_index)])
                if g_index > parent_index:
                    grand_parent_before = False
                else:
                    grand_parent_before = True

            # get the brothers index, pos and word
            pos_brothers = list()
            word_brothers = list()
            brothers_list = copy(self.gold_tree['train'][sentence_index][parent_index])
            brothers_list.remove(child_index)
            if brothers_list:
                for brother_index in brothers_list:
                    if brother_index > child_index:
                        brother_before_child = False
                    else:
                        brother_before_child = True
                    pos_brothers.append([copy(self.token_POS_dict['train']['token_POS'][(sentence_index,
                                                                                         brother_index)]),
                                         brother_before_child])
                    word_brothers.append([copy(self.token_POS_dict['train']['token'][(sentence_index, brother_index)]),
                                          brother_before_child])

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
            # build feature_7 of p-word, p-pos, c-word, c-pos
            self.update_feature_dict('7', p_word=p_word, p_pos=p_pos, c_word=c_word, c_pos=c_pos)
            # build feature_8 of p-pos, c-word, c-pos
            self.update_feature_dict('8', p_pos=p_pos, c_word=c_word, c_pos=c_pos)
            # build feature_9 of p-word, c-word, c-pos
            self.update_feature_dict('9', p_word=p_word, c_word=c_word, c_pos=c_pos)
            # build feature_10 of p-word, p-pos, c-pos
            self.update_feature_dict('10', p_word=p_word, p_pos=p_pos, c_pos=c_pos)
            # build feature_11 of p-word, p-pos, c-word
            self.update_feature_dict('11', p_word=p_word, p_pos=p_pos, c_word=c_word)
            # build feature_11 of p-word, c-word
            self.update_feature_dict('12', p_word=p_word, c_word=c_word)
            # build feature_13 of p-pos, c-pos
            self.update_feature_dict('13', p_pos=p_pos, c_pos=c_pos)
            # build feature_14 of p-pos, p-pos+1, c-pos-1, c-pos
            self.update_feature_dict('14', p_pos=p_pos, c_pos=c_pos, p_pos_plus_1=p_pos_plus_1,
                                     c_pos_minus_1=c_pos_minus_1)
            # build feature_15 of p-pos-1, p-pos, c-pos-1, c-pos
            self.update_feature_dict('15', p_pos=p_pos, c_pos=c_pos, p_pos_minus_1=p_pos_minus_1,
                                     c_pos_minus_1=c_pos_minus_1)
            # build feature_16 of p-pos, p-pos+1, c-pos, c-pos+1
            self.update_feature_dict('16', p_pos=p_pos, c_pos=c_pos, p_pos_plus_1=p_pos_plus_1,
                                     c_pos_plus_1=c_pos_plus_1)
            # build feature_17 of p-pos-1, p-pos, c-pos, c-pos+1
            self.update_feature_dict('17', p_pos=p_pos, c_pos=c_pos, p_pos_minus_1=p_pos_minus_1,
                                     c_pos_plus_1=c_pos_plus_1)
            # build feature_18 of p-pos, b-pos, c-pos for all b-pos (POS of all words between parend and child)
            if pos_between:
                for b_pos in pos_between:
                    self.update_feature_dict('18', p_pos=p_pos, c_pos=c_pos, b_pos=b_pos)

            # build feature_19 of p-pos, c-pos, distance(p,c)
            self.update_feature_dict('19', p_pos=p_pos, c_pos=c_pos, distance_p_c=p_c_distance)
            # build feature_20 of p-word, c-word, distance(p,c)
            self.update_feature_dict('20', p_word=p_word, c_word=c_word, distance_p_c=p_c_distance)
            # build feature_21 of p-pos, c-word, c-pos, is_parent_before
            self.update_feature_dict('21', p_pos=p_pos, c_word=c_word, c_pos=c_pos, is_parent_before=parent_before)
            # build feature_22 of p-word, c-word, c-pos, is_parent_before
            self.update_feature_dict('22', p_word=p_word, c_word=c_word, c_pos=c_pos, is_parent_before=parent_before)
            # build feature_23 of p-word, p-pos, c-pos, is_parent_before
            self.update_feature_dict('23', p_word=p_word, p_pos=p_pos, c_pos=c_pos, is_parent_before=parent_before)
            # build feature_24 of p-word, p-pos, c-word, is_parent_before
            self.update_feature_dict('24', p_word=p_word, p_pos=p_pos, c_word=c_word, is_parent_before=parent_before)
            # build feature_25 of p-word, c-word, distance(p,c), is_parent_before
            self.update_feature_dict('25', p_word=p_word, c_word=c_word, is_parent_before=parent_before,
                                     distance_p_c=p_c_distance)
            # build feature_26 of p-pos, c-pos, distance(p,c), is_parent_before
            self.update_feature_dict('26', p_pos=p_pos, c_pos=c_pos, is_parent_before=parent_before,
                                     distance_p_c=p_c_distance)
            # build feature_27 of p-pos, c-pos, g-pos, grand_parent_before, parent_before
            self.update_feature_dict('27', p_pos=p_pos, c_pos=c_pos, g_pos=g_pos,
                                     grand_parent_before=grand_parent_before, is_parent_before=parent_before)
            # build feature_28 of p-word, c-word, g-word, grand_parent_before, parent_before
            self.update_feature_dict('28', p_word=p_word, c_word=c_word, g_word=g_word,
                                     grand_parent_before=grand_parent_before, is_parent_before=parent_before)
            if pos_brothers:
                # build feature_29 of p-word, c-word, bro-word, brother_child_before, parent_before
                for bro_pos, brother_child_before in pos_brothers:
                    self.update_feature_dict('29', p_word=p_word, c_word=c_word, bro_pos=bro_pos,
                                             brother_child_before=brother_child_before, is_parent_before=parent_before)
                for bro_word, brother_child_before in word_brothers:
                    # build feature_30 of p-pos, c-pos, bro-pos, brother_child_before, parent_before
                    self.update_feature_dict('30', p_word=p_word, c_word=c_word, bro_word=bro_word,
                                             brother_child_before=brother_child_before, is_parent_before=parent_before)
            # build feature_31 of distance(p,c), is_parent_before
            self.update_feature_dict('31', is_parent_before=parent_before, distance_p_c=p_c_distance)

        # save all features dicts to csv
        for feature in self.features_dicts.keys():
            self.save_dictionary(feature)

        return

    def update_feature_dict(self, feature_number, p_word=None, p_pos=None, c_word=None, c_pos=None, p_pos_minus_1=None,
                            p_pos_plus_1=None, c_pos_plus_1=None, c_pos_minus_1=None, b_pos=None, distance_p_c=None,
                            is_parent_before=None, g_pos=None, g_word=None, grand_parent_before=None,
                            bro_pos=None, bro_word=None, brother_child_before=None):
        """
        This method update the relevant feature dictionary
        :param feature_number: the number of the feature
        :param p_word: the word of the parent
        :param p_pos: the POS of the parent
        :param c_word: the word of the child
        :param c_pos: the POS of the child
        :param p_pos_minus_1: POS to the left of parent in sentence
        :param p_pos_plus_1: POS to the right of parent in sentence
        :param c_pos_minus_1: POS to the left of child in sentence
        :param c_pos_plus_1: POS to the right of child in sentence
        :param b_pos: POS of a word in between parent and child nodes.
        :param distance_p_c: distance between parent and child
        :param is_parent_before: if the parent has lower index than the child
        :param g_pos: the POS of the grandparent
        :param g_word: the word of the grandparent
        :param grand_parent_before: if the grand_parent before the parent in the sentence
        :param bro_pos: the POS of the brother
        :param bro_word: the word of the brother
        :param brother_child_before: if the brother before the child in the sentence
        :return: no return, just update the object's features' dictionaries
        """

        # Create the list of relevant feature components
        option_for_features_list = [p_word, p_pos, c_word, c_pos, p_pos_minus_1, p_pos_plus_1, c_pos_plus_1,
                                    c_pos_minus_1, b_pos, str(distance_p_c), str(is_parent_before), g_pos, g_word,
                                    str(grand_parent_before), bro_pos, bro_word, str(brother_child_before)]
        option_for_features_list = [x for x in option_for_features_list if x is not None and x != 'None']
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
        :return: no return, just save the dictionary
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
        This method build a feature vector with all the features we created based on the train data
        :return: no return, just create and save the dictionaries
        """
        start_time = time.time()
        print('{}: Start building feature vector'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Start building feature vector'.format(time.asctime(time.localtime(time.time()))))

        features_vector_idx = 0

        # for each feature type in features_combination, we create its instances' feature_key
        for feature_number in self.features_combination:
            feature_instances = 0
            feature_dict = self.features_dicts[feature_number][0]
            feature_description = self.features_dicts[feature_number][1]
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

        print('{}: Finished building features vector in : {} seconds'.format(time.asctime(time.localtime(time.time())),
              time.time() - start_time))
        logging.info('{}: Finished building features vector in : {} seconds'.format(time.asctime(time.localtime(time.time())),
                     time.time() - start_time))

        # Saving dictionaries to csv
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
        print('{}: Finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Finished saving features_vector_mapping'.format(time.asctime(time.localtime(time.time()))))

        self.feature_vec_len = features_vector_idx
        print('{}: Feature vector len is: {}'.format(time.asctime(time.localtime(time.time())), self.feature_vec_len))
        logging.info('{}: Feature vector len is: {}'.
                     format(time.asctime(time.localtime(time.time())), self.feature_vec_len))

        return

    def calculate_local_feature_vec_per_feature(self, indexes_vector, feature_number, p_word=None, p_pos=None,
                                                c_word=None, c_pos=None, p_pos_minus_1=None, p_pos_plus_1=None,
                                                c_pos_plus_1=None, c_pos_minus_1=None, b_pos=None, is_full_graph=False,
                                                distance_p_c=None, is_parent_before=None, g_pos=None, g_word=None,
                                                grand_parent_before=None, bro_pos=None, bro_word=None,
                                                brother_child_before=None):
        """
        This method create a feature vector per feature number for a given edge and a given feature number
        :param indexes_vector: if is_full_graph is True - this is a list and we will insert the indexes of the features
                            if is_full_graph is False - this is a numpy array that will be the features vector
        :param feature_number: the number of the feature we working on
        :param p_word: the word of the parent
        :param p_pos: the POS of the parent
        :param c_word: the word of the child
        :param c_pos: the POS of the child
        :param p_pos_minus_1: POS to the left of parent in sentence
        :param p_pos_plus_1: POS to the right of parent in sentence
        :param c_pos_minus_1: POS to the left of child in sentence
        :param c_pos_plus_1: POS to the right of child in sentence
        :param b_pos: POS of a word in between parent and child nodes.
        :param is_full_graph: will be True if we build the full graph features vector
        :param distance_p_c: the distance between the parend and the child
        :param is_parent_before: if the parent has lower index than the child
        :param g_pos: the POS of the grandparent
        :param g_word: the word of the grandparent
        :param grand_parent_before: if grand_parent before the parent
        :param bro_pos: the POS of the brother
        :param bro_word: the word of the brother
        :param brother_child_before: if the brother before the child in the sentence
        :return: no return, the indexes_vector is updated
        """
        option_for_features_list = [p_word, p_pos, c_word, c_pos, p_pos_minus_1, p_pos_plus_1, c_pos_plus_1,
                                    c_pos_minus_1, b_pos, str(distance_p_c), str(is_parent_before), g_pos, g_word,
                                    str(grand_parent_before), bro_pos, bro_word, str(brother_child_before)]
        option_for_features_list = [x for x in option_for_features_list if x is not None and x != 'None']
        if feature_number in self.features_combination:
            # build feature
            feature_dict = self.features_dicts[feature_number][0]
            # build feature name
            feature_key = 'f' + feature_number
            for option_feature in option_for_features_list:
                feature_key += '_' + option_feature

            if feature_key in feature_dict:
                feature_idx = self.features_vector[feature_key]
                if is_full_graph:
                    indexes_vector.append(feature_idx)
                else:
                    indexes_vector[feature_idx] = 1

        return

    def get_local_feature_vec(self, sentence_index, source, target, mode, is_full_graph=False, tree=None):
        """
        This method create a feature vector for a given edge
        :param sentence_index: the index of the sentence
        :param source: the token_counter of the parent
        :param target: the token_counter of the child
        :param mode: the data type: train or test or comp
        :param is_full_graph: will be True if we build the full graph features vector
        :param tree: if this is not a full-graph case: it should get the tree to get the grandparent and the siblings
        :return: the vector feature of the given edge
        """

        if is_full_graph:  # if we build the full graph - we will build a list of indexes of the features
            indexes_vector = list()
        else:  # if we don't build the full graph - we will build a features vector
            indexes_vector = np.zeros(shape=self.feature_vec_len, dtype=int)

        p_word = copy(self.token_POS_dict[mode]['token'][(sentence_index, source)])
        p_pos = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, source)])
        c_word = copy(self.token_POS_dict[mode]['token'][(sentence_index, target)])
        c_pos = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, target)])

        # find p_pos_minus_1 and p_pos_plus_1
        parent_index = source
        if parent_index == 1:
            p_pos_minus_1 = 'root'
        elif parent_index == 0:
            p_pos_minus_1 = 'before_root'
        else:
            p_pos_minus_1 = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, parent_index - 1)])

        if (sentence_index, parent_index + 1) in self.token_POS_dict[mode]['token_POS'].keys():
            p_pos_plus_1 = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, parent_index + 1)])
        else:
            p_pos_plus_1 = 'end'

        # find c_pos_minus_1 and c_pos_plus_1
        child_index = target
        if child_index == 1:
            c_pos_minus_1 = 'root'
        elif child_index == 0:
            p_pos_minus_1 = 'before_root'
        else:
            c_pos_minus_1 = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, child_index - 1)])

        if (sentence_index, child_index + 1) in self.token_POS_dict[mode]['token_POS'].keys():
            c_pos_plus_1 = copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, child_index + 1)])
        else:
            c_pos_plus_1 = 'end'

        # create a list of all POS between the parent and the child
        pos_between = list()
        if parent_index > child_index:
            index_list = range(child_index + 1, parent_index)
            parent_before = False
        else:
            index_list = range(parent_index + 1, child_index)
            parent_before = True
        for index_between in index_list:
            pos_between.append(copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, index_between)]))

        p_c_distance = abs(parent_index - child_index) - 1

        # get the grandparent and brothers index, pos and word
        pos_brothers = list()
        word_brothers = list()
        brothers_list = list()
        g_index_list = list()
        g_pos_list = list()
        g_word_list = list()
        if parent_index == 0:
            g_pos_list.append(['before_root', True])
            g_word_list.append(['before_root', True])

        else:
            if is_full_graph:  # if we are creating features for the full graph - take the grandparent from it
                                # there might be more than one parent
                for source_index, target_list in self.full_graph[mode][sentence_index].items():
                    if parent_index in target_list:
                        g_index_list.append(source_index)
                if not g_index_list:  # if the list is empty:
                    print('no parent for index: {} in {} for full graph = {}'.format(parent_index, mode, is_full_graph))

            elif tree is not None:  # if we are creating features for the specific tree - take the grandparent from it
                for source_index, target_list in tree.items():
                    if parent_index in target_list:
                        g_index_list.append(source_index)
                if not g_index_list:  # if the list is empty:
                    print('no parent for index: {} in {} for full graph = {}'.format(parent_index, mode, is_full_graph))

            else:
                print('no full graph and no tree was given for sentence {} in mode {}'.format(sentence_index, mode))

            for g_index in g_index_list:
                if g_index > parent_index:
                    temp_grand_parent_before = False
                else:
                    temp_grand_parent_before = True
                g_pos_list.append([copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, g_index)]),
                                   temp_grand_parent_before])
                g_word_list.append([copy(self.token_POS_dict[mode]['token'][(sentence_index, g_index)]),
                                    temp_grand_parent_before])

        # get the brothers index, pos and word
        if is_full_graph:
            brothers_list = copy(self.full_graph[mode][sentence_index][parent_index])
        elif tree is not None:
            brothers_list = copy(self.gold_tree[mode][sentence_index][parent_index])

        if child_index in brothers_list:
            brothers_list.remove(child_index)
        if brothers_list:
            for brother_index in brothers_list:
                if brother_index > child_index:
                    brother_before_child = False
                else:
                    brother_before_child = True
                pos_brothers.append([copy(self.token_POS_dict[mode]['token_POS'][(sentence_index, brother_index)]),
                                    brother_before_child])
                word_brothers.append([copy(self.token_POS_dict[mode]['token'][(sentence_index, brother_index)]),
                                      brother_before_child])

        # calculate feature_1 of p-word, p-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '1', p_word=p_word, p_pos=p_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_2 of p-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '2', p_word=p_word, is_full_graph=is_full_graph)
        # calculate feature_3 of p_pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '3', p_pos=p_pos, is_full_graph=is_full_graph)
        # calculate feature_4 of c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '4', c_word=c_word, c_pos=c_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_5 of c-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '5', c_word=c_word, is_full_graph=is_full_graph)
        # calculate feature_6 of c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '6', c_pos=c_pos, is_full_graph=is_full_graph)
        # calculate feature_7 of p-word, p-pos, c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '7', p_word=p_word, p_pos=p_pos, c_word=c_word,
                                                     c_pos=c_pos, is_full_graph=is_full_graph)
        # calculate feature_8 of p-pos, c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '8', p_pos=p_pos, c_word=c_word, c_pos=c_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_9 of p-word, c-word, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '9', p_word=p_word, c_word=c_word, c_pos=c_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_10 of p-word, p-pos, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '10', p_word=p_word, p_pos=p_pos, c_pos=c_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_11 of p-word, p-pos, c-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '11', p_word=p_word, p_pos=p_pos, c_word=c_word,
                                                     is_full_graph=is_full_graph)
        # calculate feature_11 of p-word, c-word
        self.calculate_local_feature_vec_per_feature(indexes_vector, '12', p_word=p_word, c_word=c_word,
                                                     is_full_graph=is_full_graph)
        # calculate feature_13 of p-pos, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '13', p_pos=p_pos, c_pos=c_pos,
                                                     is_full_graph=is_full_graph)
        # calculate feature_14 of p-pos, p-pos+1, c-pos-1, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '14', p_pos=p_pos, c_pos=c_pos,
                                                     p_pos_plus_1=p_pos_plus_1, c_pos_minus_1=c_pos_minus_1,
                                                     is_full_graph=is_full_graph)
        # calculate feature_15 of p-pos-1, p-pos, c-pos-1, c-pos
        self.calculate_local_feature_vec_per_feature(indexes_vector, '15', p_pos=p_pos, c_pos=c_pos,
                                                     p_pos_minus_1=p_pos_minus_1, c_pos_minus_1=c_pos_minus_1,
                                                     is_full_graph=is_full_graph)
        # calculate feature_16 of p-pos, p-pos+1, c-pos, c-pos+1
        self.calculate_local_feature_vec_per_feature(indexes_vector, '16', p_pos=p_pos, c_pos=c_pos,
                                                     p_pos_plus_1=p_pos_plus_1, c_pos_plus_1=c_pos_plus_1,
                                                     is_full_graph=is_full_graph)
        # calculate feature_17 of p-pos-1, p-pos, c-pos, c-pos+1
        self.calculate_local_feature_vec_per_feature(indexes_vector, '17', p_pos=p_pos, c_pos=c_pos,
                                                     p_pos_minus_1=p_pos_minus_1, c_pos_plus_1=c_pos_plus_1,
                                                     is_full_graph=is_full_graph)
        # calculate feature_18 of p-pos, b-pos, c-pos for all b-pos (POS of all words between parend and child)
        if pos_between:
            for b_pos in pos_between:
                self.calculate_local_feature_vec_per_feature(indexes_vector, '18', p_pos=p_pos, c_pos=c_pos,
                                                             b_pos=b_pos, is_full_graph=is_full_graph)

        # build feature_19 of p-pos, c-pos, distance(p,c)
        self.calculate_local_feature_vec_per_feature(indexes_vector, '19', p_pos=p_pos, c_pos=c_pos, distance_p_c=p_c_distance,
                                                     is_full_graph=is_full_graph)
        # build feature_20 of p-word, c-word, distance(p,c)
        self.calculate_local_feature_vec_per_feature(indexes_vector, '20', p_word=p_word, c_word=c_word, distance_p_c=p_c_distance,
                                                     is_full_graph=is_full_graph)
        # build feature_21 of p-pos, c-word, c-pos, is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '21', p_pos=p_pos, c_word=c_word, c_pos=c_pos,
                                                     is_parent_before=parent_before, is_full_graph=is_full_graph)
        # build feature_22 of p-word, c-word, c-pos, is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '22', p_word=p_word, c_word=c_word, c_pos=c_pos,
                                                     is_parent_before=parent_before, is_full_graph=is_full_graph)
        # build feature_23 of p-word, p-pos, c-pos, is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '23', p_word=p_word, p_pos=p_pos, c_pos=c_pos,
                                                     is_parent_before=parent_before, is_full_graph=is_full_graph)
        # build feature_24 of p-word, p-pos, c-word, is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '24', p_word=p_word, p_pos=p_pos, c_word=c_word,
                                                     is_parent_before=parent_before, is_full_graph=is_full_graph)
        # build feature_25 of p-word, c-word, distance(p,c), is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '25', p_word=p_word, c_word=c_word,
                                                     is_parent_before=parent_before, distance_p_c=p_c_distance,
                                                     is_full_graph=is_full_graph)
        # build feature_26 of p-pos, c-pos, distance(p,c), is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '26', p_pos=p_pos, c_pos=c_pos,
                                                     is_parent_before=parent_before, distance_p_c=p_c_distance,
                                                     is_full_graph=is_full_graph)
        # build feature_27 of p-pos, c-pos, g-pos
        for g_pos, grand_parent_before in g_pos_list:
            self.calculate_local_feature_vec_per_feature(indexes_vector, '27', p_pos=p_pos, c_pos=c_pos, g_pos=g_pos,
                                                         is_full_graph=is_full_graph, is_parent_before=parent_before,
                                                         grand_parent_before=grand_parent_before)
        # build feature_28 of p-word, c-word, g-word
        for g_word, grand_parent_before in g_word_list:
            self.calculate_local_feature_vec_per_feature(indexes_vector, '28', p_word=p_word, c_word=c_word,
                                                         g_word=g_word, is_full_graph=is_full_graph,
                                                         is_parent_before=parent_before,
                                                         grand_parent_before=grand_parent_before)

        if pos_brothers:
            # build feature_29 of p-word, c-word, bro-word, brother_child_before, parent_before
            for bro_pos, brother_child_before in pos_brothers:
                self.calculate_local_feature_vec_per_feature(indexes_vector, '29', p_word=p_word, c_word=c_word,
                                                             bro_pos=bro_pos, brother_child_before=brother_child_before,
                                                             is_parent_before=parent_before, is_full_graph=is_full_graph)
            for bro_word, brother_child_before in word_brothers:
                # build feature_30 of p-pos, c-pos, bro-pos, brother_child_before, parent_before
                self.calculate_local_feature_vec_per_feature(indexes_vector, '30', p_word=p_word, c_word=c_word,
                                                             bro_word=bro_word, is_full_graph=is_full_graph,
                                                             brother_child_before=brother_child_before,
                                                             is_parent_before=parent_before)
        # build feature_31 of distance(p,c), is_parent_before
        self.calculate_local_feature_vec_per_feature(indexes_vector, '31', is_parent_before=parent_before,
                                                     distance_p_c=p_c_distance, is_full_graph=is_full_graph)

        return indexes_vector

    def create_global_feature_vector(self, tree, sentence_index, mode):
        """
        create a global feature vector for a given tree
        :param tree: a dict of that maps all the edges in the tree in the format:
        {source_node: a list of its target nodes]}. For example: {1: [2], 2: [1, 3], 3: [1]}
        :param sentence_index: the number of the sentence in the train data
        :param mode: the data type: train or test
        :return:
        """

        tree_indexes_vector = np.zeros(shape=self.feature_vec_len, dtype=int)
        for edge in tree.items():
            source = edge[0]
            target_nodes_list = edge[1]
            for target in target_nodes_list:
                edge_indexes_vector = self.get_local_feature_vec(sentence_index, source, target, mode, tree=tree)
                tree_indexes_vector = np.add(edge_indexes_vector, tree_indexes_vector)

        # csr_tree_indexes_vector = csr_matrix(tree_indexes_vector)
        # return csr_tree_indexes_vector
        return tree_indexes_vector

    def create_gold_tree_feature_vector(self, mode):
        """
        create feature vectors for the tree gold of each of the sentences in the train data
        :param mode: the data type: train or test
        :return: no return, just save the dictionary with the feature vector for each sentence
        """

        start_time = time.time()
        print('{}: Start building feature vectors for gold trees {}'.
              format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Start building feature vectors for gold trees {}'.
                     format(time.asctime(time.localtime(time.time())), mode))

        for sentence_index, sentence_tree in self.gold_tree[mode].items():
            self.gold_tree_features_vector[mode][sentence_index] =\
                self.create_global_feature_vector(sentence_tree, sentence_index, mode)
        run_time = (time.time() - start_time)/60.0
        print('{}: Finished building feature vectors for gold trees {} in : {} minutes'.
              format(time.asctime(time.localtime(time.time())), mode, run_time))
        logging.info('{}: Finished building feature vectors for gold trees {} in : {} minutes'.
                     format(time.asctime(time.localtime(time.time())), mode, run_time))

        print('{}: Saving feature vectors for gold trees {}'.format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Saving feature vectors for gold trees {}'.format(time.asctime(time.localtime(time.time())),
                                                                           mode))
        w = csv.writer(open(self.dict_path + 'features_vector_gold_tree_' + mode + '.csv', "w"))
        for key, val in self.gold_tree_features_vector[mode].items():
            w.writerow([key, val])

        print('{}: Finished saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Finished saving feature vectors {}'.format(time.asctime(time.localtime(time.time())), mode))

        return

    def create_full_feature_vector(self, mode):
        """
        create feature vectors for the full graph of each of the sentences in the mode data
        :param mode: the data type: train, test or comp
        :return: no return, just save the dictionary with the feature vector for each sentence
        """

        start_time = time.time()
        print('{}: Start building feature vectors for full graph {}'.
              format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Start building feature vectors for full graph {}'.
                     format(time.asctime(time.localtime(time.time())), mode))

        # get full graphs for the mode
        _, full_graphs = self.graph_utils.create_full_graph(self.gold_tree[mode], self.edges_existed_on_train,
                                                            self.pos_edges_existed_on_train,
                                                            self.token_POS_dict[mode]['token_POS'])
        self.full_graph[mode] = copy(full_graphs)
        for sentence_index, sentence_full_graph in full_graphs.items():
            for source, target_list in sentence_full_graph.items():
                for target in target_list:
                    self.full_graph_features_vector[mode][sentence_index][(source, target)] =\
                        self.get_local_feature_vec(sentence_index, source, target, mode, is_full_graph=True)

        print('{}: Finished building feature vectors for full graph {} in : {} seconds'.
              format(time.asctime(time.localtime(time.time())), mode, time.time() - start_time))
        logging.info('{}: Finished building feature vectors for full graph {} in : {} seconds'.
                     format(time.asctime(time.localtime(time.time())), mode, time.time() - start_time))

        print('{}: Saving feature vectors for full graph {}'.format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Saving feature vectors for full graph {}'.format(time.asctime(time.localtime(time.time())),
                                                                           mode))
        w = csv.writer(open(self.dict_path + 'features_vector_full_graph_' + mode + '.csv', "w"))
        for key, val in self.full_graph_features_vector[mode].items():
            w.writerow([key, val])

        print('{}: Finished saving feature vectors for full graph {}'.
              format(time.asctime(time.localtime(time.time())), mode))
        logging.info('{}: Finished saving feature vectors for full graph{}'.
                     format(time.asctime(time.localtime(time.time())), mode))

        return


if __name__ == '__main__':
    all_start_time = time.time()
    curr_directory = os.path.abspath(os.curdir)    # '/Users/reutapel/Documents/Technion/Msc/NLP/hw2/NLP_HW2/'
    train_file = os.path.join(curr_directory, 'HW2-files', 'train.labeled')
    test_file = os.path.join(curr_directory, 'HW2-files', 'test.labeled')
    comp_file = os.path.join(curr_directory, 'HW2-files', 'comp.unlabeled')

    features = range(1, 19)
    features = [str(i) for i in features]
    model_obj = ParserModel(curr_directory, train_file, test_file, comp_file, features)

    run_time_cv = (time.time() - all_start_time) / 60.0
    print('{}: Finished all parser model creation in : {} minutes'.
          format(time.asctime(time.localtime(time.time())), run_time_cv))
