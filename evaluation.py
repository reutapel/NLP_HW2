import csv
import itertools
import logging
import time
import os
from copy import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import pickle


class Evaluate:

    """
    this class evaluates the results by calculating accuracy, and creates the predicted file for the competition.
    also makes analysis of test results.
    """

    def __init__(self, model, inference_obj, directory):

        """
        :param model: Dependency tree model object
        :param inference_obj: perceptron obj that calls the CLE inference class
        """
        print('{}: Building class Evaluate instance'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Building class Evaluate instance'.format(time.asctime(time.localtime(time.time()))))

        self.model = model
        self.inference_obj = inference_obj
        self.gold_tree = None
        self.token_POS_dict = None
        self.inference_mode = None
        self.data = None
        self.dir = directory
        self.directory = os.path.join(directory, 'evaluations')

    def update_inference_mode(self, inference_mode, weight_vec):

        """
        :param inference_mode: for updates the class variables to the correct work mode
        :param weight_vec: current working weights
        :return:
        """

        self.inference_mode = inference_mode
        if self.inference_mode == 'train':
            self.gold_tree = self.model.gold_tree[inference_mode]
            self.token_POS_dict = self.model.token_POS_dict[inference_mode]
            self.data = copy(self.model.train_data)
            print('{}: Evaluation updated to train mode'.format(time.asctime(time.localtime(time.time()))))
            logging.info('{}: Evaluation updated to train mode'.format(time.asctime(time.localtime(time.time()))))
        elif self.inference_mode == 'test':
            self.model.create_gold_tree_dictionary(inference_mode)
            self.model.create_gold_tree_feature_vector(inference_mode)
            self.model.create_full_feature_vector(inference_mode)
            self.gold_tree = self.model.gold_tree[inference_mode]
            self.token_POS_dict = self.model.token_POS_dict[inference_mode]
            self.data = copy(self.model.test_data)
            print('{}: Evaluation updated to test mode'.format(time.asctime(time.localtime(time.time()))))
            logging.info('{}: Evaluation updated to test mode'.format(time.asctime(time.localtime(time.time()))))
        else:
            self.model.create_gold_tree_dictionary(inference_mode)
            self.model.create_full_feature_vector(inference_mode)
            self.gold_tree = self.model.gold_tree[inference_mode]
            self.data = copy(self.model.comp_data)
            print('{}: Evaluation updated to comp mode'.format(time.asctime(time.localtime(time.time()))))
            logging.info('{}: Evaluation updated to comp mode'.format(time.asctime(time.localtime(time.time()))))
        # change perceptron to train/test mode, should influence it's gold tree to be train/test gold tree,
        # and function edge score to train/test mode.
        self.inference_obj.inference_mode(self.inference_mode,weight_vec)
        return

    def calculate_accuracy(self, weight_vec, weights, inference_mode=None,):

        """
        this method calculates the accuracy of the prediction according to the gold tree on /train test
        :param inference_mode: for updates the class variables to the correct work mode
        :return: accuracy on the data set
        """

        if inference_mode == 'comp':
            print('can not calculate accuracy for non train/test mode')
            logging.info('can not calculate accuracy for non train/test mode')
            return

        # change relevant class variables of this class and the learning model class
        self.update_inference_mode(inference_mode, weight_vec)

        data_mistake_num = 0
        data_num_tokens = len(self.token_POS_dict['token'].keys())
        sentences_count = 0
        mistakes_dict = dict()
        print('{}: Start calculating accuracy for weights: {}'.format(time.asctime(time.localtime(time.time())),
                                                                      weights))
        logging.info('{}: Start calculating accuracy for weights: {}'.format(time.asctime(time.localtime
                                                                                          (time.time())), weights))
        for t in range(len(self.gold_tree)):
            sentence_mistake_num = 0
            gold_sentence = self.gold_tree[t]
            pred_tree = self.inference_obj.calculate_mst(t)
            for source, targets in gold_sentence.items():
                missed_targets = set(targets).difference(set(pred_tree[source]))
                wrong_targets = set(pred_tree[source]).difference(set(targets))
                data_mistake_num += len(missed_targets)
                sentence_mistake_num += len(missed_targets)
                if t not in mistakes_dict.keys():
                    mistakes_dict[t] = dict()
                    mistakes_dict[t][source] = dict()
                    mistakes_dict[t][source]['missed_targets'] = missed_targets
                    mistakes_dict[t][source]['wrong_targets'] = wrong_targets
                else:
                    mistakes_dict[t][source] = dict()
                    mistakes_dict[t][source]['missed_targets'] = missed_targets
                    mistakes_dict[t][source]['wrong_targets'] = wrong_targets
            sentences_count += 1
        accuracy = 1 - data_mistake_num / (data_num_tokens - sentences_count)

        self.analyzer(mistakes_dict, inference_mode, accuracy ,weights)
        logging.info('{}: Accuracy for mode {} and weights: {} is : {:%} '.format(time.asctime(time.localtime(time.time())),
                                                             self.inference_mode, weights, accuracy))
        print('{}: Accuracy for mode {} and weights: {} is : {:%} '.format(time.asctime(time.localtime(time.time())),
                                                             self.inference_mode, weights, accuracy))
        print('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        mistakes_dict_name = 'weights_{}_accuracy_{}_mistakes_dict_{}'.format(weights, accuracy, self.inference_mode)
        w = csv.writer(open(os.path.join(self.directory, "{}.csv".format(mistakes_dict_name)), "w"))
        for key, val in mistakes_dict.items():
            w.writerow([key, val])
        print('{}: finished saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))

        return accuracy, mistakes_dict_name

    def reverse_dict(self, pred_tree):
        """
        :param pred_tree: the prediction tree returned from calculate mst
        :return: the reverse tree where the key is the target and the value is the head
        """

        pred_tree_reverse = {}
        for head, targets in pred_tree:
            for target in targets:
                pred_tree_reverse[target] = head

        return pred_tree_reverse

    # TODO: test function
    def infer(self,best_weights_vec, best_weights, inference_mode=None):

        """
        this method uses the inference object in order to create the predictions file
        :param inference_mode: for updates the class variables to the correct work mode
        """

        # change relevant class variables
        self.update_inference_mode(inference_mode, best_weights_vec)
        #for sentence_index in range(len(self.gold_tree)):
        #sentence_index = len(self.gold_tree)
        sentence_index = -1
        pred_tree = dict()
        pred_tree_reverse = dict()
        for index, row in self.data.iterrows():
            if row['token_counter'] == 1:
                sentence_index += 1
                pred_tree = self.inference_obj.calculate_mst(sentence_index)
                pred_tree_reverse = self.reverse_dict(pred_tree)
            row['token_head'] = pred_tree_reverse[row['token_counter']]

        # delete additional column for comp format
        if inference_mode == 'comp':
            self.data = self.data.drop('sentence_index',1)

        # save to file
        saved_file_name = 'inference file for mode:{} - {} and best weights: {}'.format(inference_mode,
                          time.asctime(time.localtime(time.time())), best_weights)
        self.data.to_csv(saved_file_name,sep='\t', header=False)

        return saved_file_name

    def analyzer(self, mistakes_dict, inference_mode, accuracy, weights):
        """
        this method analyzes the mistakes of the prediction
        :param mistakes_dict: the dict that contains the missed and wrong predictions
        :inference_mode: mode of analysis
        :return: all the analysis
        """

        # reversing dict for easier analysis
        print('{}: starting analyzing mistakes for weights: {}'.format(time.asctime(time.localtime(time.time())),
                                                                       weights))
        logging.info('{}: starting analyzing mistakes for weights: {}'.format(time.asctime(time.localtime(time.time())),
                                                                              weights))
        analysis_dict = dict()
        for t in mistakes_dict.keys():
            analysis_dict[t] = dict()
            first_wrong = True
            first_missed = True
            for source, mis_dict in mistakes_dict[t].items():
                if first_wrong:
                    analysis_dict[t]['wrong_targets'] = dict()
                    first_wrong = False
                wrong_targets = mis_dict['wrong_targets']
                for wrong_target in wrong_targets:
                    analysis_dict[t]['wrong_targets'][wrong_target] = source
                if first_missed:
                    analysis_dict[t]['missed_targets'] = dict()
                    first_missed = False
                missed_targets = mis_dict['missed_targets']
                for missed_target in missed_targets:
                    analysis_dict[t]['missed_targets'][missed_target] = source

        # now build a dict for confusion matrix of POS: key is (missed source, missed target) value is count
        confusion_POS = dict()
        for t, missed_wrong_dict in analysis_dict.items():
            for missed_target in missed_wrong_dict['missed_targets']:
                missed_source = missed_wrong_dict['missed_targets'][missed_target]
                wrong_source = missed_wrong_dict['wrong_targets'][missed_target]
                if (self.token_POS_dict['token_POS'][t, missed_source],
                    self.token_POS_dict['token_POS'][t, wrong_source]) in confusion_POS.keys():
                    confusion_POS[(self.token_POS_dict['token_POS'][t, missed_source],
                                   self.token_POS_dict['token_POS'][t, wrong_source])] += 1
                else:
                    confusion_POS[(self.token_POS_dict['token_POS'][t, missed_source],
                                   self.token_POS_dict['token_POS'][t, wrong_source])] = 1

        print('{}: finished analyzing mistakes for weights: {}'.format(time.asctime(time.localtime(time.time()))
                                                                       , weights))
        logging.info('{}: finished analyzing mistakes for weights: {}'.format(time.asctime(time.localtime(time.time()))
                                                                              , weights))

        print('{}: saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        confusion_POS_name = 'confusion_POS_{}_acc_{}_{}'.format(inference_mode,accuracy, weights)
        w = csv.writer(open(os.path.join(self.directory, "{}.csv".format(confusion_POS_name)),"w"))
        for key, val in confusion_POS.items():
            w.writerow([key, val])
        print('{}: finished saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))

        return confusion_POS