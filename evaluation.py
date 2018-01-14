import csv
import itertools
import logging
import time
import os
from copy import copy
import pandas as pd

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
        self.directory = os.path.join(directory, 'evaluations')


    def update_inference_mode(self, inference_mode):

        """
        :param inference_mode: for updates the class variables to the correct work mode
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
            self.gold_tree = self.model.gold_tree[inference_mode]
            self.token_POS_dict = self.model.token_POS_dict[inference_mode]
            self.data = copy(self.model.test_data)
            print('{}: Evaluation updated to test mode'.format(time.asctime(time.localtime(time.time()))))
            logging.info('{}: Evaluation updated to test mode'.format(time.asctime(time.localtime(time.time()))))
        else:
            self.gold_tree = self.model.gold_tree[inference_mode]
            self.data = copy(self.model.comp_data)
            print('{}: Evaluation updated to comp mode'.format(time.asctime(time.localtime(time.time()))))
            logging.info('{}: Evaluation updated to comp mode'.format(time.asctime(time.localtime(time.time()))))
        # change perceptron to train/test mode, should influence it's gold tree to be train/test gold tree,
        # and function edge score to train/test mode.
        self.inference_obj.inference_mode(self.inference_mode)
        return

    def calculate_accuracy(self, inference_mode=None):

        """
        this method calculates the accuracy of the prediction according to the gold tree on /train test
        :param inference_mode: for updates the class variables to the correct work mode
        :return:
        """
        #
        # if inference_mode == 'comp':
        #     print('can not calculate accuracy for non train/test mode')
        #     logging.info('can not calculate accuracy for non train/test mode')
        #     return

        # change relevant class variables
        self.update_inference_mode(inference_mode)

        data_mistake_num = 0
        data_num_tokens = len(self.token_POS_dict['token'].keys())
        sentences_count = 0
        mistakes_dict = dict()

        print('{}: Start calculating accuracy'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: Start calculating accuracy'.format(time.asctime(time.localtime(time.time()))))
        for t in range(len(self.gold_tree)):
            sentence_mistake_num = 0
            gold_sentence = self.gold_tree[t]
            try:
                pred_tree = self.inference_obj.calculate_mst(t)
            except AssertionError as err:
                pred_tree = err.args
                print("The algorithm returned a bad tree, continuing with accuracy. \n tree: {}".format(pred_tree))
                logging.error("The algorithm returned a bad tree, continuing with accuracy. \n tree: {}".format(pred_tree))
            finally:
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

        self.analyzer(mistakes_dict, inference_mode, accuracy)
        logging.info('{}: Accuracy for {} is : {:%} '.format(time.asctime(time.localtime(time.time())),
                                                             self.inference_mode, accuracy))
        print('{}: accuracy for {} is: {:%} '.format(time.asctime(time.localtime(time.time())), self.inference_mode,
                                                     accuracy))

        print('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        mistakes_dict_name = 'accuracy_{}_mistakes_dict_{}'.format(accuracy, self.inference_mode)
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
    def infer(self, inference_mode=None):

        """
        this method uses the inference object in order to create the predictions file
        :param inference_mode: for updates the class variables to the correct work mode
        """

        # change relevant class variables
        self.update_inference_mode(inference_mode)
        #for sentence_index in range(len(self.gold_tree)):
        #sentence_index = len(self.gold_tree)
        sentence_index = -1
        pred_tree = dict()
        pred_tree_reverse = dict()
        for index, row in self.data.iterrows():
            if row['token_counter'] == 1:
                sentence_index += 1
                try:
                    pred_tree = self.inference_obj.calculate_mst(sentence_index)
                except AssertionError as err:
                    pred_tree = err.args
                    print("The algorithm returned a bad tree, continuing with inference. \n tree: {}".format(pred_tree))
                    logging.error("The algorithm returned a bad tree, continuing with inference. \n tree: {}".format(pred_tree))
                finally:
                    pred_tree_reverse = self.reverse_dict(pred_tree)
            row['token_head'] = pred_tree_reverse[row['token_counter']]

        # delete additional column for comp format
        if inference_mode == 'comp':
            self.data = self.data.drop('sentence_index',1)

        # save to file
        saved_file_name = 'inference file for mode:{} - {}'.format(inference_mode,
                          time.asctime(time.localtime(time.time())))
        self.data.to_csv(saved_file_name,sep='\t', header=False)

        return saved_file_name
#TODO: test function & save in matrix confusion mode and not in dict mode
    def analyzer(self, mistakes_dict, inference_mode, accuracy):
        """
        this method analyzes the mistakes of the prediction
        :param mistakes_dict: the dict that contains the missed and wrong predictions
        :inference_mode: mode of analysis
        :return: all the analysis
        """

        # reversing dict for easier analysis
        print('{}: starting analyzing mistakes'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: starting analyzing mistakes'.format(time.asctime(time.localtime(time.time()))))
        analysis_dict = dict()
        for t in mistakes_dict.keys():
            analysis_dict[t] = dict()
            for source, mis_dict in mistakes_dict[t].items():
                analysis_dict[t]['wrong_targets'] = dict()
                wrong_targets = mis_dict['wrong_targets']
                for wrong_target in wrong_targets:
                    analysis_dict[t]['wrong_targets'][wrong_target] = source
                analysis_dict[t]['missed_targets'] = dict()
                missed_targets = mis_dict['missed_targets']
                for missed_target in missed_targets:
                    analysis_dict[t]['missed_targets'][missed_target] = source

        # now build a dict for confusion matrix of POS: key is (missed source, missed target) value is count
        confusion_POS = dict()
        for t, missed_dict in analysis_dict.items():
            for missed_target in mis_dict['missed_targets']:
                missed_source = mis_dict['missed_targets'][missed_target]
                wrong_source = mis_dict['wrong_targets'][missed_target]
                if (self.token_POS_dict[inference_mode]['token_POS'][t, missed_source],
                    self.token_POS_dict[inference_mode]['token_POS'][t, wrong_source]) in confusion_POS.keys():
                    confusion_POS[(self.token_POS_dict[inference_mode]['token_POS'][t, missed_source],
                              self.token_POS_dict[inference_mode]['token_POS'][t, wrong_source])] +=1
                else:
                    confusion_POS[(self.token_POS_dict[inference_mode]['token_POS'][t, missed_source],
                              self.token_POS_dict[inference_mode]['token_POS'][t, wrong_source])] = 1

        print('{}: finished analyzing mistakes'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished analyzing mistakes'.format(time.asctime(time.localtime(time.time()))))

        print('{}: saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        confusion_POS_name = 'confusion_POS_{}_acc_{}'.format(inference_mode,accuracy)
        w = csv.writer(open(os.path.join(self.directory, "{}.csv".format(confusion_POS_name)),"w"))
        for key, val in confusion_POS.items():
            w.writerow([key, val])
        print('{}: finished saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving confusion_POS'.format(time.asctime(time.localtime(time.time()))))

        return confusion_POS


# class Evaluate:
#     """
#     this class evaluates the results and creates the confusion matrix
#     """
#
#     def __init__(self, model, data_file, viterbi_result, write_file_name,
#                  confusion_file_name, comp=False, comp_file_name=None):
#         """
#         :param comp: if this run is for competition
#         :param model: MEMM model object
#         :param data_file: test data file
#         :param viterbi_result: results of the viterbi results
#         :type viterbi_result: dict with key:= sentence_index, value:= list (in form word_tag)
#         :param write_file_name: where to save the Viterbi results
#         :param confusion_file_name: where to save the confusion matrix
#         """
#         self.most_misses_tags = {}
#         self.data_file_name = data_file
#         self.predict_dict, self.viterbi_unseen_words = viterbi_result
#         self.viterbi_unseen_words = [list(loc) for loc in set(tuple(loc) for loc in self.viterbi_unseen_words)]
#         self.model = model
#         self.write_file_name = write_file_name
#         self.confusion_file_name = confusion_file_name
#         self.tags = list(model.tags_dict.keys())
#         self.tags.sort()
#         self.unseen_confusion_matrix = {}
#         self.confusion_matrix = {}
#         self.misses_matrix = {}
#         self.eval_res = {}
#         self.eval_res.update()
#         self.k = 10  # number of words in the confusion matrix
#         self.unseen_tags_set = set()
#         if not comp:
#             self.word_results_dictionary = self.eval_test_results()
#
#     def run(self):
#         self.write_result_doc()
#         self.write_confusion_doc()  # write tags confusion matrix
#
#         return self.word_results_dictionary
#
#     def eval_test_results(self):
#         # predicted_values
#         miss = 0
#         hit = 0
#         hit_unseen = 0
#         miss_unseen = 0
#
#         for tag1 in self.tags:
#             for tag2 in self.tags:
#                 tag_key = tag1 + '_' + tag2
#                 self.confusion_matrix.setdefault(tag_key, 0)
#                 self.misses_matrix.setdefault(tag_key, 0)
#
#         word_tag_tuples_dict = []
#         # with open(data_file_name, 'r') as training:  # real values
#         #     for sequence in training:
#         # todo: consider make the test tagging one time
#         with open(self.data_file_name, 'r') as train:
#             for index, seq in enumerate(train):
#                 seq = seq.rstrip('\n')
#                 d = seq.split(' ')
#                 word_tag_tuples_dict.append([])
#                 for i, val in enumerate(d):
#                     word_tag_tuples_dict[index].append(val.split('_'))
#                     predict_tuple = self.predict_dict[index][i].split('_')
#                     # print('sequence_index is: {}, predict_item is: {}').format(sequence_index, predict_item)
#                     predict_word = predict_tuple[0]
#                     predict_tag = predict_tuple[1]  # our predicted tag
#                     gold_word = word_tag_tuples_dict[index][i][0]
#                     gold_tag = word_tag_tuples_dict[index][i][1]
#                     if predict_word != gold_word:
#                         print('problem between prediction word: {0} and test word {1} indexes : {2}'
#                               .format(predict_word, gold_word, str((index, i))))
#                     confusion_matrix_key = "{0}_{1}".format(gold_tag, predict_tag)
#                     if gold_tag not in self.tags:
#                         self.add_missing_tags(gold_tag, predict_tag)
#                     self.confusion_matrix[confusion_matrix_key] += 1
#                     if predict_tag != gold_tag:  # tag miss
#                         miss += 1
#                         self.misses_matrix[confusion_matrix_key] += 1
#                     else:
#                         hit += 1
#
#         print('Misses: {0}, Hits: {1}'.format(miss, hit))
#         accuracy = 100 * float(hit) / float(miss + hit)
#         print('Model Accuracy: {:.2f}%'.format(accuracy))
#         self.eval_res.update({'Full': {'miss': miss, 'hit': hit, 'accuracy': accuracy}})
#
#         for unseen_word in self.viterbi_unseen_words:
#             sentence_idx, word_idx = unseen_word
#             gold_word, gold_tag = word_tag_tuples_dict[sentence_idx][word_idx]
#             predict_word, predict_tag = self.predict_dict[sentence_idx][word_idx].split('_')
#             if gold_word != predict_word:
#                 print('problem between prediction word: {0} and test word {1} indexes : {2}'
#                       .format(predict_word, gold_word, str((sentence_idx, word_idx))))
#             self.unseen_tags_set.update((gold_tag, predict_tag))
#             keys = self.get_all_possible_tags(gold_tag, predict_tag)
#             for key in keys:
#                 self.unseen_confusion_matrix.setdefault(key, 0)
#             confusion_matrix_key = "{0}_{1}".format(gold_tag, predict_tag)
#             self.unseen_confusion_matrix[confusion_matrix_key] += 1
#             if predict_tag != gold_tag:
#                 miss_unseen += 1
#             else:
#                 hit_unseen += 1
#         print('Unseen Confusion')
#         print('Misses: {0}, Hits: {1}'.format(miss_unseen, hit_unseen))
#         accuracy = 0
#         if miss_unseen + hit_unseen > 0:
#             accuracy = 100 * float(hit_unseen) / float(miss_unseen + hit_unseen)
#         print('Model Accuracy: {:.2f}%'.format(accuracy))
#         self.eval_res.update({'unseen words': {'miss': miss_unseen, 'hit': hit_unseen, 'accuracy': accuracy}})
#
#         unseen_tag_list = sorted(self.unseen_tags_set)
#
#         keys_set = set()
#         for i in range(len(unseen_tag_list)):
#             for j in range(i, len(unseen_tag_list)):
#                 keys = self.get_all_possible_tags(unseen_tag_list[i], unseen_tag_list[j])
#                 keys_set.update(keys)
#         for key in keys_set:
#             self.unseen_confusion_matrix.setdefault(key, 0)
#
#         miss_seen = miss - miss_unseen
#         hit_seen = hit - hit_unseen
#         accuracy = 0
#         if miss_seen + hit_seen > 0:
#             accuracy = 100 * float(hit_seen) / float(miss_seen + hit_seen)
#         print("Seen Confusion")
#         print('Misses: {0}, Hits: {1}'.format(miss_seen, hit_seen))
#         print('Model Accuracy: {:.2f}%'.format(accuracy))
#         self.eval_res.update({'seen words': {'miss': miss_seen, 'hit': hit_seen, 'accuracy': accuracy}})
#
#         return \
#             {
#                 'Missse': miss,
#                 'Hits': hit,
#                 'Accuracy': float(hit) / float(miss + hit),
#                 'Unseen words Misses': miss_unseen,
#                 'Unseen words Hits': hit_unseen,
#                 'Unseen words Accuracy': float(hit_unseen) / float(miss_unseen + hit_unseen)
#                 # ,'confusion_matrix per word': self.confusion_matrix
#             }
#
#     def write_result_doc(self):
#
#         file_name = self.write_file_name
#         lines_count = len(self.predict_dict)
#         with open(file_name, 'w') as f:
#             for sentence_index, sequence_list in self.predict_dict.items():
#                 sentence_len = len(sequence_list)
#                 for word_index, word_tag_string in enumerate(sequence_list):
#                     sep = ' '
#                     if word_index + 1 == sentence_len:  # EOL
#                         if sentence_index + 1 < lines_count:  # if EOL but not EOF, add \n
#                             sep = '\n'
#                         else:
#                             sep = ''
#                     f.write("{0}{1}".format(word_tag_string, sep))
#         return
#
#     def write_confusion_doc(self):
#         """
#             build confusion matrix doc
#             build structure of line and columns
#         """
#
#         file_name = self.confusion_file_name
#         confusion_matrix_to_write = self.confusion_matrix
#
#         book = xlwt.Workbook(encoding="utf-8")
#
#         # confusion matrix
#         self.create_confusion_sheet(book, self.tags, confusion_matrix_to_write, "Confusion Matrix")
#
#         top_k_tags_set, confusion_matrix_to_write = self.get_most_missed_tags()
#         # top-K confusion matrix
#         self.create_confusion_sheet(book, top_k_tags_set, confusion_matrix_to_write, "Top-{} Confusion Matrix"
#                                     .format(self.k))
#         # unseen confusion matrix
#         unseen_tags_list = sorted(self.unseen_tags_set)
#         confusion_matrix_to_write = self.unseen_confusion_matrix
#         self.create_confusion_sheet(book, unseen_tags_list, confusion_matrix_to_write, "Unseen Confusion Matrix")
#         book.save(file_name)
#
#     def create_confusion_sheet(self, book, tag_list, confusion_matrix_to_write, sheet_name):
#         """

#
#     def create_summary_file(self, lamda, model_features, test_file, train_file,
#                             summary_file_name, weight_file_name, comp):
#         """
#         this method is creating a summary file of the run
#         :param weight_file_name: the location of the weights vector file
#         :param model_features: the features that where used in this model
#         :param lamda: lambda value
#         :param test_file: test data location
#         :param train_file: train data location
#         :param summary_file_name: where to save the summary
#         :return: None
#         """
#         with open(summary_file_name, "w") as summary_file:
#             summary = csv.writer(summary_file)
#             summary.writerow(['Running time:', datetime.now().ctime()])
#             summary.writerow(['Model features:'])
#             summary.writerow(model_features)
#             summary.writerow(['Model test file:', test_file])
#             summary.writerow(['Model train file:', train_file])
#             summary.writerow(['Model lambda:', lamda])
#             summary.writerow(['Predicted doc:', self.write_file_name])
#             summary.writerow(['Weight file name:', weight_file_name])
#             if not comp:
#                 summary.writerow(['Confusion matrix:', self.confusion_file_name])
#                 for eval_type, eval_res in self.eval_res.items():
#                     summary.writerow(['{} Results:'.format(eval_type)])
#                     summary.writerow(['Misses: {} '.format(eval_res['miss']), 'Hits: {}'.format(eval_res['hit'])])
#                     summary.writerow(['Model Accuracy: {:.2f}%'.format(eval_res['accuracy'])])
#                 summary.writerow(['Most missed Tags combinations:'])
#                 for tag_name, miss_val in self.most_misses_tags.items():
#                     gold, predicted = tag_name.split('_')
#                     summary.writerow(['real tag: {}'.format(gold),
#                                       'predicted tag: {}'.format(predicted), "count: {}".format(miss_val)])
#         return