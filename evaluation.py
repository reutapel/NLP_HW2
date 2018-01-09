import csv
import itertools
from datetime import datetime
import xlwt
from collections import defaultdict
import logging
import time
import os


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
        print('building class Evaluate instance')
        logging.info('building class Evaluate instance')

        self.model = model
        self.inference_obj = inference_obj
        self.gold_tree = None
        self.token_POS_dict = None
        self.inference_mode = None
        self.directory = os.path.join(directory + 'evaluations\\')

    def update_inference_mode(self, inference_mode):

        """
        :param inference_mode: for updates the class variables to the correct work mode
        :return:
        """

        self.inference_mode = inference_mode
        if self.inference_mode == 'train':
            self.gold_tree = self.model.train_gold_tree
            self.token_POS_dict = self.model.train_token_POS_dict
            print('Evaluation updated to train mode')
            logging.info('Evaluation updated to train mode')
        elif self.inference_mode == 'test':
            self.gold_tree = self.model.test_gold_tree
            self.token_POS_dict = self.model.test_token_POS_dict
            print('Evaluation updated to test mode')
            logging.info('Evaluation updated to test mode')
        else:
            self.gold_tree = self.model.comp_gold_tree
            print('Evaluation updated to comp mode')
            logging.info('Evaluation updated to comp mode')
        return

    def calculate_accuracy(self, inference_mode=None):

        """
        :param inference_mode: for updates the class variables to the correct work mode
        :return:
        """

        if inference_mode == 'comp':
            print('can not calculate accuracy for non train/test mode')
            logging.info('can not calculate accuracy for non train/test mode')
            return

        # change relevant class variables
        self.update_inference_mode(inference_mode)

        # change perceptron to train/test mode, should influence it's gold tree to be train/test gold tree,
        # and function edge score to train/test mode.
        self.inference_obj.inference_mode(self.inference_mode)

        data_mistake_num = 0
        data_num_tokens = len(self.token_POS_dict['token'].keys())
        sentences_count = 0
        mistakes_dict = dict()

        print('start calculating accuracy')
        logging.info('start calculating accuracy')
        for t in range(len(self.gold_tree)):
            sentence_mistake_num = 0
            gold_sentence = self.gold_tree[t]
            pred_tree = self.inference_obj.calculate_mst(t)
            for source, targets in gold_sentence.items():  # todo: Rom added parentheses
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
                    # mistakes_dict[t][source][['missed_targets']] = missed_targets  # todo: Rom removed
                    mistakes_dict[t][source] = dict()
                    mistakes_dict[t][source]['missed_targets'] = missed_targets
                    mistakes_dict[t][source]['wrong_targets'] = wrong_targets
            sentences_count += 1
        accuracy = 1 - data_mistake_num / (data_num_tokens - sentences_count)
        # todo: Rom changed format
        logging.info('{}: Accuracy for {} is : {:%} '.format(time.asctime(time.localtime(time.time())),
                                                             self.inference_mode, accuracy))
        print('{}: accuracy for {} is: {:%} '.format(time.asctime(time.localtime(time.time())), self.inference_mode,
                                                     accuracy))

        print('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        mistakes_dict_name = 'accuracy_{}_mistakes_dict_{}_{}'.format(accuracy, self.inference_mode,
                                                                      time.asctime(time.localtime(time.time())).replace(
                                                                          ' ', '_').replace(':', '_'))
        w = csv.writer(open(self.directory + mistakes_dict_name + '.csv', "w"))
        for key, val in mistakes_dict.items():
            w.writerow([key, val])
        print('{}: finished saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: finished saving mistakes_dict'.format(time.asctime(time.localtime(time.time()))))

        return accuracy, mistakes_dict_name

    def infer(self, comp_file_name, inference_mode=None):

        """
        :param comp_file_name: if evaluation class is for inference on competition data
        :param inference_mode: for updates the class variables to the correct work mode
        """

        # change relevant class variables
        self.update_inference_mode(inference_mode)

        # change perceptron to comp mode in order to only calculate mst, using comp gold tree and edge score comp
        self.inference_obj.inference_mode(inference_mode)
        comp_file_name = comp_file_name

        return

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
#         this method creates a new confusion matrix sheet by the name sheet_name
#         :param sheet_name:
#         :param book: the excel workbook object
#         :param tag_list: list of all the tags
#         :param confusion_matrix_to_write:
#         :return: None
#         """
#         sheet1 = book.add_sheet(sheet_name)
#
#         # Header pattern
#         header_pattern = xlwt.Pattern()
#         header_pattern.pattern = xlwt.Pattern.SOLID_PATTERN
#         header_pattern.pattern_fore_colour = 9
#         bold_font = xlwt.Font()
#         bold_font.bold = True
#         align = xlwt.Alignment()
#         align.horz = xlwt.Alignment.HORZ_CENTER
#         thick_border = xlwt.Borders()
#         thick_border.right = xlwt.Borders.THICK
#         thick_border.left = xlwt.Borders.THICK
#         thick_border.top = xlwt.Borders.THICK
#         thick_border.bottom = xlwt.Borders.THICK
#         header_style = xlwt.XFStyle()
#         header_style.pattern = header_pattern
#         header_style.borders = thick_border
#         header_style.font = bold_font
#         header_style.alignment = align
#
#         # Regualr pattern
#         reg_border = xlwt.Borders()
#         reg_border.right = xlwt.Borders.DASHED
#         reg_border.left = xlwt.Borders.DASHED
#         reg_border.top = xlwt.Borders.DASHED
#         reg_border.bottom = xlwt.Borders.DASHED
#         style = xlwt.XFStyle()
#         style.borders = reg_border
#         style.num_format_str = '0'
#         style.alignment = align
#
#         # mistakes pattern
#         pattern_mistake = xlwt.Pattern()
#         pattern_mistake.pattern = xlwt.Pattern.SOLID_PATTERN
#         pattern_mistake.pattern_fore_colour = 29
#         style_mistake = xlwt.XFStyle()
#         style_mistake.pattern = pattern_mistake
#         style_mistake.num_format_str = '0'
#         style_mistake.borders = reg_border
#         style_mistake.alignment = align
#
#         # correct pattern
#         pattern_hit = xlwt.Pattern()
#         pattern_hit.pattern = xlwt.Pattern.SOLID_PATTERN
#         pattern_hit.pattern_fore_colour = 42
#         style_hit = xlwt.XFStyle()
#         style_hit.pattern = pattern_hit
#         style_hit.num_format_str = '0'
#         style_hit.borders = reg_border
#         style_hit.alignment = align
#
#         # sum pattern
#         pattern_sum = xlwt.Pattern()
#         pattern_sum.pattern = xlwt.Pattern.SOLID_PATTERN
#         pattern_sum.pattern_fore_colour = 22
#         style_sum = xlwt.XFStyle()
#         style_sum.pattern = pattern_sum
#         style_sum.num_format_str = '0'
#         style_sum.borders = thick_border
#         style_sum.font = bold_font
#         style_sum.alignment = align
#
#         # FP pattern
#         style_fp = xlwt.XFStyle()
#         style_fp.pattern = pattern_sum
#         style_fp.num_format_str = '0.00%'
#         style_fp.borders = thick_border
#         style_fp.font = bold_font
#         style_fp.alignment = align
#
#         last_pos = len(tag_list) + 1
#         sheet1.write(0, 0, ' ', header_style)
#
#         for idx_tag, cur_tag in enumerate(tag_list):
#             sheet1.write(0, idx_tag + 1, cur_tag, header_style)
#         sheet1.write(0, last_pos, 'Recall', header_style)
#         sheet1.write(0, last_pos + 1, 'Total', header_style)
#         col_count_hit = [0] * len(tag_list)
#         col_count_miss = [0] * len(tag_list)
#         for row_tag_idx, row_tag in enumerate(tag_list):
#             row_count_hit = 0
#             row_count_miss = 0
#             sheet1.write(row_tag_idx + 1, 0, row_tag, header_style)
#             for col_tag_idx, col_tag in enumerate(tag_list):
#                 cur_value = confusion_matrix_to_write["{0}_{1}".format(row_tag, col_tag)]
#                 if cur_value == 0:
#                     sheet1.write(row_tag_idx + 1, col_tag_idx + 1, cur_value, style)
#                 else:
#                     if row_tag_idx == col_tag_idx:
#                         sheet1.write(row_tag_idx + 1, col_tag_idx + 1, cur_value, style_hit)
#                         row_count_hit += cur_value
#                         col_count_hit[col_tag_idx] += cur_value
#                     else:
#                         sheet1.write(row_tag_idx + 1, col_tag_idx + 1, cur_value, style_mistake)
#                         row_count_miss += cur_value
#                         col_count_miss[col_tag_idx] += cur_value
#             row_count = row_count_hit + row_count_miss
#             if row_count == 0:
#                 sheet1.write(row_tag_idx + 1, last_pos, row_count, style_fp)  # recall
#             else:
#                 sheet1.write(row_tag_idx + 1, last_pos, row_count_hit / row_count, style_fp)  # recall
#             sheet1.write(row_tag_idx + 1, last_pos + 1, row_count, style_sum)  # total
#         sheet1.write(last_pos, 0, 'Precision', header_style)
#         sheet1.write(last_pos + 1, 0, 'Total', header_style)
#         total_count = 0
#         total_hit = 0
#         for col_idx, col_hit in enumerate(col_count_hit):
#             col_count = col_hit + col_count_miss[col_idx]
#             if col_count == 0:
#                 sheet1.write(last_pos, col_idx + 1, col_count, style_fp)  # recall
#             else:
#                 sheet1.write(last_pos, col_idx + 1, col_hit / col_count, style_fp)  # recall
#             total_count += col_count
#             total_hit += col_hit
#             sheet1.write(last_pos + 1, col_idx + 1, col_count, style_sum)
#         sheet1.write(last_pos, last_pos, total_hit / total_count, style_fp)
#         sheet1.write(last_pos, last_pos + 1, ':Accuracy', style)
#         return
#
#     def get_most_missed_tags(self):
#         top_tags_list = sorted(self.misses_matrix.items(), key=lambda x: x[1], reverse=True)[:self.k]
#         tag_set = set()
#         top_k_confusion_matrix = {}
#         tags_keys = set()
#         for key, val in top_tags_list:
#             self.most_misses_tags.update({key: val})
#             gold, predict = key.split('_')
#             tag_set.update((gold, predict))
#         tag_set = sorted(tag_set)
#         # todo: check whether we can cut the loops
#         for i in range(len(tag_set)):
#             for j in range(i, len(tag_set)):
#                 keys = self.get_all_possible_tags(tag_set[i], tag_set[j])
#                 tags_keys.update(keys)
#         for key in tags_keys:
#             value = self.confusion_matrix.get(key, 0)
#             top_k_confusion_matrix.update({key: value})
#         return tag_set, top_k_confusion_matrix
#
#     def get_all_possible_tags(self, gold, predict):
#         """
#         this method generates all possible combination of a given two tags gold and predict
#         :param gold: first tag
#         :param predict: second tag
#         :return:  a set of tags
#         """
#         keys = []
#         for tag_1, tag_2 in itertools.product([gold, predict], repeat=2):
#             keys.append("{}_{}".format(tag_1, tag_2))
#         return keys
#
#     def add_missing_tags(self, gold_tag, predict_tag):
#         res = self.get_all_possible_tags(gold_tag, predict_tag)
#         for confusion_matrix_key in res:
#             self.confusion_matrix.setdefault(confusion_matrix_key, 0)
#             self.misses_matrix.setdefault(confusion_matrix_key, 0)
#         return
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
