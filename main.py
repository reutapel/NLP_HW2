import time

import os
from sklearn.model_selection import KFold
import logging
from datetime import datetime
from struct_perceptron import StructPerceptron
from parser_model import ParserModel
from evaluation import Evaluate
from copy import copy
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import math

# open log connection
sub_dirs = ["logs", "evaluations", "dict", "weights"]
base_directory = os.path.abspath(os.curdir)
run_dir = datetime.now().strftime("advanced_stepwise_%d_%m_%Y_%H_%M_%S")
directory = os.path.join(base_directory, "output", run_dir)
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(directory, sub_dir))
directory += os.sep
LOG_FILENAME = datetime.now().strftime(os.path.join(directory, 'logs', 'LogFile_%d_%m_%Y_%H_%M.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def cross_validation(features_dict, train_file_to_use, test_file_to_use, comp_file_to_use, number_of_iter,
                     number_of_sentence_train, is_cv=True):
    # Cross validation part 1: split the data to folds
    kf = KFold(n_splits=5, shuffle=True)
    split_list_cv = kf.split(range(number_of_sentence_train))
    number_sentences_to_choose = math.floor(number_of_sentence_train*0.25)
    test_index_np = np.random.choice(number_of_sentence_train, number_sentences_to_choose, replace=False)
    all_indexes = np.arange(number_of_sentence_train)
    train_index_np = np.setdiff1d(all_indexes, test_index_np)
    split_list = [train_index_np, test_index_np]
    features = list(features_dict.values())
    features = features[0][0]

    cv_start_time = time.time()
    logging.info('{}: Start running stepwise with CV={}'.format(time.asctime(time.localtime(time.time())), is_cv))
    print('{}: Start running stepwise with CV={}'.format(time.asctime(time.localtime(time.time())), is_cv))

    selected_features = copy(features)
    remaining_features = copy(features)
    remain_number_of_candidate = copy(len(remaining_features))
    # run with all features
    accuracy = main(train_file_to_use, test_file_to_use, comp_file_to_use, 'test',
                    [features], number_of_iter, comp=False, train_index=train_index_np,
                    test_index=test_index_np, best_weights_list=None)
    current_acc, new_acc = accuracy, accuracy
    while remaining_features and current_acc == new_acc and remain_number_of_candidate > 1:
        acc_with_candidates = list()
        for candidate in remaining_features:
            features_to_run = copy(remaining_features)
            features_to_run.remove(candidate)
            k = 0
            candidate_acc_list = list()

            for train_index, test_index in split_list_cv:
                if not is_cv:  # do not run CV
                    train_index, test_index = split_list[0], split_list[1]
                logging.info('{}: Start running fold number {}'.format(time.asctime(time.localtime(time.time())), k))
                print('{}: Start running fold number {}'.format(time.asctime(time.localtime(time.time())), k))
                accuracy = main(train_file_to_use, test_file_to_use, comp_file_to_use, 'test',
                                [features_to_run], number_of_iter, comp=False, train_index=train_index,
                                test_index=test_index, best_weights_list=None)
                candidate_acc_list.append(accuracy)

                run_time_cv = (time.time() - cv_start_time) / 60.0
                print("{}: Finish running iteration {}. Run time is: {} minutes".
                      format(time.asctime(time.localtime(time.time())), k, run_time_cv))
                logging.info('{}: Finish running iteration {}. Run time is: {} minutes'.
                             format(time.asctime(time.localtime(time.time())), k, run_time_cv))
                k += 1
                if not is_cv and k == 1:
                    break
            # the average accuracy of the CV for the features we test
            acc_with_candidates.append((sum(candidate_acc_list)/float(len(candidate_acc_list)), candidate))

        # after testing all possible candidate we want to remove - find the one that the model without it got the
        # highest accuracy
        acc_with_candidates.sort()
        new_acc, best_candidate = acc_with_candidates.pop()
        if current_acc <= new_acc:
            selected_features.remove(best_candidate)
            remaining_features.remove(best_candidate)
            current_acc = new_acc
            logging.info('{}: Selected features are: {} and the best accuracy is: {}'.
                         format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))
            print('{}: Selected features are: {} and the best accuracy is: {}'.
                  format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))

        else:
            logging.info('{}: No candidate was chosen. Number of selected features is {}.'.
                         format((time.asctime(time.localtime(time.time()))), len(selected_features)))
            print('{}: No candidate was chosen. Number of selected features is {}.'.
                  format((time.asctime(time.localtime(time.time()))), len(selected_features)))
            logging.info('{}: Selected features are: {} and the best accuracy is: {}'.
                         format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))
            print('{}: Selected features are: {} and the best accuracy is: {}'.
                  format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))

        # one candidate can be chosen, if not- we go to the next step.
        remain_number_of_candidate -= 1

    logging.info('{}: Selected features are: {} and the best accuracy is: {}'.
                 format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))
    print('{}: Selected features are: {} and the best accuracy is: {}'.
          format((time.asctime(time.localtime(time.time()))), selected_features, new_acc))


def main(train_file_to_use, test_file_to_use, comp_file_to_use, test_type, features_combination_list, number_of_iter,
         comp, train_index=None, test_index=None, best_weights_list=None):

    # start all combination of features
    for features_combination in features_combination_list:
        # Create features for train and test gold trees
        print('{}: Start creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                         features_combination))
        logging.info('{}: Start creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                                features_combination))
        train_start_time = time.time()
        parser_model_obj = ParserModel(directory, train_file_to_use, test_file_to_use, comp_file_to_use,
                                       features_combination, use_edges_existed_on_train, use_pos_edges_existed_on_train,
                                       train_index=train_index, test_index=test_index)

        model_finish_time = time.time()
        model_run_time = (model_finish_time - train_start_time) / 60.0
        print('{}: Finish creating parser model for features : {} in {} minutes'.
              format(time.asctime(time.localtime(time.time())), features_combination, model_run_time))
        logging.info('{}: Finish creating parser model for features : {} in {} minutes'
                     .format(time.asctime(time.localtime(time.time())), features_combination, model_run_time))

        # Run perceptron to learn the best weights
        print('{}: Start Perceptron for features : {} and number of iterations: {}'.
              format(time.asctime(time.localtime(time.time())), features_combination, number_of_iter))
        logging.info('{}: Start Perceptron for features : {} and number of iterations: {}'.
                     format(time.asctime(time.localtime(time.time())), features_combination, number_of_iter))
        perceptron_obj = StructPerceptron(model=parser_model_obj, directory=directory,
                                          feature_combination=features_combination)
        weights = perceptron_obj.perceptron(num_of_iter=number_of_iter)

        train_run_time = (time.time() - model_finish_time) / 60.0
        print('{}: Finish Perceptron for features : {} and num_of_iter: {}. run time: {} minutes'.
              format(time.asctime(time.localtime(time.time())), features_combination, number_of_iter, train_run_time))
        logging.info('{}: Finish Perceptron for features : {} and num_of_iter: {}. run time: {} minutes'.
                     format(time.asctime(time.localtime(time.time())), features_combination, number_of_iter,
                            train_run_time))

        evaluate_obj = Evaluate(parser_model_obj, perceptron_obj, directory)
        best_weights_name = str()
        if test_type != 'comp':
            weights_directory = perceptron_obj.directory
            weight_file_names = [f for f in listdir(weights_directory) if isfile(join(weights_directory, f))]
            accuracy = dict()
            mistakes_dict_names = dict()
            for weights in weight_file_names:
                with open(os.path.join(weights_directory, weights), 'rb') as fp:
                    weight_vec = pickle.load(fp)
                weights = weights[:-4]
                if train_index is not None and weights != 'final_weight_vec_20':
                    continue
                accuracy[weights], mistakes_dict_names[weights] = evaluate_obj.calculate_accuracy(weight_vec,
                                                                                                  weights, test_type)
            print('{}: The model hyper parameters and results are: \n num_of_iter: {} \n test file: {} \n'
                  'train file: {} \n test type: {} \n features combination list: {} \n accuracy: {:%} \n'
                  'mistakes dict name: {}'
                  .format(time.asctime(time.localtime(time.time())), number_of_iter, test_file_to_use,
                          train_file_to_use, test_type, features_combination_list, accuracy[weights],
                          mistakes_dict_names[weights]))
            logging.info('{}: The model hyper parameters and results are: \n num_of_iter: {} \n test file: {}'
                         '\n train file: {} \n test type: {} \n features combination list: {} \n accuracy: {} \n'
                         'mistakes dict name: {}'
                         .format(time.asctime(time.localtime(time.time())), number_of_iter, test_file_to_use,
                                 train_file_to_use, test_type, features_combination_list, accuracy[weights],
                                 mistakes_dict_names[weights]))

            # get the weights that gave the best accuracy and save as best weights
            best_weights = max(accuracy, key=accuracy.get)
            with open(os.path.join(weights_directory, best_weights + '.pkl'), 'rb') as fp:
                best_weights_vec = pickle.load(fp)
            best_weights_name = os.path.join(weights_directory, "best_weights_" + best_weights + '.pkl')
            with open(best_weights_name, 'wb') as f:
                pickle.dump(best_weights_vec, f)

            if train_index is not None:  # running CV
                return accuracy['final_weight_vec_20']

            logging.info('{}: best weights for {}, {}, {}, with accuracy {}, name is: {} '
                         .format(time.asctime(time.localtime(time.time())), num_of_iter, test_type,
                                 features_combination_list, accuracy[best_weights],best_weights_name))
            print('{}: best weights for {}, {}, {}, with accuracy {}, name is: {} '
                  .format(time.asctime(time.localtime(time.time())), num_of_iter, test_type,
                          features_combination_list, accuracy[best_weights], best_weights_name))
    if comp:
        for best_weights_vec_loaded in best_weights_list:
            inference_file_name = evaluate_obj.infer(best_weights_vec_loaded, test_type)
            print('{}: The inferred file name is: {} for weights: {} '.format(time.asctime(time.localtime
                                                                                           (time.time())),
                                                                              inference_file_name, best_weights_vec_loaded))
            logging.info('{}: The inferred file name is: {} for weights: {} '.format(time.asctime(
                time.localtime(time.time())), inference_file_name, best_weights_vec_loaded))

    logging.info('-----------------------------------------------------------------------------------')

    return


if __name__ == "__main__":
    logging.info('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    print('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    train_file = os.path.join(base_directory, 'HW2-files', 'train_small.labeled')
    test_file = os.path.join(base_directory, 'HW2-files', 'test_small.labeled')
    comp_file = os.path.join(base_directory, 'HW2-files', 'comp.unlabeled')
    # change name to chosen weights for running comp inference
    best_weights_vec_loaded_basic = os.path.join(base_directory, 'output',
                                                 'advanced_model_5080100_iter_20_01_2018_00_23_28', 'weights',
                                                 'best_weights_final_weight_vec_50.pkl')
    best_weights_vec_loaded_advanced = os.path.join(base_directory, 'output',
                                                    'advanced_model_5080100_iter_20_01_2018_00_23_28', 'weights',
                                                    'best_weights_final_weight_vec_80.pkl')
    best_weights_list = [best_weights_vec_loaded_basic, best_weights_vec_loaded_advanced]

    advanced_features = range(1, 31)
    advanced_features = [str(i) for i in advanced_features]
    basic_features = range(1, 14)
    basic_features = [str(i) for i in basic_features]
    basic_features.remove('7')
    basic_features.remove('9')
    basic_features.remove('11')
    basic_features.remove('12')
    feature_type_dict = {
        'all_features': [advanced_features]}
        # 'basic_model': [basic_features]}

    num_of_iter_list = [100]
    cv = False
    stepwise = True
    comp = False
    use_edges_existed_on_train, use_pos_edges_existed_on_train = True, True
    if cv:
        # if running with all train data: number_of_sentence_train = 5000, else: put the number ot sentences
        # you have in the small train you run
        cross_validation(feature_type_dict, train_file, test_file, comp_file, number_of_iter=20,
                         number_of_sentence_train=5000)
    elif stepwise:
        cross_validation(feature_type_dict, train_file, test_file, comp_file, number_of_iter=20,
                         number_of_sentence_train=5000, is_cv=False)
    else:
        for num_of_iter in num_of_iter_list:
            start_time = time.time()
            if not comp:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'test', feature_type_list, num_of_iter, comp,
                         train_index=None, test_index=None)
            else:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'comp', feature_type_list, num_of_iter, comp,
                         train_index=None, test_index=None, best_weights_list=best_weights_list)
            run_time = (time.time() - start_time) / 60.0
            print("{}: Finish running with num_of_iter: {}. Run time is: {} minutes".
                  format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))
            logging.info('{}: Finish running with num_of_iter:{} . Run time is: {} minutes'.
                         format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))

