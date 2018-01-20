import time

import os
from sklearn.model_selection import KFold
import logging
from datetime import datetime
from struct_perceptron import StructPerceptron
from parser_model import ParserModel
from evaluation import Evaluate
from os import listdir
from os.path import isfile, join
import pickle

# open log connection
sub_dirs = ["logs", "evaluations", "dict", "weights"]
base_directory = os.path.abspath(os.curdir)
run_dir = datetime.now().strftime("advanced_model_5080100_iter_%d_%m_%Y_%H_%M_%S")
directory = os.path.join(base_directory, "output", run_dir)
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(directory, sub_dir))
directory += os.sep
LOG_FILENAME = datetime.now().strftime(os.path.join(directory, 'logs', 'LogFile_%d_%m_%Y_%H_%M.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


def cross_validation(train_file_for_cv):
    # Cross validation part 1: split the data to folds
    text_file = open(train_file_for_cv, 'r')
    train_data = text_file.read().split('\n')
    kf = KFold(n_splits=5, shuffle=True)

    lambda_list = [100.0]
    for lamda in lambda_list:
        CV_start_time = time.time()
        logging.info('{}: Start running 5-fold CV for lambda: {}'.format(time.asctime(time.localtime(time.time())),
                                                                          lamda))
        print('{}: Start running 5-fold CV for lambda: {}'.format(time.asctime(time.localtime(time.time())), lamda))
        k = 0

        for train_index, test_index in kf.split(train_data):
            # Create the train and test data according to the folds, and save the new data
            train_k_fold = list(train_data[i] for i in train_index)
            test_k_fold = list(train_data[i] for i in test_index)
            train_file_cv = datetime.now().strftime(directory + 'data/train_cv_file_%d_%m_%Y_%H_%M.wtag')
            test_file_cv = datetime.now().strftime(directory + 'data/test_cv_file_%d_%m_%Y_%H_%M.wtag')
            with open(train_file_cv, 'w', newline='\n') as file:
                for sentence in train_k_fold:
                    file.write(str(sentence) + '\n')
            with open(test_file_cv, 'w', newline='\n') as file:
                for sentence in test_k_fold:
                    file.write(str(sentence) + '\n')

            advanced_features = range(1, 19)
            advanced_features = [str(i) for i in advanced_features]
            basic_features = range(1, 14)
            basic_features = [str(i) for i in basic_features]
            feature_type_dict_cv = {
                'all_features': [advanced_features],
                'basic_model': [basic_features]}

            for feature_type_name_cv, feature_type_list_cv in feature_type_dict_cv.items():
                logging.info('{}: Start running fold number {} for lambda: {}'.
                             format(time.asctime(time.localtime(time.time())), k, lamda))
                print('{}: Start running fold number {} for lambda: {}'
                      .format(time.asctime(time.localtime(time.time())), k, lamda))
                main(train_file_cv, test_file_cv, 'test_cv_fold_' + str(k), feature_type_list_cv, lamda, comp=False)

            run_time_cv = (time.time() - CV_start_time) / 60.0
            print("{}: Finish running iteration {} of 10-fold CV for lambda: {}. Run time is: {} minutes".
                  format(time.asctime(time.localtime(time.time())), k, lamda, run_time_cv))
            logging.info('{}: Finish running iteration {} 10-fold CV for lambda:{} . Run time is: {} minutes'.
                         format(time.asctime(time.localtime(time.time())), k, lamda, run_time_cv))
            k += 1


def main(train_file_to_use, test_file_to_use, comp_file_to_use, test_type, features_combination_list, num_of_iter, comp,
         best_weights_list = None):

    # start all combination of features
    for features_combination in features_combination_list:
        # Create features for train and test gold trees
        print('{}: Start creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                         features_combination))
        logging.info('{}: Start creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                                features_combination))
        train_start_time = time.time()
        parser_model_obj = ParserModel(directory, train_file_to_use, test_file_to_use, comp_file_to_use,
                                       features_combination)

        model_finish_time = time.time()
        model_run_time = (model_finish_time - train_start_time) / 60.0
        print('{}: Finish creating parser model for features : {} in {} minutes'.
              format(time.asctime(time.localtime(time.time())), features_combination, model_run_time))
        logging.info('{}: Finish creating parser model for features : {} in {} minutes'
                     .format(time.asctime(time.localtime(time.time())), features_combination, model_run_time))

        # Run perceptron to learn the best weights
        print('{}: Start Perceptron for features : {} and number of iterations: {}'.
              format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter))
        logging.info('{}: Start Perceptron for features : {} and number of iterations: {}'.
                     format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter))
        perceptron_obj = StructPerceptron(model=parser_model_obj, directory=directory)
        weights = perceptron_obj.perceptron(num_of_iter=num_of_iter)

        train_run_time = (time.time() - model_finish_time) / 60.0
        print('{}: Finish Perceptron for features : {} and num_of_iter: {}. run time: {} minutes'.
              format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter, train_run_time))
        logging.info('{}: Finish Perceptron for features : {} and num_of_iter: {}. run time: {} minutes'.
                     format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter, train_run_time))

        evaluate_obj = Evaluate(parser_model_obj, perceptron_obj, directory)
        best_weights_name = str()
        if test_type != 'comp':
            weights_directory = os.path.join(directory, 'weights')
            weight_file_names = [f for f in listdir(weights_directory) if isfile(join(weights_directory, f))]
            accuracy = dict()
            mistakes_dict_names = dict()
            for weights in weight_file_names:
                with open(weights_directory + '\\' + weights, 'rb') as fp:
                    weight_vec = pickle.load(fp)
                weights = weights[:-4]
                accuracy[weights], mistakes_dict_names[weights] = evaluate_obj.calculate_accuracy(weight_vec,
                                                                                                  weights, test_type)
            print('{}: The model hyper parameters and results are: \n num_of_iter: {} \n test file: {} \n train file: {} '
                  '\n test type: {} \n features combination list: {} \n accuracy: {:%} \n mistakes dict name: {}'
                  .format(time.asctime(time.localtime(time.time())), num_of_iter, test_file_to_use, train_file_to_use,
                          test_type, features_combination_list, accuracy[weights], mistakes_dict_names[weights]))
            logging.info('{}: The model hyper parameters and results are: \n num_of_iter: {} \n test file: {}'
                         '\n train file: {} \n test type: {} \n features combination list: {} \n accuracy: {} \n'
                         'mistakes dict name: {}'
                         .format(time.asctime(time.localtime(time.time())), num_of_iter, test_file_to_use,
                                 train_file_to_use, test_type, features_combination_list, accuracy[weights],
                                 mistakes_dict_names[weights]))

            # get the weights that gave the best accuracy and save as best weights
            best_weights = max(accuracy, key=accuracy.get)
            with open(weights_directory + '\\' + best_weights + '.pkl', 'rb') as fp:
                best_weights_vec = pickle.load(fp)
            best_weights_name = weights_directory + '\\' + "best_weights_" + best_weights + '.pkl'
            with open(best_weights_name, 'wb') as f:
                pickle.dump(best_weights_vec, f)

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


if __name__ == "__main__":
    logging.info('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    print('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    train_file = os.path.join(base_directory, 'HW2-files', 'train.labeled')
    test_file = os.path.join(base_directory, 'HW2-files', 'test.labeled')
    comp_file = os.path.join(base_directory, 'HW2-files', 'comp.unlabeled')
    # change name to chosen weights
    best_weights_vec_loaded_basic = os.path.join(base_directory, 'output', 'advanced_model_5080100_iter_19_01_2018_15_08_00',
                                           'weights', 'best_weights_final_weight_vec_100.pkl')
    best_weights_vec_loaded_advanced = os.path.join(base_directory, 'output', 'advanced_model_5080100_iter_19_01_2018_15_08_00',
                                           'weights', 'best_weights_final_weight_vec_80.pkl')
    best_weights_list = [best_weights_vec_loaded_basic, best_weights_vec_loaded_advanced]
    cv = False
    comp = False
    if cv:
        cross_validation(train_file)
    else:
        advanced_features = range(1, 27)
        advanced_features = [str(i) for i in advanced_features]
        basic_features = range(1, 14)
        basic_features = [str(i) for i in basic_features]
        basic_features.remove('7')
        basic_features.remove('9')
        basic_features.remove('11')
        basic_features.remove('12')
        feature_type_dict = {
            'all_features': [advanced_features],
             'basic_model': [basic_features]}

        num_of_iter_list = [100]
        for num_of_iter in num_of_iter_list:
            start_time = time.time()
            if not comp:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'test', feature_type_list, num_of_iter, comp)
            else:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'comp', feature_type_list, num_of_iter, comp,
                         best_weights_list)
            run_time = (time.time() - start_time) / 60.0
            print("{}: Finish running with num_of_iter: {}. Run time is: {} minutes".
                  format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))
            logging.info('{}: Finish running with num_of_iter:{} . Run time is: {} minutes'.
                         format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))

