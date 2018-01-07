import time
from sklearn.model_selection import KFold
import logging
from datetime import datetime
from struct_perceptron import StructPerceptron
from parser_model import ParserModel

# open log connection
directory = "C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP\\NLP_HW2\\"
LOG_FILENAME = datetime.now().strftime(directory + 'logs\\LogFile_basic_model_%d_%m_%Y_%H_%M.log')
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


def main(train_file_to_use, test_file_to_use, comp_file_to_use, test_type, features_combination_list, num_of_iter, comp):

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

        print('{}: Finish creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                          features_combination))
        logging.info('{}: Finish creating parser model for features : {}'.format(time.asctime(time.localtime(time.time())),
                                                                                 features_combination))

        # Run perceptron to learn the best weights
        print('{}: Start Perceptron for features : {} and number of iterations: {}'.
              format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter))
        logging.info('{}: Start Perceptron for features : {} and number of iterations: {}'.
                     format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter))
        perceptron_obj = StructPerceptron(model=parser_model_obj)
        weights = perceptron_obj.perceptron(num_of_iter=num_of_iter)

        train_run_time = (time.time() - train_start_time) / 60.0
        print('{}: Finish Perceptron for features : {} and lambda: {}. run time: {}'.
              format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter, train_run_time))
        logging.info('{}: Finish Perceptron for features : {} and lambda: {}. run time: {}'.
                     format(time.asctime(time.localtime(time.time())), features_combination, num_of_iter, train_run_time))

        # Evaluate the results of the model
        # TODO: change according to the new evaluation part
        # write_file_name = datetime.now().strftime(directory + 'file_results/result_MEMM_basic_model_final__' + test_type +
        #                                           '%d_%m_%Y_%H_%M.wtag')
        # evaluate_class = Evaluate(parser_model_obj, weights, write_file_name, comp=comp)
        # if not comp:
        #     word_results_dictionary = evaluate_class.run()
        # if comp:
        #     evaluate_class.write_result_doc()
        # logging.info('{}: The model hyper parameters: \n num_of_iter:{} \n test file: {} \n train file: {}'
        #              .format(time.asctime(time.localtime(time.time())), num_of_iter, test_file_to_use,
        #                      train_file_to_use))
        # logging.info('{}: Related results files are: \n {}'.format(time.asctime(time.localtime(time.time())),
        #                                                            write_file_name))
        #
        # # print(word_results_dictionary)
        # summary_file_name = '{0}analysis/summary_{1}_{2.day}_{2.month}_{2.year}_{2.hour}_{2.minute}.csv' \
        #     .format(directory, test_type, datetime.now())
        # evaluate_class.create_summary_file(num_of_iter, features_combination, test_file_to_use, train_file_to_use,
        #                                    summary_file_name, perceptron_obj, comp)
        #
        # logging.info('{}: Following Evaluation results for features {}'.
        #              format(time.asctime(time.localtime(time.time())), features_combination))
        # if not comp:
        #     logging.info('{}: Evaluation results are: \n {} \n'.format(time.asctime(time.localtime(time.time())),
        #                                                                word_results_dictionary))
        # logging.info('-----------------------------------------------------------------------------------')


if __name__ == "__main__":
    logging.info('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    print('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    train_file = directory + 'HW2-files/train.labeled'
    test_file = directory + 'HW2-files/test.labeled'
    comp_file = directory + 'HW2-files/comp.unlabeled'
    cv = False
    comp = True
    if cv:
        cross_validation(train_file)
    else:
        advanced_features = range(1, 19)
        advanced_features = [str(i) for i in advanced_features]
        basic_features = range(1, 14)
        basic_features = [str(i) for i in basic_features]
        feature_type_dict = {
            'all_features': [advanced_features],
            'basic_model': [basic_features]}

        num_of_iter_list = [20, 50, 80, 100]
        for num_of_iter in num_of_iter_list:
            start_time = time.time()
            if not comp:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'test', feature_type_list, num_of_iter, comp)
            else:
                for feature_type_name, feature_type_list in feature_type_dict.items():
                    main(train_file, test_file, comp_file, 'comp', feature_type_list, num_of_iter, comp)
            run_time = (time.time() - start_time) / 60.0
            print("{}: Finish running with num_of_iter: {}. Run time is: {} minutes".
                  format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))
            logging.info('{}: Finish running with num_of_iter:{} . Run time is: {} minutes'.
                         format(time.asctime(time.localtime(time.time())), num_of_iter, run_time))

