# Load libraries
from utils import settings
from classification import machine_learning
import random
import numpy
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from csv import writer
import pandas as pd
from sklearn import metrics
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedShuffleSplit


def main(user_id, scores_array):
    data_set_init = \
        pandas.read_csv('D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_trainings_temp_20180328.csv')

    array = data_set_init.values

    users_start_row_number, users_end_row_number, user_ids = machine_learning.get_begin_end_points(array)
    # print(user_ids)
    # print(users_start_row_number)
    # print(users_end_row_number)

    selected_negative_indexes, selected_positive_indexes \
        = machine_learning.get_selected_indexes(users_start_row_number, users_end_row_number, user_ids, user_id)
    # print(len(selected_negative_indexes), ' ', len(selected_positive_indexes))

    data_set_positives = machine_learning.get_data_set(selected_positive_indexes, array, True)
    # print(len(data_set_positives))

    data_set_negatives = machine_learning.get_data_set(selected_negative_indexes, array, False)
    # print(len(data_set_negatives))

    data_set = data_set_positives + data_set_negatives
    data = numpy.array(data_set).tolist()
    # print(len(data))

    columns = ['Mouse Action', 'Traveled Distance', 'Elapsed Time', 'min_v_x',
               'max_v_x', 'mean_v_x', 'std_v_x', 'min_v_y', 'max_v_y',
               'mean_v_y', 'std_v_y', 'min_v', 'max_v', 'mean_v', 'std_v',
               'min_a', 'max_a', 'mean_a', 'std_a', 'min_jerk', 'max_jerk',
               'mean_jerk', 'std_jerk', 'min_angle', 'max_angle', 'mean_angle',
               'std_angle', 'sum_of_angles', 'min_ang_vel', 'max_ang_vel', 'mean_ang_vel',
               'std_ang_vel', 'min_curv', 'max_curv', 'mean_curv', 'std_curv', 'min_d_curv',
               'max_d_curv', 'mean_d_curv', 'std_d_curv', 'number_of_events', 'straightness',
               'number_of_critical_points', 'number_of_pauses', 'paused_time', 'paused_time_ratio',
               'time_to_click', 'direction', 'length_of_line', 'largest_deviation', 'User']
    data_set = pandas.DataFrame(data, columns=columns)

    # # Summarize the dataset
    # # shape
    # print(data_set.shape)
    #
    # # head
    # print(data_set.head(20))
    #
    # # descriptions
    # print(data_set.describe())
    #
    # # class distribution
    # print(data_set.groupby('User').size())

    array = data_set.values

    X = array[:, 0:50]  # features
    Y = array[:, 50]  # class

    # A 20% levalasztott halmazban egyelo aranyban legyenek pozitiv es negativ mintak
    sss = StratifiedShuffleSplit(test_size=settings.VALIDATION_SIZE, random_state=settings.SEED)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_validation = X[train_index], X[test_index]
        Y_train, Y_validation = Y[train_index], Y[test_index]

    # print('X_train: ', len(X_train))
    # print('Y_train: ', len(Y_train))
    # print('X_validation: ', len(X_validation))
    # print('Y_validation: ', len(Y_validation))

    # Make predictions
    # Make predictions on validation dataset
    rndfr = RandomForestClassifier()
    rndfr.fit(X_train, Y_train)

    X_validation_1 = []
    X_validation_0 = []

    # separate the pos and neg data
    for i in range(0, len(X_validation)):
        if Y_validation[i] == 0:
            X_validation_0.append(X_validation[i])
        else:
            X_validation_1.append(X_validation[i])
        i += 1

    # length of sequence
    k = settings.SEQUENCE_LENGTH
    counter_k = 0
    avg = 0
    for t1 in X_validation_1:
        t = rndfr.predict_proba([t1])
        if counter_k < k:
            avg += t[0, 1]
        else:
            counter_k = 0
            avg /= k
            row = ['1', str(avg)]
            scores_array.append(row)
            # print(avg)
            avg = 0
        counter_k += 1

    counter_k = 0
    avg = 0
    for t0 in X_validation_0:
        t = rndfr.predict_proba([t0])
        if counter_k < k:
            avg += t[0, 1]
        else:
            counter_k = 0
            avg /= k
            row = ['0', str(avg)]
            scores_array.append(row)
            # print(avg)
            avg = 0
        counter_k += 1

    # counter = 0
    # scores_array = []
    # for t1 in X_validation:
    #     t = rndfr.predict_proba([t1])
    #     classid = int(Y_validation[counter])
    #     print(classid)
    #     if classid == 0:
    #         print("0, " + str(t[0, 1]))
    #         s = ['0', str(t[0, 1])]
    #         scores_array.append(s)
    #     else:
    #         print("1, " + str(t[0, 1]))
    #         s = ['1', str(t[0, 1])]
    #         scores_array.append(s)
    #         counter += 1

    write_to_file(scores_array)


def write_to_file(prediction_results):
    with open(settings.ALL_SCORES_OUTPUT_SEQUENCES, 'a', newline='') as csv_output:
        data_writer = writer(csv_output, delimiter=',')
        for x in prediction_results:
            data_writer.writerow(x)


def computeAUC(score_file_name, figure_name):
    data = pd.read_csv(score_file_name, names=['label', 'score'])
    # data = pd.read_csv(score_file_name)
    labels = data['label']
    scores = data['score']
    labels = [int(e) for e in labels]
    scores = [float(e) for e in scores]
    auc_value = metrics.roc_auc_score(numpy.array(labels), numpy.array(scores))
    # plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    # EER = thresholds(numpy.argmin(abs(tpr - fpr)))
    print("EER: " + str(eer))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.4f (EER = %0.4f)' % (auc_value, eer))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'ROC curve (AUC)'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig(figure_name)
    # end plot
    return auc_value


def main_1():
    scores_array = []
    user_ids = [7, 9, 12, 15, 16, 20, 21, 23, 29, 35]
    for user_id in user_ids:
        print('user: ', str(user_id))
        main(user_id, scores_array)
    figure_name = 'ROC_curve_sequences_' + str(settings.SEQUENCE_LENGTH) + '_all_users.png'
    computeAUC(settings.ALL_SCORES_OUTPUT_SEQUENCES, figure_name)


main_1()

