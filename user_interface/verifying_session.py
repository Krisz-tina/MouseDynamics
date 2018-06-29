# Load libraries
import math
from utils import settings
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
from user_interface import preprocessing_session


def get_begin_end_points(array):
    i = 0
    users_start_row_number = [i]
    users_end_row_number = []
    user_ids = []
    row = array[i]
    while i < len(array):
        user_id = int(row[-1])
        user_ids.append(user_id)
        while row[-1] == user_id:
            i += 1
            if i != len(array):
                row = array[i]
            else:
                users_end_row_number.append(i - 1)
                return users_start_row_number, users_end_row_number, user_ids
        users_end_row_number.append(i - 1)
        users_start_row_number.append(i)
    users_end_row_number.append(i - 1)
    return users_start_row_number, users_end_row_number, user_ids


def select_random_indexes(random_numbers, start_row, end_row, number_of_actions):
    selected_indexes = []
    for i in range(0, number_of_actions):
        rnd = random.randint(start_row, end_row)
        while random_numbers[rnd] == 1:
            rnd = random.randint(start_row, end_row)
        random_numbers[rnd] = 1
        selected_indexes.append(rnd)
    return selected_indexes


def get_selected_indexes(users_start_row_number, users_end_row_number, user_ids, user_id, number_of_actions_per_user):
    selected_negative_indexes = []
    # selected_positive_indexes = []
    random_numbers = [0] * (users_end_row_number[-1] + 1)
    for i in range(0, len(user_ids)):
        if user_ids[i] != user_id:
            selected_negative_indexes \
                += select_random_indexes(random_numbers, users_start_row_number[i], users_end_row_number[i], number_of_actions_per_user)
        # else:
        #     selected_positive_indexes \
        #         = select_random_indexes(random_numbers, users_start_row_number[i], users_end_row_number[i], settings.NUMBER_OF_ACTIONS)
    return selected_negative_indexes  # , selected_positive_indexes


def get_data_set(selected_indexes, array, positive):
    data_set = []
    for i in range(0, len(selected_indexes)):
        row = array[selected_indexes[i]]
        if not positive:
            row[-1] = 0.0  # Set user id equally
        data_set.append(row)
    return data_set


def main(user, session):
    file_name = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/test_files/' + user + '/' + session
    file_name_output = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/test_files/' + user + '/' + session
    preprocessing_session.determine_actions(file_name, file_name_output)

    data_set_init = \
        pandas.read_csv('D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_trainings_temp_20180328.csv')

    array = data_set_init.values

    users_start_row_number, users_end_row_number, user_ids = get_begin_end_points(array)

    data_set_positives = pandas.read_csv(file_name_output)
    number_of_actions_per_user = math.ceil(len(data_set_positives) / 9)

    selected_negative_indexes \
        = get_selected_indexes(users_start_row_number, users_end_row_number, user_ids, user, number_of_actions_per_user)

    data_set_negatives = get_data_set(selected_negative_indexes, array, False)

    if len(data_set_negatives) != len(data_set_positives):
        data_set_negatives = data_set_negatives[0:len(data_set_positives)]
    data_set = data_set_positives.values

    d = numpy.array(data_set).tolist()
    data = numpy.concatenate([d, numpy.array(data_set_negatives).tolist()])

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

    array = data_set.values

    X = array[:, 0:50]  # features
    Y = array[:, 50]  # class
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    # Make predictions
    # Make predictions on validation dataset
    rndfr = RandomForestClassifier()
    rndfr.fit(X_train, Y_train)
    predictions = rndfr.predict(X_validation)
    return accuracy_score(Y_validation, predictions)
