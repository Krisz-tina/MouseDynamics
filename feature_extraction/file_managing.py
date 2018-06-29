from csv import writer
from os import listdir
from os import path
from feature_extraction import preprocessing_all_sessions
from utils import settings


def get_training_files():
    training_files = []
    for directory in listdir(settings.ROOT_TRAINING):
        directory = settings.ROOT_TRAINING + '/' + directory
        if path.isdir(directory):
            for file in listdir(directory):
                file = directory + '/' + file
                training_files.append(file)
    return training_files


def get_test_files():
    test_files = []
    for directory in listdir(settings.ROOT_TEST):
        directory = settings.ROOT_TEST + '/' + directory
        if path.isdir(directory):
            for file in listdir(directory):
                file = directory + '/' + file
                test_files.append(file)
    return test_files


def process_training_files():
    with open(settings.OUTPUT_TRAINING, 'w', newline='') as csv_output:
        data_writer = writer(csv_output, delimiter=',')
        header = ['User', 'Session', 'Index', 'Mouse Action', 'From', 'To',
                  'Traveled Distance', 'Elapsed Time',
                  'min_v_x', 'max_v_x', 'mean_v_x', 'std_v_x',
                  'min_v_y', 'max_v_y', 'mean_v_y', 'std_v_y',
                  'min_v', 'max_v', 'mean_v', 'std_v',
                  'min_a', 'max_a', 'mean_a', 'std_a',
                  'min_jerk', 'max_jerk', 'mean_jerk', 'std_jerk',
                  'min_angle', 'max_angle', 'mean_angle', 'std_angle', 'sum_of_angles',
                  'min_ang_vel', 'max_ang_vel', 'mean_ang_vel', 'std_ang_vel',
                  'min_curv', 'max_curv', 'mean_curv', 'std_curv',
                  'min_d_curv', 'max_d_curv', 'mean_d_curv', 'std_d_curv',
                  'number_of_events', 'straightness', 'number_of_critical_points',
                  'number_of_pauses', 'paused_time', 'paused_time_ratio', 'time_to_click',
                  'direction', 'length_of_line', 'largest_deviation', 'is_legal'
                  ]
        data_writer.writerow(header)
        training_files = get_training_files()
        for file in training_files:
            print(file)
            preprocessing_all_sessions.determine_actions(file, data_writer)


def process_test_files():
    with open(settings.OUTPUT_TEST, 'w', newline='') as csv_output:
        data_writer = writer(csv_output, delimiter=',')
        header = ['User', 'Session', 'Index', 'Mouse Action', 'From', 'To',
                  'Traveled Distance', 'Elapsed Time',
                  'min_v_x', 'max_v_x', 'mean_v_x', 'std_v_x',
                  'min_v_y', 'max_v_y', 'mean_v_y', 'std_v_y',
                  'min_v', 'max_v', 'mean_v', 'std_v',
                  'min_a', 'max_a', 'mean_a', 'std_a',
                  'min_jerk', 'max_jerk', 'mean_jerk', 'std_jerk',
                  'min_angle', 'max_angle', 'mean_angle', 'std_angle', 'sum_of_angles',
                  'min_ang_vel', 'max_ang_vel', 'mean_ang_vel', 'std_ang_vel',
                  'min_curv', 'max_curv', 'mean_curv', 'std_curv',
                  'min_d_curv', 'max_d_curv', 'mean_d_curv', 'std_d_curv',
                  'number_of_events', 'straightness', 'number_of_critical_points',
                  'number_of_pauses', 'paused_time', 'paused_time_ratio', 'time_to_click',
                  'direction', 'length_of_line', 'largest_deviation', 'is_legal'
                  ]
        data_writer.writerow(header)
        test_files = get_test_files()
        for file in test_files:
            print(file)
            preprocessing_all_sessions.determine_actions(file, data_writer)


if __name__ == '__main__':
    process_training_files()
