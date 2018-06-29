PATH = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/test_files/user9/session_4088341904'
OUTPUT = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/test_files/user9/session_4088341904'
ROOT_TRAINING = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/training_files'
ROOT_TEST = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/test_files'
OUTPUT_TEST = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_tests.csv'
OUTPUT_TRAINING = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_trainings.csv'

BALABIT_FEATURES_INPUT = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_features_one_per_user.csv'
BALABIT_FEATURES_OUTPUT = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/balabit_features_one_per_user_output.csv'

SCORES_OUTPUT = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/scores_.csv'
ALL_SCORES_OUTPUT = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/all_scores.csv'
ALL_SCORES_OUTPUT_SEQUENCES = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/all_scores_sequences.csv'

IS_VALID = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/test_files/public_labels.csv'

# preprocessing
X_LIMIT = 1920
Y_LIMIT = 1080
EVENT_LIMIT = 5
TIME_LIMIT = 10
DISTANCE_LIMIT = 1  # the limit between long click and drag-and-drop
LONG_CLICK_LIMIT = 3  # the bound between a click and a long click
CRITICAL_POINT = 0.0005  # ha ennel kisebb a curv/s (change of curvature)
ANGLE_LIMIT = 20.0
TIME_PAUSE_LIMIT = 0.1

# mouse actions
MOUSE_MOVE = 1
POINT_CLICK = 2
DRAG_AND_DROP = 3
LONG_CLICK = 4
DOUBLE_CLICK = 5
MULTIPLE_CLICKS = 6
SCROLL_DOWN = 7
SCROLL_UP = 8
UNKNOWN_ACTION = 0

# machine learning
NUMBER_OF_ACTIONS = 900
NUMBER_OF_ACTIONS_PER_CLASS = 100
SEED = 7
THRESHOLD = 0.5
USER_ID = 9
VALIDATION_SIZE = 0.2
SEQUENCE_LENGTH = 10

# plot
MOUSE_MOVE_COLOR = 'green'
POINT_CLICK_COLOR = 'blue'
DRAG_AND_DROP_COLOR = 'orange'
LONG_CLICK_COLOR = 'yellow'
DOUBLE_CLICK_COLOR = 'black'
MULTIPLE_CLICKS_COLOR = 'purple'
SCROLL_DOWN_COLOR = 'red'
SCROLL_UP_COLOR = 'brown'
