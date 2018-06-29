import csv
import numpy
import math
from utils import settings


data = {
    'x': [],
    'y': [],
    'dt': [],
    'ds_x': [],
    'ds_y': [],
    'ds': [],
    'v_x': [],
    'v_y': [],
    'v': [],
    'a_x': [],
    'a_y': [],
    'a': [],
    'angles': [],
    'curvatures': [],
    'change_of_curvatures': []
}


def distance(mouse_events):
    data['ds_x'] = []
    data['ds_y'] = []
    data['ds'] = []
    data['x'] = [(int(mouse_events[0][4]))]
    data['y'] = [(int(mouse_events[0][5]))]
    for i in range(1, len(mouse_events)):
        x1 = int(mouse_events[i-1][4])
        x2 = int(mouse_events[i][4])
        y1 = int(mouse_events[i-1][5])
        y2 = int(mouse_events[i][5])
        data['x'].append(int(mouse_events[i][4]))
        data['y'].append(int(mouse_events[i][5]))
        data['ds_x'].append((x2 - x1))
        data['ds_y'].append((y2 - y1))
        data['ds'].append(distance_between(x1, y1, x2, y2))
    return data['ds_x'], data['ds_y'], data['ds']


def time(mouse_events):
    data['dt'] = []
    for i in range(1, len(mouse_events)):
        data['dt'].append(float(mouse_events[i][1]) - float(mouse_events[i - 1][1]))
    return data['dt']


def velocity():
    data['v_x'] = []
    data['v_y'] = []
    data['v'] = []
    for i in range(0, len(data['ds'])):
        if data['dt'][i] != 0.0:
            data['v_x'].append(data['ds_x'][i] / data['dt'][i])
            data['v_y'].append(data['ds_y'][i] / data['dt'][i])
            data['v'].append(data['ds'][i] / data['dt'][i])
            # v.append(math.sqrt(data['v_x'][i-1]*data['v_x'][i-1] + data['v_y'][i-1]*data['v_y'][i-1]))
    return data['v_x'], data['v_y'], data['v']


def acceleration():
    data['a_x'] = []
    data['a_y'] = []
    data['a'] = []
    for i in range(1, len(data['v'])):
        if data['dt'][i] != 0.0:
            dv_x = data['v_x'][i] - data['v_x'][i-1]
            dv_y = data['v_y'][i] - data['v_y'][i-1]
            dv = data['v'][i] - data['v'][i-1]
            data['a_x'].append(dv_x / data['dt'][i])
            data['a_y'].append(dv_y / data['dt'][i])
            data['a'].append(dv / data['dt'][i])
    return data['a_x'], data['a_y'], data['a']


def jerk():
    jerk_list = []
    for i in range(1, len(data['a'])):
        if data['dt'][i+1] != 0.0:
            da = data['a'][i] - data['a'][i-1]
            jerk_list.append(da / data['dt'][i + 1])
    return jerk_list


def angles():  # teta, line and X axis
    data['angles'] = []
    for i in range(0, len(data['ds'])):
        data['angles'].append(math.atan2(data['ds_y'][i], data['ds_x'][i]))
    return data['angles']


def sum_of_angles():  # tetas
    return summarize(data['angles'])


def angular_velocity():  # omega
    ang_vel = []
    for i in range(1, len(data['angles'])):
        if data['dt'][i] != 0.0:
            d_angle = data['angles'][i] - data['angles'][i-1]
            ang_vel.append(d_angle / data['dt'][i])
    return ang_vel


def curvature():
    data['curvatures'] = []
    for i in range(1, len(data['angles'])):
        if data['ds'][i] != 0:
            d_angle = data['angles'][i] - data['angles'][i-1]
            data['curvatures'].append(d_angle / data['ds'][i])
    return data['curvatures']


def change_of_curvature():
    data['change_of_curvatures'] = []
    for i in range(1, len(data['curvatures'])):
        if data['ds'][i+1] != 0:
            c = data['curvatures'][i] - data['curvatures'][i-1]
            data['change_of_curvatures'].append(c / data['ds'][i+1])
    return data['change_of_curvatures']


def number_of_events(mouse_events):
    return len(mouse_events)


def straightness():
    x1 = int(data['x'][0])
    y1 = int(data['y'][0])
    x2 = int(data['x'][-1])
    y2 = int(data['y'][-1])
    line = distance_between(x1, y1, x2, y2)
    s = summarize(data['ds'])
    if s == 0:
        return 0
    return line / s


def number_of_critical_points():
    nr = 0
    nr2 = 0
    for i in range(0, len(data['change_of_curvatures'])):
        if data['change_of_curvatures'][i] < settings.CRITICAL_POINT:  # elmozdulásra eső szögváltozás
            nr += 1
    for i in range(2, len(data['x'])):
        x1 = data['x'][i-2]
        y1 = data['y'][i-2]
        x2 = data['x'][i-1]
        y2 = data['y'][i-1]
        x3 = data['x'][i]
        y3 = data['y'][i]
        angle = angle_between_three_point(x1, y1, x2, y2, x3, y3)
        if angle < settings.ANGLE_LIMIT:
            nr2 += 1
    return nr2


def angle_between_three_point(x1, y1, x2, y2, x3, y3):
    a = numpy.array([x1, y1])
    b = numpy.array([x2, y2])
    c = numpy.array([x3, y3])

    d1 = a - b  # [x1-x2, y1-y2]
    d2 = c - b  # [x3-x2, y3-y2]

    norm_d1 = numpy.linalg.norm(d1)  # |d1| = sqrt((x1-x2)^2 + (y1-y2)^2)
    norm_d2 = numpy.linalg.norm(d2)  # |d2| = sqrt((x3-x2)^2 + (y3-y2)^2)
    d1_d2 = numpy.dot(d1, d2)  # d1*d2 = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)

    if norm_d1 * norm_d2 != 0.0:
        cosine_angle = d1_d2 / (norm_d1 * norm_d2)  # cos(a) = d1*d2 / |d1|*|d2|
        if cosine_angle > 1.0:
            cosine_angle = 1.0
        if cosine_angle < -1.0:
            cosine_angle = -1.0
        a = numpy.arccos(cosine_angle)  # a = arccos(cos(a))

        return math.degrees(a)
    return 361


def pauses():
    number_of_pauses = 0
    paused_time = 0
    for i in range(0, len(data['dt'])):
        if data['dt'][i] < settings.TIME_PAUSE_LIMIT:
            number_of_pauses += 1
            paused_time += data['dt'][i]
    paused_time_ratio = paused_time / summarize(data['dt'])
    return number_of_pauses, paused_time, paused_time_ratio


def time_to_click(index):
    if index == -1:
        return 0
    else:
        return data['dt'][index]


def direction_of():
    """ Determines the directions of a mouse action. """

    x1 = data['x'][0]
    y1 = data['y'][0]
    x2 = data['x'][-1]
    y2 = data['y'][-1]

    angle_radian = math.atan2(y2-y1, x2-x1)
    angle = math.degrees(angle_radian)
    if angle < 0:
        angle_radian += 2*math.pi
        angle = math.degrees(angle_radian)

    if 22.5 <= angle < 67.5:
        return 1
    if 67.5 <= angle < 112.5:
        return 2
    if 112.5 <= angle < 157.5:
        return 3
    if 157.5 <= angle < 202.5:
        return 4
    if 202.5 <= angle < 247.5:
        return 5
    if 247.5 <= angle < 292.5:
        return 6
    if 292.5 <= angle < 337.5:
        return 7
    return 8


def length_of_line():
    x1 = data['x'][0]
    y1 = data['y'][0]
    x2 = data['x'][-1]
    y2 = data['y'][-1]

    return distance_between(x1, y1, x2, y2)


def largest_deviation():
    deviations = []
    for i in range(1, len(data['x'])-1):
        deviations.append(distance_between_point_and_line(data['x'][i], data['y'][i]))
    return maximum(deviations)


def distance_between_point_and_line(x1, y1):
    p1 = numpy.array([data['x'][0], data['y'][0]])
    p2 = numpy.array([data['x'][-1], data['y'][-1]])
    p3 = numpy.array([x1, y1])
    if numpy.linalg.norm(p2 - p1) != 0:
        d = numpy.linalg.norm(numpy.cross(p2 - p1, p1 - p3)) / numpy.linalg.norm(p2 - p1)
        return d
    return 0


def is_legal(session):
    s = 'session_' + session
    with open(settings.IS_VALID, 'r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        for row in data_reader:
            if row[0] == s:
                return row[1]
    return -1


def minimum(array):
    return min(array)


def maximum(array):
    return max(array)


def mean(array):
    return numpy.mean(array)


def standard_deviation(array):
    return numpy.std(array)


def distance_between(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def summarize(array):
    return sum(array)

# todo max-min 