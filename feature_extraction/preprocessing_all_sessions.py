import csv
from utils import settings
from feature_extraction import characteristics


def get_user_name_from_file_name(file_path):
    path = file_path[::-1]  # reverse path
    pos = path.find('/')  # find first '/'
    rest = path[pos + 1::]  # cut from first '/'
    pos = rest.find('/')  # find second '/'
    user = rest[:pos]  # cut til second pos
    user = user[::-1]  # reverse
    user = user[4:]  # cut 'user' from 'userNM'
    return user


def get_session_from_file_name(file_path):
    path = file_path[::-1]  # reverse path
    pos = path.find('/')  # find first '/'
    session = path[:pos]
    session = session[::-1]  # reverse
    length = len('session_')
    return session[length:]


def get_mouse_action_type(mouse_events):
    # mouse move
    if mouse_events[-1][3] == 'Move':
        return settings.MOUSE_MOVE

    # mouse click
    if mouse_events[-1][3] == 'Released' and mouse_events[-2][3] == 'Pressed' \
            and (len(mouse_events) == 2 or mouse_events[-3][3] == 'Move'):
        return settings.POINT_CLICK

    # double click / multiple clicks
    if mouse_events[-1][3] == 'Released' and mouse_events[-2][3] == 'Pressed':
        index = len(mouse_events) - 3
        number_of_clicks = 1
        while index > 0 and mouse_events[index][3] == 'Released' and mouse_events[index - 1][3] == 'Pressed':
            index -= 2
            number_of_clicks += 1
        if number_of_clicks == 2:
            return settings.DOUBLE_CLICK
        else:
            return settings.MULTIPLE_CLICKS

    # long click / drag and drop
    if mouse_events[-1][3] == 'Released':
        index = len(mouse_events) - 2
        number_of_drag = 1
        while mouse_events[index][3] == 'Drag':
            index -= 1
            number_of_drag += 1
        if mouse_events[index][3] == 'Pressed':
            prev_x = mouse_events[index][4]
            prev_y = mouse_events[index][5]
            travelled_distance_x = 0
            travelled_distance_y = 0
            for i in range(index + 1, len(mouse_events) - 2):
                travelled_distance_x += abs(int(prev_x) - int(mouse_events[i][4]))
                travelled_distance_y += abs(int(prev_y) - int(mouse_events[i][5]))
                prev_x = mouse_events[i][4]
                prev_y = mouse_events[i][5]
            if travelled_distance_x < 3 and travelled_distance_y < 3:
                if number_of_drag > settings.LONG_CLICK_LIMIT:
                    return settings.LONG_CLICK
                else:
                    return settings.POINT_CLICK
            else:
                return settings.DRAG_AND_DROP

    # scrolling down
    if mouse_events[-1][3] == 'Down' and mouse_events[0][3] == 'Down':
        return settings.SCROLL_DOWN

    # scrolling up
    if mouse_events[-1][3] == 'Up' and mouse_events[0][3] == 'Up':
        return settings.SCROLL_UP

    return settings.UNKNOWN_ACTION


def is_long_event(mouse_events):
    number_of_moves = 0
    for i in range(0, len(mouse_events)):
        if mouse_events[i][3] == 'Move':
            number_of_moves += 1
    if number_of_moves <= settings.EVENT_LIMIT:
        return False
    return True


def is_long_drag_and_drop(mouse_events):
    number_of_drags = 0
    for i in range(0, len(mouse_events)):
        if mouse_events[i][3] == 'Drag':
            number_of_drags += 1
    if number_of_drags <= settings.EVENT_LIMIT:
        return False
    return True


def determine_actions(file_path, data_writer):
    """ Determine an action, decides if it's appropriate. """

    with open(file_path, 'r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        next(data_reader)  # read header from raw data

        user_name = get_user_name_from_file_name(file_path)
        session = get_session_from_file_name(file_path)
        previous_row = next(data_reader)
        mouse_events = [previous_row]

        data = {
            'user_name': user_name,
            'session': session,
            'index_event': 2,
            'index_action': 1,
            'mouse_events': mouse_events,
            'from': 2,
            'to': 2,
            'previous_row': previous_row,
            'row': previous_row,
            'data_writer': data_writer
        }
        attributes = {
            'ds_x': 0,
            'ds_y': 0,
            'd': 0,
            'ds': 0,
            'dt': 0,
            'vx': 0,
            'vy': 0,
            'v': 0,
            'min_vx': 0,
            'max_vx': 0,
            'mean_vx': 0,
            'std_vx': 0,
            'min_vy': 0,
            'max_vy': 0,
            'mean_vy': 0,
            'std_vy': 0,
            'min_v': 0,
            'max_v': 0,
            'mean_v': 0,
            'std_v': 0,
            'min_a': 0,
            'max_a': 0,
            'mean_a': 0,
            'std_a': 0,
            'min_jerk': 0,
            'max_jerk': 0,
            'mean_jerk': 0,
            'std_jerk': 0,
            'min_angle': 0,
            'max_angle': 0,
            'mean_angle': 0,
            'std_angle': 0,
            'sum_of_angles': 0,
            'min_ang_vel': 0,
            'max_ang_vel': 0,
            'mean_ang_vel': 0,
            'std_ang_vel': 0,
            'min_curv': 0,
            'max_curv': 0,
            'mean_curv': 0,
            'std_curv': 0,
            'min_d_curv': 0,
            'max_d_curv': 0,
            'mean_d_curv': 0,
            'std_d_curv': 0,
            'number_of_events': 0,
            'straightness': 0,
            'critical_points': 0,
            'number_of_pauses': 0,
            'paused_time': 0,
            'paused_time_ratio': 0,
            'time_to_click': 0,
            'direction': 0,
            'length_of_line': 0,
            'largest_deviation': 0,
            'is_legal': 0
        }

        for row in data_reader:  # reading starts at row 3.
            data['row'] = row
            data['mouse_events'].append(data['row'])
            data['index_event'] += 1

            # Check where should split raw data into mouse actions
            if data['row'][3] == 'Move':
                if data['previous_row'][3] == 'Released' \
                        or data['previous_row'][3] == 'Down' \
                        or data['previous_row'][3] == 'Up':

                    if len(data['mouse_events']) > settings.EVENT_LIMIT:
                        data['mouse_action'] = get_mouse_action_type(data['mouse_events'][:-1])

                        if data['mouse_action'] == settings.DRAG_AND_DROP:
                            handle_drag_and_drop(data, attributes)

                        else:
                            record_mouse_action(data, -1, attributes)
                    initialize_variables(data)

                else:
                    # Bad mouse event pattern
                    if data['previous_row'][3] != 'Move' \
                            and data['previous_row'][3] != 'Released' \
                            and data['previous_row'][3] != 'Down' \
                            and data['previous_row'][3] != 'Up':
                        initialize_variables(data)

            else:
                # Beginning of a scroll action
                # Check pause between mouse events, if it splits a complete mouse action
                if (data['row'][3] == 'Down' and data['previous_row'][3] != 'Down'
                    or data['row'][3] == 'Up' and data['previous_row'][3] != 'Up') \
                        or (float(data['row'][1]) - float(data['previous_row'][1]) > settings.TIME_LIMIT
                            and (data['previous_row'][3] == 'Released'
                                 or data['previous_row'][3] == 'Down'
                                 or data['previous_row'][3] == 'Up'
                                 or data['previous_row'][3] == 'Move')):
                    data['mouse_action'] = get_mouse_action_type(data['mouse_events'][:-1])
                    record_mouse_action(data, -1, attributes)
                    initialize_variables(data)

            data['previous_row'] = data['row']

        data['mouse_action'] = get_mouse_action_type(data['mouse_events'])
        record_mouse_action(data, 0, attributes)
        return


def initialize_variables(data):
    data['from'] = data['index_event']
    data['mouse_events'] = [data['row']]


def record_mouse_action(data, limit, attributes):
    # We do not need scroll actions and multiple clicks.
    if data['mouse_action'] == settings.SCROLL_DOWN \
            or data['mouse_action'] == settings.SCROLL_UP \
            or data['mouse_action'] == settings.MULTIPLE_CLICKS:
        return

    data['to'] = data['index_event'] + limit

    if limit == -1:
        mouse_events = data['mouse_events'][:-1]
    else:
        mouse_events = data['mouse_events']

    filtered_mouse_events = filter_bad_records(mouse_events)
    if is_long_event(filtered_mouse_events):
        determine_attributes(filtered_mouse_events, attributes, data['session'])
        if attributes['ds'] == 0:
            return
        make_new_record(data, attributes)

        data['index_action'] += 1


def split_drag_and_drop(mouse_events):
    number_of_mouse_moves = 0
    index = 0
    while mouse_events[index][3] != 'Drag':
        number_of_mouse_moves += 1
        index += 1
    return number_of_mouse_moves


def handle_drag_and_drop(data, attributes):
    split = split_drag_and_drop(data['mouse_events'][:-1]) - 1
    if split != 0:
        data['to'] = data['from'] + split - 1
        mouse_events = data['mouse_events'][0:split]
        filtered_mouse_events = filter_bad_records(mouse_events)
        if is_long_event(filtered_mouse_events):
            data['mouse_action'] = get_mouse_action_type(filtered_mouse_events)
            determine_attributes(filtered_mouse_events, attributes, data['session'])
            if attributes['ds'] == 0:
                return
            make_new_record(data, attributes)
            data['index_action'] += 1

    data['from'] = data['to'] + 1
    data['to'] = data['index_event'] - 1
    mouse_events = data['mouse_events'][split:]
    filtered_mouse_events = filter_bad_records(mouse_events)
    if is_long_drag_and_drop(filtered_mouse_events):
        determine_attributes(filtered_mouse_events, attributes, data['session'])
        if attributes['ds'] == 0:
            return
        data['mouse_action'] = settings.DRAG_AND_DROP
        make_new_record(data, attributes)

        data['index_action'] += 1


def filter_bad_records(mouse_events):
    filtered_mouse_events = []
    previous_row = mouse_events[0]
    for i in range(1, len(mouse_events)):
        row = mouse_events[i]
        if (float(previous_row[1]) == float(row[1])
            and previous_row[3] == row[3]) \
                or (int(previous_row[4]) >= settings.X_LIMIT
                    or int(previous_row[5]) >= settings.Y_LIMIT):
            previous_row = row
            continue
        filtered_mouse_events.append(previous_row)
        previous_row = row
    filtered_mouse_events.append(previous_row)
    return filtered_mouse_events


def get_first_press(mouse_events):
    index = 0
    while index < len(mouse_events) and mouse_events[index][3] != 'Pressed':
        index += 1
    if index >= len(mouse_events) - 1:
        return -1
    return index


def determine_attributes(filtered_mouse_events, attributes, session):
    attributes['ds_x'], attributes['ds_y'], attributes['d'] = characteristics.distance(filtered_mouse_events)
    attributes['dt'] = characteristics.summarize(characteristics.time(filtered_mouse_events))
    attributes['ds'] = characteristics.summarize(attributes['d'])
    if attributes['ds'] == 0:
        return
    attributes['vx'], attributes['vy'], attributes['v'] = characteristics.velocity()

    attributes['min_vx'] = characteristics.minimum(attributes['vx'])
    attributes['min_vy'] = characteristics.minimum(attributes['vy'])
    attributes['min_v'] = characteristics.minimum(attributes['v'])
    attributes['max_vx'] = characteristics.maximum(attributes['vx'])
    attributes['max_vy'] = characteristics.maximum(attributes['vy'])
    attributes['max_v'] = characteristics.maximum(attributes['v'])
    attributes['mean_vx'] = characteristics.mean(attributes['vx'])
    attributes['mean_vy'] = characteristics.mean(attributes['vy'])
    attributes['mean_v'] = characteristics.mean(attributes['v'])
    attributes['std_vx'] = characteristics.standard_deviation(attributes['vx'])
    attributes['std_vy'] = characteristics.standard_deviation(attributes['vy'])
    attributes['std_v'] = characteristics.standard_deviation(attributes['v'])

    a_x, a_y, a = characteristics.acceleration()
    attributes['min_a'] = characteristics.minimum(a)
    attributes['max_a'] = characteristics.maximum(a)
    attributes['mean_a'] = characteristics.mean(a)
    attributes['std_a'] = characteristics.standard_deviation(a)

    jerk = characteristics.jerk()
    attributes['min_jerk'] = characteristics.minimum(jerk)
    attributes['max_jerk'] = characteristics.maximum(jerk)
    attributes['mean_jerk'] = characteristics.mean(jerk)
    attributes['std_jerk'] = characteristics.standard_deviation(jerk)

    angle = characteristics.angles()
    attributes['min_angle'] = characteristics.minimum(angle)
    attributes['max_angle'] = characteristics.maximum(angle)
    attributes['mean_angle'] = characteristics.mean(angle)
    attributes['std_angle'] = characteristics.standard_deviation(angle)
    attributes['sum_of_angles'] = characteristics.sum_of_angles()

    ang_vel = characteristics.angular_velocity()
    attributes['min_ang_vel'] = characteristics.minimum(ang_vel)
    attributes['max_ang_vel'] = characteristics.maximum(ang_vel)
    attributes['mean_ang_vel'] = characteristics.mean(ang_vel)
    attributes['std_ang_vel'] = characteristics.standard_deviation(ang_vel)

    curvature = characteristics.curvature()
    attributes['min_curv'] = characteristics.minimum(curvature)
    attributes['max_curv'] = characteristics.maximum(curvature)
    attributes['mean_curv'] = characteristics.mean(curvature)
    attributes['std_curv'] = characteristics.standard_deviation(curvature)

    d_curvature = characteristics.change_of_curvature()
    attributes['min_d_curv'] = characteristics.minimum(d_curvature)
    attributes['max_d_curv'] = characteristics.maximum(d_curvature)
    attributes['mean_d_curv'] = characteristics.mean(d_curvature)
    attributes['std_d_curv'] = characteristics.standard_deviation(d_curvature)

    attributes['number_of_events'] = characteristics.number_of_events(filtered_mouse_events)
    attributes['straightness'] = characteristics.straightness()
    attributes['critical_points'] = characteristics.number_of_critical_points()
    attributes['number_of_pauses'], attributes['paused_time'], attributes['paused_time_ratio'] = \
        characteristics.pauses()

    index = get_first_press(filtered_mouse_events)
    attributes['time_to_click'] = characteristics.time_to_click(index)
    attributes['direction'] = characteristics.direction_of()
    attributes['length_of_line'] = characteristics.length_of_line()
    attributes['largest_deviation'] = characteristics.largest_deviation()
    attributes['is_legal'] = characteristics.is_legal(session)


def make_new_record(data, attributes):
    data['data_writer'].writerow(
        [data['user_name'], data['session'], data['index_action'], data['mouse_action'], data['from'], data['to'],
         attributes['ds'], attributes['dt'],
         attributes['min_vx'], attributes['max_vx'], attributes['mean_vx'], attributes['std_vx'],
         attributes['min_vy'], attributes['max_vy'], attributes['mean_vy'], attributes['std_vy'],
         attributes['min_v'], attributes['max_v'], attributes['mean_v'], attributes['std_v'],
         attributes['min_a'], attributes['max_a'], attributes['mean_a'], attributes['std_a'],
         attributes['min_jerk'], attributes['max_jerk'], attributes['mean_jerk'], attributes['std_jerk'],
         attributes['min_angle'], attributes['max_angle'], attributes['mean_angle'], attributes['std_angle'],
         attributes['sum_of_angles'],
         attributes['min_ang_vel'], attributes['max_ang_vel'], attributes['mean_ang_vel'], attributes['std_ang_vel'],
         attributes['min_curv'], attributes['max_curv'], attributes['mean_curv'], attributes['std_curv'],
         attributes['min_d_curv'], attributes['max_d_curv'], attributes['mean_d_curv'], attributes['std_d_curv'],
         attributes['number_of_events'], attributes['straightness'], attributes['critical_points'],
         attributes['number_of_pauses'], attributes['paused_time'], attributes['paused_time_ratio'],
         attributes['time_to_click'],
         attributes['direction'], attributes['length_of_line'], attributes['largest_deviation'],
         attributes['is_legal']])

#todo ellenorzini hogy bal vagy jobb click..? -> drag and drop