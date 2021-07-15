from utils.qlearning import history


def env_base_goal(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        if history.fluent == 'b':
            history.fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        if history.fluent == 'a':
            history.fluent = 'b'
    # passenger not on taxi
    else:
        if not history.fluent:
            history.fluent = 'a'

    return history.fluent


def pass_through_center(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger in the middle
    # d : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        if history.fluent == 'c':
            history.fluent = 'd'
    # passenger on taxi on the middle
    elif passenger_location == 4 and (taxi_row, taxi_col) == (2, 2):
        if history.fluent == 'b':
            history.fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        if history.fluent == 'a':
            history.fluent = 'b'
    # passenger not on taxi
    else:
        if not history.fluent:
            history.fluent = 'a'

    return history.fluent


def pass_through_3_corners(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on Green
    # e : Taxi with passenger on Yellow
    # f : Taxi with passenger on destination

    corners = list(pos_to_colors.keys())
    corners.remove([k for k, v in pos_to_colors.items() if v == idx_colors[destination]][0])

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        if history.fluent == 'e':
            history.fluent = 'f'
    # passenger on taxi on Yellow
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[2]:
        if history.fluent == 'd':
            history.fluent = 'e'
    # passenger on taxi on Green
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[1]:
        if history.fluent == 'c':
            history.fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[0]:
        if history.fluent == 'b':
            history.fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        if history.fluent == 'a':
            history.fluent = 'b'
    # passenger not on taxi
    else:
        if not history.fluent:
            history.fluent = 'a'

    return history.fluent


def pass_through_2_corners(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on Green
    # e : Taxi with passenger on destination

    corners = list(pos_to_colors.keys())
    corners.remove([k for k, v in pos_to_colors.items() if v == idx_colors[destination]][0])

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        if history.fluent == 'd':
            history.fluent = 'e'
    # passenger on taxi on Green
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[1]:
        if history.fluent == 'c':
            history.fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[0]:
        if history.fluent == 'b':
            history.fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        if history.fluent == 'a':
            history.fluent = 'b'
    # passenger not on taxi
    else:
        if not history.fluent:
            history.fluent = 'a'

    return history.fluent


def pass_through_1_corner(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on destination

    corners = list(pos_to_colors.keys())
    corners.remove([k for k, v in pos_to_colors.items() if v == idx_colors[destination]][0])

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        if history.fluent == 'c':
            history.fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == corners[0]:
        if history.fluent == 'b':
            history.fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        if history.fluent == 'a':
            history.fluent = 'b'
    # passenger not on taxi
    else:
        if not history.fluent:
            history.fluent = 'a'

    return history.fluent
