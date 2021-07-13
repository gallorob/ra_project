def env_base_goal(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent


def pass_through_center(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger in the middle
    # d : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'd'
    # passenger on taxi on the middle
    elif passenger_location == 4 and (taxi_row, taxi_col) == (2, 2):
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent


def pass_through_4_corners(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on Green
    # e : Taxi with passenger on Yellow
    # f : Taxi with passenger on Blue
    # g : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'g'
    # passenger on taxi on Blue
    elif passenger_location == 4 and (taxi_row, taxi_col) == (4, 3):
        fluent = 'f'
    # passenger on taxi on Yellow
    elif passenger_location == 4 and (taxi_row, taxi_col) == (4, 0):
        fluent = 'e'
    # passenger on taxi on Green
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 4):
        fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 0):
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent


def pass_through_3_corners(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on Green
    # e : Taxi with passenger on Yellow
    # f : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'f'
    # passenger on taxi on Yellow
    elif passenger_location == 4 and (taxi_row, taxi_col) == (4, 0):
        fluent = 'e'
    # passenger on taxi on Green
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 4):
        fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 0):
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent


def pass_through_2_corners(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on Green
    # e : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'e'
    # passenger on taxi on Green
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 4):
        fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 0):
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent


def pass_through_1_corner(taxi_row, taxi_col, passenger_location, destination, pos_to_colors, idx_colors) -> str:
    fluent = ''
    # a : Taxi with no passenger, anywhere
    # b : Taxi with passenger, not on destination
    # c : Taxi with passenger on Red
    # d : Taxi with passenger on destination

    # passenger at right location - goal reached
    if passenger_location == destination and pos_to_colors[(taxi_row, taxi_col)] == idx_colors[destination]:
        fluent = 'd'
    # passenger on taxi on Red
    elif passenger_location == 4 and (taxi_row, taxi_col) == (0, 0):
        fluent = 'c'
    # passenger on taxi
    elif passenger_location == 4:  # taxi at destination
        fluent = 'b'
    # passenger not on taxi
    else:
        fluent = 'a'

    return fluent
