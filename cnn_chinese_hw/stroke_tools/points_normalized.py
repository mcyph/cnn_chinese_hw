from sys import maxsize


def points_normalized(LStrokes, width=1000, height=1000):
    """
    Make sure the borders of LStrokes are 
    the same between the two sets of data
    """
    min_x = maxsize
    max_x = -maxsize
    min_y = maxsize
    max_y = -maxsize

    for LStroke in LStrokes:
        for x, y in LStroke:
            # TODO: What if max_x is GREATER than min_x?
            if x < min_x: min_x = x
            if y < min_y: min_y = y

            if x > max_x: max_x = x
            if y > max_y: max_y = y

    # TODO: REMOVE ZERO DIVISION ERRORS!
    try:
        x_num_times = width/(max_x-min_x)
    except ZeroDivisionError:
        x_num_times = 1

    try:
        y_num_times = height/(max_y-min_y)
    except ZeroDivisionError:
        y_num_times = 1

    return_list = []
    for LStroke in LStrokes:
        L = []
        for x, y in LStroke:
            x -= min_x
            y -= min_y
            x *= x_num_times
            y *= y_num_times
            x = max(0, min(round(x), width))  # should this be width-1?
            y = max(0, min(round(y), height))
            L.append((x, y))
        return_list.append(L)
    #print("AFTER:", return_list)
    return return_list
