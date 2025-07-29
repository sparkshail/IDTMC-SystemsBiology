import random

num_disjoint = [3]

# size 1
transitions_2 = [[[.15,.25],  [.35,.45], [.15,.25], [.15,.25]],
                 [[.75,.85], [.15,.25]],
                 [[.75,.85], [.05,.15], [.05,.15]],
                 [[.05,.15], [.05,.15], [.05,.15], [.05,.15], [.55, .65]],
                 [[.15,.25], [.15,.25], [.15,.25], [.35,.45]],
                 [[1.0,1.0]],
                 [[1.0,1.0]],
                 [[.75,.85], [.15,.25]]]

# size 2
transitions_3 = [[[.175,.225], [.375,.425], [.175,.225], [.175,.225]],
               [[.775,.825], [.175,.225]],
               [[.775,.825], [.075,.125], [.075,.125]],
               [[.075,.125], [.075,.125], [.075,.125], [.075,.125], [.575,.625]],
               [[.175,.225], [.175,.225], [.175,.225], [.375,.425]],
               [[1.0,1.0]],
               [[1.0,1.0]],
               [[.775,.825], [.175,.225]]]


def get_ran_value(rang):
    r = round(random.uniform(rang[0], rang[1]), 4)
    return r


def split(rang,n):
    ranges = []
    low = rang[0]
    high = rang[1]

    split_by = round((high - low) / n,3)

    cur_low = low
    for i in range(0,n):
        cur_high = round(cur_low + split_by,3)
        ranges.append([cur_low,cur_high])
        cur_low = cur_high

    return ranges


def split_by_model(split_intervals, num_splits):
    new_models = []
    for i in range(0, num_splits):
        new_models.append([])
    cur_index = 0

    for i in range(0,num_splits):
        line_to_add = []
        for line in split_intervals:  # the current line
            for intervals in line: # the intervals
                line_to_add.append(intervals[i])
            new_models[i].append(line_to_add)
            line_to_add = []

    return new_models


def check_validity(line):
    result = "True"
    checker_low = 0
    checker_high = 0
    for i in range(0, len(line)):
        checker_low = checker_low + line[i][0]
        checker_high = checker_high + line[i][1]
    if checker_low > 1:
        result = "Large"
    if checker_high < 1:
        result = "Small"
    return result


def generate_base_models(consts, num_disjoint):
    models = {}
    all_intervals = {}

    for num_splits in num_disjoint:
        models[num_splits] = []

    for num_splits in num_disjoint:
        new_consts = []
        line_num = 1
        for line in consts:
            temp_intervals = []
            for interval in line:
                cur_splits = split(interval, num_splits)
                all_intervals[tuple(interval)] = cur_splits
                temp_intervals.append(cur_splits)
            new_consts.append(temp_intervals)
            line_num = line_num + 1

        models[num_splits] = split_by_model(new_consts, num_splits)

    return models, all_intervals


def get_smallest(intervals):
    smallest = intervals[0]
    min_index = 0
    for i in range(1,len(intervals)):
        if intervals[i][1] < smallest[1]:
            smallest = intervals[i]
            min_index = i
    return smallest, min_index


def get_largest(intervals, can_go_smaller = None):
    largest = intervals[0]
    max_index = 0
    for i in range(1, len(intervals)):
        if intervals[i][1] > largest[1]:
            largest = intervals[i]
            max_index = i
    return largest, max_index


def get_location(interval_collection, desired_interval, can_go_smaller=None):
    for key,value in interval_collection.items():
        for i in range(0,len(value)):
            if value[i] == desired_interval:
                return key, i


def get_original(interval_collection, desired_interval):
    for key, value in interval_collection.items():
        for i in range(0,len(value)):
            if value[i] == desired_interval:
                return key


def get_line_location(line, desired_interval):
    for i in range(0,len(line)):
        if line[i] == desired_interval:
            return i


def change_interval(interval_collection, line, res, change_index, can_go_smaller = None):
    smallest_interval, smallest_location = get_smallest(line)
    largest_interval, largest_location = get_largest(line)

    for key,value in interval_collection.items():
        num_disjoint = len(interval_collection[key])
        break

    if res == "Small":
        key, index = get_location(interval_collection, smallest_interval)
        new_interval = interval_collection[key][change_index]
        new_line = line
        new_line.pop(smallest_location)
        new_line.insert(smallest_location, new_interval)
    else:
        temp = line.copy()
        original_interval = get_original(interval_collection, largest_interval)
        if can_go_smaller[original_interval] == False:
            temp.remove(largest_interval)
            largest_interval = get_largest(temp, can_go_smaller=None)[0]

        key, index = get_location(interval_collection, largest_interval, can_go_smaller)
        change_index = get_line_location(line, largest_interval)
        new_interval = interval_collection[key][0]
        new_line = line
        new_line.remove(largest_interval)
        new_line.insert(largest_location, new_interval)
        can_go_smaller[tuple(original_interval)] = False

    return new_line


def update_main(consts, num_disjoint):
    models, interval_collection = generate_base_models(consts, num_disjoint)
    model_num = 1
    for key,value in models.items(): # number of disjoint intervals
        print("Number of Disjoint Intervals ", num_disjoint)
        for model in value: # submodel
            print("Model #", model_num)
            for line in model:
                cur_line = line
                cur_res = check_validity(cur_line)
                can_go_smaller = {}

                for key, value in interval_collection.items():
                    small_change_index = len(interval_collection[key])-1
                    can_go_smaller[tuple(key)] = True

                large_change_index = 0
                while True:
                    new_line = []
                    if cur_res == "Small":
                        new_line = change_interval(interval_collection, line, cur_res, small_change_index)
                        small_change_index -= 1
                    if cur_res == "Large":
                        new_line = change_interval(interval_collection, line, cur_res, large_change_index, can_go_smaller)
                    cur_res = check_validity(new_line)

                    if cur_res == "True":
                        break

                try:
                    print("Line", new_line)
                except:
                    print("Line", line)
                    new_line = line.copy()

                while True:
                    probs = []
                    total_prob = 0

                    if line == [[0.23, 0.25], [0.39, 0.41], [0.19, 0.21], [0.19, 0.21]] or line == [[0.23, 0.25], [0.19, 0.21], [0.19, 0.21], [0.39, 0.41]]\
                            or line == [[0.215, 0.225], [0.395, 0.405], [0.195, 0.205], [0.195, 0.205]] or line == [[0.215, 0.225], [0.195, 0.205], [0.195, 0.205], [0.395, 0.405]]:
                        print("lowest bounds")
                        break

                    for transition in new_line:
                        cur_prob = get_ran_value(transition)
                        probs.append(cur_prob)
                        total_prob = total_prob + cur_prob
                    if sum(probs) == 1:
                        print(probs)
                        print("\n")
                        break

            print("\n")
            model_num += 1


out = update_main(transitions_3,num_disjoint)


