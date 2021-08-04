from datetime import datetime
import numpy as np


def sort_by(list_to_sort, list_to_sort_by, descend=True):
    """
    sort one list by another list
    :param list list_to_sort:
    :param list list_to_sort_by:
    :param bool descend:
    :return list sorted_list:
    """

    sorted_lists = [
        (cid, did)
        for did, cid in sorted(
            zip(list_to_sort_by, list_to_sort), key=lambda x: x[0]
        )
    ]
    if descend:
        sorted_lists = sorted_lists[::-1]
    ordered = np.array(sorted_lists)[:, 0]
    ordered_by = np.array(sorted_lists)[:, 1]

    return list(ordered), list(ordered_by)


def flatten_list(lst):
    """
    this is the fastest
    """
    out = []
    for sublist in lst:
        out.extend(sublist)
    return out


def is_datetime(string):
    try:
        date_time = datetime.strptime(string, "%Y%m%d_%H_%M_%S")
        return True
    except ValueError:  # FIXME: custom exception required
        print("string is in not in date_time format: {}".format(string))
        return False


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def pad_track(this_track, n_points):
    this_track = np.pad(
        this_track,
        (0, n_points - len(this_track)),
        "constant",
        constant_values=(0, np.nan),
    )
    return this_track
