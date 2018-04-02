from datetime import datetime
import seaborn as sns
import numpy as np


def sort_by(list_to_sort, list_to_sort_by, descend=True):
    """
    sort one list by another list
    :param list list_to_sort:
    :param list list_to_sort_by:
    :param bool descend:
    :return list sorted_list:
    """

    sorted_lists = [(cid, did) for did, cid in sorted(zip(list_to_sort_by, list_to_sort))]
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
        date_time = datetime.strptime(string, '%Y%m%d_%H_%M_%S')
        return True
    except ValueError:  # FIXME: custom exception required
        print('string is in not in date_time format: {}'.format(string))
        return False


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def neaten_plots(axes, top=True, right=True, left=False, bottom=False):
    for ax in axes:
        sns.despine(ax=ax, top=top, right=right, left=left, bottom=bottom)