import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from looming_spots.analysis import tracks
import itertools as it
import collections
import pandas as pd
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# pandas2ri.activate()
# rstats = importr('stats')


def get_contingencies(session_list):
    n_flees = sum(tracks.n_flees_all_sessions(session_list))
    n_non_flees = sum(tracks.n_non_flees_all_sessions(session_list))
    return n_flees, n_non_flees


def get_all_group_contingencies(session_dictionary):
    """

    :param session_dictionary: key is the group, value is a list of session.Session objects
    :return:
    """
    contingencies_dict = {}

    for label, sessions in session_dictionary.items():
        contingencies_dict.setdefault(label, get_contingencies(sessions))
    contingencies_df = pd.DataFrame.from_dict(contingencies_dict)
    contingencies_df.set_index([['flees', 'non-flees']])
    return contingencies_df


def plot_contingency_stacked_bar(session_dictionary):
    fig = plt.figure()
    contingency_df = get_all_group_contingencies(session_dictionary)
    contingency_df.T.plot(kind='bar', stacked=True, colors=['r', 'k'])
    return fig


def compare_all_groups_fisher(contingency_table):  # TODO: clean this
    result_list = list(map(dict, itertools.combinations(contingency_table.items(), 2)))
    df_dict = {}

    conditions = []
    n_flees = []
    n_non_flees = []
    comparisons = []
    p_values = []

    for comparison in result_list:
        for k, v in comparison.items():
            conditions.append(k)
            n_flees.append(v[0])
            n_non_flees.append(v[1])
            comparisons.append('_vs_'.join(comparison.keys()))
            p_values.append(scipy.stats.fisher_exact([v for v in comparison.values()])[1])

    df_dict.setdefault('condition', conditions)
    df_dict.setdefault('n_flees', n_flees)
    df_dict.setdefault('n_non_flees', n_non_flees)
    df_dict.setdefault('comparison', comparisons)
    df_dict.setdefault('p_value', p_values)

    return pd.DataFrame.from_dict(df_dict)


def filter_by(session_list, filter_date=None, filter_date_range=None):
    if filter_date_range:
        session_list, session_dates = filter_by_date_range(session_list[0],
                                                           session_list[1],
                                                           filter_date_range[0],
                                                           filter_date_range[1])
    elif filter_date:
        session_list, session_dates = filter_by_date_exact(session_list[0], session_list[1], filter_date)
    else:
        session_list = session_list[0]
    return session_list, session_dates  # FIXME:


def get_rates_for_timepoints(session_lists, time_points):
    stats_dict = collections.OrderedDict()
    for session_list in session_lists:
        for tp in time_points:
            tp_session_list, session_dates = filter_by_date_exact(session_list[0], session_list[1], tp)
            n_flees = tracks.n_flees_all_sessions(tp_session_list)
            n_non_flees = tracks.n_non_flees_all_sessions(tp_session_list)

            label = 'condition{}'.format(tp)
            if label not in stats_dict.keys():
                stats_dict[label] = np.array([n_flees, n_non_flees])
            else:
                stats_dict[label][0] += n_flees
                stats_dict[label][1] += n_non_flees
    return stats_dict


def filter_by_date_range(sessions_list, days_since_habituation, lower_limit, upper_limit):
    sessions_array = np.array(sessions_list)
    days_since_habituation = np.array(days_since_habituation)
    above_lower = days_since_habituation > lower_limit
    below_upper = days_since_habituation < upper_limit
    target_days = np.logical_and(above_lower, below_upper)
    return sessions_array[target_days], days_since_habituation[target_days]


def filter_by_date_exact(sessions_list, days_since_habituation, days):
    sessions_array = np.array(sessions_list)
    days_since_habituation = np.array(days_since_habituation)
    target_days = days_since_habituation == days
    return sessions_array[target_days], days_since_habituation[target_days]


def get_all_p_values(contingency_table_dict):
    all_combinations = list(map(dict, it.combinations(contingency_table_dict.items(), 2)))
    p_values = {}
    for pairing in all_combinations:
        keys = list(pairing.keys())
        results = list(pairing.values())
        new_key = '{}_{}'.format(keys[0], keys[1])
        print(results)
        _, p_value = scipy.stats.fisher_exact([results[0], results[1]])
        p_values[new_key] = p_value
    return p_values


def r_wilcoxon(arr1, arr2):
    pd_arr1 = pd.Series(arr1)
    pd_arr2 = pd.Series(arr2)
    result = rstats.wilcox_test(pd_arr1, pd_arr2, paired=True, exact=True)
    p_val = result[2][0]  # by string?
    return p_val


def get_sorted(flees, days):
    sorted_days = [day for _, day in sorted(zip(flees, days))]
    sorted_flees = [flee for flee, _ in sorted(zip(flees, days))]
    return sorted_flees, sorted_days


