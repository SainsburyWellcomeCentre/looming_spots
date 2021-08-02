import operator

import pandas as pd
import numpy as np

from looming_spots.db import loom_trial_group
from looming_spots.db.constants import FILE_PATH


def load_df(file_path=FILE_PATH):
    df = pd.read_csv(file_path)
    df = df[~df["exclude"]]
    return df


def get_mouse_ids_in_experiment(experiment_key):
    df = load_df(FILE_PATH)
    exp_df = get_experiment_subset_df(df, experiment_key)
    mouse_ids = get_mouse_ids(exp_df)
    return list(mouse_ids)


def get_mtgs_in_experiment(experiment_key):
    mids = get_mouse_ids_in_experiment(experiment_key)
    return [loom_trial_group.MouseLoomTrialGroup(mid, experiment_key) for mid in mids]


def get_mouse_ids(df, ignore_future_experiments=True):
    df = df[df["result"] != "tbd"]
    mouse_ids = np.array(df["mouse_id"])
    mouse_ids = np.unique(mouse_ids)
    return mouse_ids


def get_experiment_subset_df(df, experiment_key):
    exp_df = df[df["experiment"] == experiment_key]
    return exp_df


def get_incomplete_experiments(df):
    unfinished_experiment_df = df[df["result"] == "tbd"]
    filtered_unfinished_experiment_df = unfinished_experiment_df[
        ["mouse_id", "experiment", "test_date"]
    ]
    return filtered_unfinished_experiment_df


def query_df(df, key, value):
    subset_df = df[df[key] == value]
    return subset_df


def query_df_exclude(df, key, value):
    subset_df = df[df[key] != value]
    return subset_df


def get_result_from_mouse_ids(mouse_ids, df):
    for mid in mouse_ids:
        print(
            df[df["mouse_id"] == mid][
                ["mouse_id", "test_type", "context", "delay", "result"]
            ]
        )


def comparison_from_str(a, cmp_str, b):
    """
    :param a: a subset dataframe
    :param cmp_str: a comparison string (e.g. == something)
    :param b:
    :return:
    """

    comparators = get_comparator_functions()

    b = cast_to_same_type(b, a)
    if isinstance(b, list):
        return a.isin(b)
    return comparators[cmp_str](a, b)


def cast_to_same_type(b, a):
    """
    Converts the query key from a string to the appropriate type in the data series
    assumes all elements of data series are of the same type
    :param b:
    :param a:
    :return:
    """

    a_type_function = type(a.iloc()[0])
    if b.startswith("["):
        b = b.strip("[]").split(",")
        return [a_type_function(b_element.strip()) for b_element in b]

    return a_type_function(b)


def get_comparator_functions():
    comparators = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "and": operator.and_,
        "or": operator.or_,
        "in": operator.contains,
    }

    return comparators


def filter_df(db, filter_dict):

    f_df = db.copy()

    for k, v in filter_dict.items():

        if k not in f_df.keys():
            raise ValueError(k)

        components = v.split(" ", 1)
        cmp_str = components[0]
        cmp_val = components[-1]
        query_result = comparison_from_str(f_df[k], cmp_str, cmp_val)
        n_results = np.count_nonzero(query_result)

        if n_results == 0:
            return []

        f_df = f_df[query_result]

    return f_df


def get_mouse_ids_with_test_combination(
    db, include_test_phases, exclude_test_phases
):
    """
    this is a function that is supposed to filter a database by the experimental sessions that have been carried out
    (i.e. pre- post- and habituation sessions).
    :param db:
    :param include_test_phases: records that have these session types will be included
    :param exclude_test_phases: records that do not have these session types will be excluded
    :return: a list of the mouse ids of
    """
    all_query_results = []

    exclude_results = []

    for test_phase in include_test_phases:
        mouse_ids = get_mouse_ids_from_query(
            db, {"test_type": f"== {test_phase}"}
        )
        all_query_results.append(mouse_ids)

    all_query_results = all_query_results[0].intersection(
        *all_query_results[1:]
    )
    if len(exclude_test_phases) > 0:
        for test_phase in exclude_test_phases:
            mouse_ids = get_mouse_ids_from_query(
                db, {"test_type": f"== {test_phase}"}
            )
            exclude_results.append(mouse_ids)

    if len(exclude_results) > 0:
        exclude_results = exclude_results[0].intersection(*exclude_results[1:])
        return all_query_results - exclude_results

    if len(all_query_results) == 1:
        return all_query_results[0]  # why?

    return all_query_results


def get_mouse_ids_from_query(db, filter_dict):
    filtered_db = filter_df(db, filter_dict)
    if filtered_db is False:
        print("no records of this type found")
        return {}
    return set(filtered_db["mouse_id"])


def get_combination(
    include=["post_test", "habituation"],
    exclude=["pre_test"],
    matching_dict_pre_test={"test_type": "== pre_test", "context": "== A9"},
    matching_dict_post_test={"test_type": "== post_test", "context": "== A9"},
    matching_dict_habituation={
        "test_type": "== habituation",
        "context": "== A9r",
    },
):
    log = load_df()
    mouse_ids = get_mouse_ids_with_test_combination(log, include, exclude)
    filtered_db = get_subset_df_from_mouse_ids(log, mouse_ids)

    mouse_ids_pre_combo = (
        get_mouse_ids_from_query(filtered_db, matching_dict_pre_test)
        if len(matching_dict_pre_test) > 1
        else {}
    )
    mouse_ids_post_combo = (
        get_mouse_ids_from_query(filtered_db, matching_dict_post_test)
        if len(matching_dict_post_test) > 1
        else {}
    )
    mouse_ids_habituation_combo = (
        get_mouse_ids_from_query(filtered_db, matching_dict_habituation)
        if len(matching_dict_post_test) > 1
        else {}
    )

    return mouse_ids_habituation_combo.intersection(
        mouse_ids_post_combo
    )  # make suitable for all 3


def get_pre_tests(
    include=("pre_test",), exclude=(), matching_dict_pre_test=None
):
    """
    matching_dict={
        "test_type": "== pre_test",
        "context": "== A9",
        "line": "== wt",
        "stimulus": "== looming",
        "contrast": "== 0.1600",
        "surgery": "== FALSE",
    }
    :param include:
    :param exclude:
    :param matching_dict_pre_test:
    :return:
    """
    log = load_df()
    mouse_ids = get_mouse_ids_with_test_combination(log, include, exclude)
    filtered_db = get_subset_df_from_mouse_ids(log, mouse_ids)

    mouse_ids_pre_combo = (
        get_mouse_ids_from_query(filtered_db, matching_dict_pre_test)
        if len(matching_dict_pre_test) > 1
        else {}
    )

    return mouse_ids_pre_combo


def get_subset_df_from_mouse_ids(db, mouse_ids):
    return db[db.mouse_id.isin(mouse_ids)]
