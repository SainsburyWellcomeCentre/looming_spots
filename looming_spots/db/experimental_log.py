import pandas as pd
import numpy as np

from looming_spots.db import loom_trial_group
from looming_spots.db.dataframe_queries import filter_df
from looming_spots.constants import EXPERIMENTAL_RECORDS_PATH


"""
Mouse records are kept by the user, in a csv file. This module allows easy access to mouse ids that belong
in each experimentally defined group. Usually by keywords defined in the spreadsheet.

"""


def load_df(file_path=EXPERIMENTAL_RECORDS_PATH):
    df = pd.read_csv(file_path)
    df = df[~df["exclude"]]
    return df


def get_mouse_ids_in_experiment(experiment_key):
    df = load_df(EXPERIMENTAL_RECORDS_PATH)
    exp_df = get_experiment_subset_df(df, experiment_key)
    mouse_ids = get_mouse_ids(exp_df)
    return list(mouse_ids)


def get_mtgs_in_experiment(experiment_key):
    mids = get_mouse_ids_in_experiment(experiment_key)
    return [
        loom_trial_group.MouseLoomTrialGroup(mid, experiment_key)
        for mid in mids
    ]


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


def get_result_from_mouse_ids(mouse_ids, df):
    for mid in mouse_ids:
        print(
            df[df["mouse_id"] == mid][
                ["mouse_id", "test_type", "context", "delay", "result"]
            ]
        )


def get_mouse_ids_with_test_combination(
    db, include_test_phases, exclude_test_phases
):
    """
    this is a function that is supposed to filter a database by the experimental sessions that have been carried out
    (i.e. pre- post- and lsie sessions).
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
    include=["post_test", "lsie"],
    exclude=["pre_test"],
    matching_dict_pre_test={"test_type": "== pre_test", "context": "== A9"},
    matching_dict_post_test={"test_type": "== post_test", "context": "== A9"},
    matching_dict_lsie={
        "test_type": "== lsie",
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
    mouse_ids_lsie_combo = (
        get_mouse_ids_from_query(filtered_db, matching_dict_lsie)
        if len(matching_dict_post_test) > 1
        else {}
    )

    return mouse_ids_lsie_combo.intersection(
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
