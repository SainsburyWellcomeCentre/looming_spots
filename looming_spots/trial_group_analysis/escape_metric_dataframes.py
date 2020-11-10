import pandas as pd

import numpy as np
from looming_spots.db import loom_trial_group


def get_behaviour_metric_dataframe(mtgs, metric, test_type):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        if test_type == "pre_test":
            trials = mtg.pre_test_trials()[:3]
        elif test_type == "post_test":
            trials = mtg.post_test_trials()[:3]
        elif test_type == "variable_contrast":
            trials = mtg.loom_trials()[:18]

        event_metric_dict = {}
        vals = []
        for t in trials:
            val = t.metric_functions[metric]()
            vals.append(val)

        mids = [mtg.mouse_id] * len(trials)
        event_metric_dict.setdefault("mouse id", mids)
        event_metric_dict.setdefault(
            "loom number", [t.loom_trial_idx for t in trials]  #get_stimulus_number()
        )
        event_metric_dict.setdefault("metric value", vals)
        event_metric_dict.setdefault(
            "test type", ["variable contrast"] * len(trials)
        )
        event_metric_dict.setdefault("metric", [metric] * len(trials))
        event_metric_dict.setdefault("contrast", [t.contrast for t in trials])
        event_metric_dict.setdefault("escape", [t.is_flee() for t in trials])
        metric_df = pd.DataFrame.from_dict(event_metric_dict)
        all_df = all_df.append(metric_df, ignore_index=True)
    return all_df


def get_escape_metric_df_trials(
    trials, metric, mid=None, experimental_condition=None
):
    all_df = pd.DataFrame()
    event_metric_dict = {}
    vals = []
    for t in trials:
        val = t.metric_functions[metric]()
        vals.append(val)

    mids = [mid] * len(trials)
    experimental_conditions = [experimental_condition] * len(trials)
    event_metric_dict.setdefault("mouse id", mids)
    event_metric_dict.setdefault(
        "experimental condition", experimental_conditions
    )
    event_metric_dict.setdefault(
        "loom number", [t.get_stimulus_number() for t in trials]
    )
    event_metric_dict.setdefault("metric value", vals)
    event_metric_dict.setdefault(
        "test type", ["variable contrast"] * len(trials)
    )
    event_metric_dict.setdefault("metric", [metric] * len(trials))
    event_metric_dict.setdefault("contrast", [t.contrast for t in trials])
    event_metric_dict.setdefault("escape", [t.is_flee() for t in trials])
    metric_df = pd.DataFrame.from_dict(event_metric_dict)
    all_df = all_df.append(metric_df, ignore_index=True)
    return all_df


def get_escape_metrics_mtgs(mtgs, metric, experimental_condition=None):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        df = get_escape_metric_df_trials(
            mtg.all_trials, metric, int(mtg.mouse_id), experimental_condition
        )
        all_df = all_df.append(df)
    return all_df


def get_behaviour_metrics_df(mids, group_label, metrics):
    all_df = pd.DataFrame()
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    base_df = get_behaviour_metric_dataframe()
    for metric in metrics:
        values = get_metric_values(metric)
    return all_df


def get_behaviour_metrics_dataframe(mtgs, metrics, test_type, experimental_group_label):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        trials = get_trials(mtg, test_type)
        event_metric_dict = get_base_df_dict(mtg, test_type, trials)

        add_all_metrics(event_metric_dict, metrics, trials)
        event_metric_dict.setdefault('experimental group', [experimental_group_label]*len(trials))
        metric_df = pd.DataFrame.from_dict(event_metric_dict)
        all_df = all_df.append(metric_df, ignore_index=True)
    return all_df


def add_all_metrics(event_metric_dict, metrics, trials):
    for metric in metrics:
        vals = get_metric_values(metric, trials)
        event_metric_dict.setdefault(metric, vals)


def get_base_df_dict(mtg, test_type, trials):
    event_metric_dict = {}
    mids = [mtg.mouse_id] * len(trials)
    event_metric_dict.setdefault("mouse id", mids)
    event_metric_dict.setdefault(
        "loom number", [t.loom_trial_idx for t in trials]  # get_stimulus_number()
    )
    event_metric_dict.setdefault(
        "test type", [test_type] * len(trials)
    )
    event_metric_dict.setdefault("contrast", [t.contrast for t in trials])
    event_metric_dict.setdefault("escape", [t.is_flee() for t in trials])
    return event_metric_dict


def get_metric_values(metric, trials):
    vals = []
    for t in trials:
        val = t.metric_functions[metric]()
        vals.append(val)
    return vals


def get_trials(mtg, test_type):
    if test_type == "pre_test":
        trials = mtg.pre_test_trials()[:3]
    elif test_type == "post_test":
        trials = mtg.post_test_trials()[:3]
    elif test_type == "variable_contrast":
        trials = mtg.loom_trials()[:18]
    return trials


def get_track_dataframe(mtgs, test_type):
    all_df = pd.DataFrame()

    for mtg in mtgs:
        trials = get_trials(mtg, test_type)
        for t in trials:
            track_dict = {}
            x = t.normalised_x_track[:600]
            y = t.normalised_x_track[:600]
            track_dict.setdefault('x', x)
            track_dict.setdefault('y', x)
            track_dict.setdefault('timepoint', np.arange(len(x)))
            track_dict.setdefault('loom number', [t.loom_trial_idx]*len(x))
            track_dict.setdefault('mid', [mtg.mouse_id]*len(x))
            metric_df = pd.DataFrame.from_dict(track_dict)
            all_df = all_df.append(metric_df, ignore_index=True)
    return all_df

