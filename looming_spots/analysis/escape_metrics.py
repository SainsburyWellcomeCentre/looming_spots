import pandas as pd


def get_behaviour_metric_dataframe(mtgs, metric, test_type):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        if test_type == "pre_test":
            trials = mtg.pre_test_trials()
        elif test_type == "post_test":
            trials = mtg.post_test_trials()
        elif test_type == "variable_contrast":
            trials = mtg.all_trials[:18]
        event_metric_dict = {}
        vals = []
        for t in trials:
            val = t.metric_functions[
                metric
            ]()  # / t.normalisation_dict[metric]
            vals.append(val)

        mids = [mtg.mouse_id] * len(trials)
        event_metric_dict.setdefault("mouse id", mids)
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


def get_escape_metric_df_trials(trials, metric, mid=None, experimental_condition=None):
    all_df = pd.DataFrame()
    event_metric_dict = {}
    vals = []
    for t in trials:
        val = t.metric_functions[
            metric
        ]()  # / t.normalisation_dict[metric]
        vals.append(val)

    mids = [mid] * len(trials)
    experimental_conditions = [experimental_condition] * len(trials)
    event_metric_dict.setdefault("mouse id", mids)
    event_metric_dict.setdefault("experimental condition", experimental_conditions)
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
        df = get_escape_metric_df_trials(mtg.all_trials, metric, int(mtg.mouse_id), experimental_condition)
        all_df = all_df.append(df)
    return all_df
