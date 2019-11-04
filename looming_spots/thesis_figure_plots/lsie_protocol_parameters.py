from looming_spots.track_analysis.escape_metric_dataframes import (
    get_behaviour_metric_dataframe,
)
from looming_spots.db import experimental_log, loom_trial_group
import pandas as pd


def plot_effect_of_delay_on_habituation_efficacy(metric="classified as flee"):
    records_df = experimental_log.load_df()
    results_df = pd.DataFrame()
    groups = []

    for contrast_protocol in [2, 7]:
        for delay in [0, 10]:

            experiment_string = (
                f"== variable_stimulus_number_habituation_{delay}min_delay"
            )
            gradient_protocol_string = f"== {contrast_protocol}"

            test_contrast_df = experimental_log.filter_df(
                records_df,
                {
                    "experiment": experiment_string,
                    "gradient_protocol": gradient_protocol_string,
                },
            )

            if len(test_contrast_df) > 0:
                test_contrast_mids = test_contrast_df["mouse_id"].unique()
                groups.append(test_contrast_mids)

            gradient_contrast_df = experimental_log.filter_df(
                records_df,
                {
                    "experiment": experiment_string,
                    "gradient_protocol": gradient_protocol_string,
                },
            )

            if len(gradient_contrast_df) > 0:
                gradient_contrast_mids = test_contrast_df["mouse_id"].unique()
                groups.append(gradient_contrast_mids)

            for mid_group in groups:
                mtgs = [
                    loom_trial_group.MouseLoomTrialGroup(mid)
                    for mid in mid_group
                ]
                df = get_behaviour_metric_dataframe(mtgs, metric, "post_test")
                df["experiment"] = [experiment_string.split(" ")[-1]] * len(df)
                df["contrast_protocol"] = [contrast_protocol] * len(df)
                df["delay"] = [delay] * len(df)
                results_df = results_df.append(df)

    return results_df


def n_stimuli_effect(metric="classified as flee"):
    records_df = experimental_log.load_df()
    results_df = pd.DataFrame()

    for contrast_protocol in [14, 15, 16]:
        for delay in [0]:
            groups = []
            experiment_string = (
                f"== variable_stimulus_number_habituation_{delay}min_delay"
            )
            gradient_protocol_string = f"== {contrast_protocol}"

            test_contrast_df = experimental_log.filter_df(
                records_df,
                {
                    "experiment": experiment_string,
                    "gradient_protocol": gradient_protocol_string,
                },
            )

            if len(test_contrast_df) > 0:
                test_contrast_mids = test_contrast_df["mouse_id"].unique()
                groups.append(test_contrast_mids)

            for mid_group in groups:
                print(
                    "contrast protocol: {contrast_protocol}, mouse ids: {mid_group}"
                )
                mtgs = [
                    loom_trial_group.MouseLoomTrialGroup(mid)
                    for mid in mid_group
                ]
                df = get_behaviour_metric_dataframe(mtgs, metric, "post_test")
                df["experiment"] = [experiment_string.split(" ")[-1]] * len(df)
                df["contrast_protocol"] = [contrast_protocol] * len(df)
                df["delay"] = [delay] * len(df)
                results_df = results_df.append(df)

    return results_df
