import pandas as pd
import numpy as np

from looming_spots.thesis_figure_plots.signal_latency_plot import get_all_variables
from signal_latency_plot import get_mtgs, LSIE_SNL_KEYS


def get_df(mtg, group):
    normalisation_factor, normalisation_factor_trace, \
    post_test_trials, pre_test_latency, pre_test_trials, \
    theoretical_escape_threshold = get_all_variables(mtg)

    df_dict = {}

    latencies = []
    speeds = []
    escapes = []
    max_integrals_by_end_of_stimulus = []
    max_integrals_by_end_of_5th_loom = []
    trial_numbers = []
    integral_reached_by_latency = []
    percent_of_expected = []

    first_trial = pre_test_trials[0]
    first_trial_latency = int(first_trial.metric_functions['latency peak detect samples']())

    expected_integral_at_escape_onset = first_trial.integral_downsampled()[first_trial_latency]
    difference_from_expected = []

    for t in (pre_test_trials + post_test_trials):
        latency = t.metric_functions['latency peak detect samples']()

        if latency is not None:
            latency = int(latency)
            if latency > 350:
                latency = None

        trial_numbers.append(t.loom_number)
        latencies.append(latency)
        speeds.append(t.metric_functions['speed']())
        escapes.append(t.metric_functions['classified as flee']())

        max_integral_reached_by_end_of_stimulus = np.nanmax(t.integral_downsampled()[:340])
        max_integral_reached_by_5th_loom = np.nanmax(t.integral_downsampled()[:312])

        max_integrals_by_end_of_stimulus.append(max_integral_reached_by_end_of_stimulus)
        max_integrals_by_end_of_5th_loom.append(max_integral_reached_by_5th_loom)

        if latency is not None:
            integral_at_latency = t.integral_downsampled()[latency]
            integral_reached_by_latency.append(integral_at_latency)
            difference_from_expected.append(expected_integral_at_escape_onset - integral_at_latency)
            percent_of_expected.append(integral_at_latency/expected_integral_at_escape_onset)
            print(
                f'loomnumber:{t.loom_number}_{mtg.mouse_id}:::::_LATENCCCY:{latency}: ')
        else:
            print(f'loomnumber:{t.loom_number}_{mtg.mouse_id}:::::_{expected_integral_at_escape_onset - max_integral_reached_by_end_of_stimulus}')
            difference_from_expected.append(expected_integral_at_escape_onset - max_integral_reached_by_end_of_stimulus)
            percent_of_expected.append(max_integral_reached_by_end_of_stimulus/expected_integral_at_escape_onset)
            integral_reached_by_latency.append(np.nan)

    if mtg.mouse_id == '898990':
        escapes = [True] * len(pre_test_trials + post_test_trials)

    theoretical_escape_thresholds = [theoretical_escape_threshold] * len(pre_test_trials + post_test_trials)

    df_dict.setdefault('escape ∆F threshold', theoretical_escape_thresholds)
    df_dict.setdefault('mouse id', [mtg.mouse_id] * len(pre_test_trials + post_test_trials))
    df_dict.setdefault('latency', latencies)
    df_dict.setdefault('speed', speeds)
    df_dict.setdefault('escape', escapes)
    df_dict.setdefault('deltaf max in trial', max_integrals_by_end_of_stimulus)
    df_dict.setdefault('deltaf max in trial up to 5th', max_integrals_by_end_of_5th_loom)
    df_dict.setdefault('integral at latency', integral_reached_by_latency)
    df_dict.setdefault('expected integral', [expected_integral_at_escape_onset]*len(pre_test_trials + post_test_trials))
    df_dict.setdefault('difference from expected', difference_from_expected)
    df_dict.setdefault('percent of expected', percent_of_expected)
    df_dict.setdefault('trial number', trial_numbers)
    df_dict.setdefault('group', [group]*len(pre_test_trials + post_test_trials))
    df_dict.setdefault('trial type', ['pretest']*len(pre_test_trials) +['posttest']*len(post_test_trials))

    return pd.DataFrame.from_dict(df_dict)


def get_df_non_escape_relative_to_estimated_threshold():
    df_all = pd.DataFrame()
    mtgs, groups = get_mtgs(LSIE_SNL_KEYS)
    for mtg, group in zip(mtgs, groups):
        df = get_df(mtg, group)
        df_all = df_all.append(df, ignore_index=True)
    df_all.to_csv('/home/slenzi/thesis_latency_plots/df_threshold_differences.csv')


if __name__ == '__main__':
    get_df_non_escape_relative_to_estimated_threshold()