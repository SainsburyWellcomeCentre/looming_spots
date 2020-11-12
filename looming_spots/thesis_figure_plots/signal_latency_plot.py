from looming_spots.db import loom_trial_group, experimental_log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from looming_spots.db.constants import LOOM_ONSETS
from looming_spots.thesis_figure_plots import photometry_example_traces

ALL_SNL_KEYS = ['photometry_habituation_tre-GCaMP_24hr_pre',
                'photometry_habituation_tre-GCaMP_same_day_pre',
                'photometry_habituation_tre-GCaMP-contrasts']

LSIE_SNL_KEYS = ['photometry_habituation_tre-GCaMP_24hr_pre',
                'photometry_habituation_tre-GCaMP_same_day_pre',]


def get_pre_test_and_high_contrast_trials_mtg(mtg):
    trials = mtg.pre_test_trials()
    return [t for t in trials if t.contrast == 0]


def get_pre_test_and_high_contrast_trials(mtgs):
    all_trials = []
    for mtg in mtgs:
        if mtg.exp_key == 'photometry_habituation_tre-GCaMP-contrasts':
            trials = [t for t in mtg.all_trials[:18] if t.contrast == 0]
        else:
            trials = mtg.pre_test_trials()[:3]
        all_trials.extend(trials)
    return all_trials


def get_snl_pre_test_and_high_contrast_trials():
    mtgs = get_mtgs(ALL_SNL_KEYS)
    trials = get_pre_test_and_high_contrast_trials(mtgs)
    for t in trials:
        fig = plt.figure()
        title = f'deltaF_with_track__mouse_{t.mouse_id}__trial_{t.loom_number}'
        plt.title(title)
        t.plot_delta_f_with_track('k')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.png')
        fig.close()


def get_mtgs(keys):
    mtgs = []
    for key in keys:
        mtgs.extend(experimental_log.get_mtgs_in_experiment(key))
    return mtgs


def calculate_theoretical_escape_threshold(mtg):
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])

    theoretical_escape_threshold = np.mean([t.integral_escape_metric(int(t.latency_peak_detect())) for t in pre_test_trials])

    print('latencies:', [int(t.latency_peak_detect()) for t in pre_test_trials])
    print('min thresholds:', [np.max(t.integral_escape_metric(int(t.latency_peak_detect()))) for t in pre_test_trials])

    theoretical_escape_threshold_minimum = np.min([np.max(t.integral_escape_metric(int(t.latency_peak_detect()))) for t in pre_test_trials])
    theoretical_escape_threshold_maximum = np.max([np.max(t.integral_escape_metric(int(t.latency_peak_detect()))) for t in pre_test_trials])

    for t in post_test_trials:
        latency = t.latency_peak_detect()
        title = f'theoretical_threshold_{mtg.mouse_id}__loom_number_{t.loom_trial_idx}'
        fig, axes = plt.subplots(3, 1)
        plt.title(title)
        plt.sca(axes[0])
        plt.axhline(theoretical_escape_threshold, color='k', linewidth=2)
        #[plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]

        #plot average latency to escape in pre test

        plt.axvline(int(pre_test_latency), color='r')

        #plot integral at latency
        if latency is not None:
            print(f'latency: {latency}')
            if latency < 600:
                plt.axhline(t.integral_downsampled()[int(latency)], color='b', ls='--')
                plt.axvline(latency, color='r', ls='--')

        plt.axhline(np.nanmax(t.integral_downsampled()[:335]), color='b')
        plt.plot(t.integral_downsampled())
        plt.xlim([0, 600])
        plt.axhspan(theoretical_escape_threshold_minimum, theoretical_escape_threshold_maximum, color='r', alpha=0.2)
        t.plot_stimulus()
        plt.sca(axes[1])

        if latency is not None:
            print(f'latency: {latency}')
            if latency < 600:
                plt.axvline(latency, color='r', ls='--')
        t.plot_stimulus()
        #[plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.plot(t.normalised_x_track[:600])
        plt.sca(axes[1])
        plt.ylim([0, 1])
        plt.xlim([0, 600])
        plt.sca(axes[2])
        t.plot_delta_f_with_track('k')

        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.eps',format='eps')
        plt.close()
    plot_pre_test_trial(mtg, pre_test_trials)


def plot_pre_test_trial(mtg, pre_test_trials):
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])

    for t in pre_test_trials:
        fig, axes = plt.subplots(2, 1)
        plt.sca(axes[0])
        latency = t.latency_peak_detect()
        title = f'pre_test__{mtg.mouse_id}__loom_number_{t.loom_number}'
        plt.title(title)
        plt.axhline(t.integral_escape_metric(int(latency)), color='g')
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.plot(t.integral_downsampled())
        plt.axvline(int(t.latency_peak_detect()), color='r', ls='--')
        plt.ylim([0, np.nanmax(t.integral_downsampled())])
        plt.xlim([0, 600])

        plt.sca(axes[1])
        t.plot_delta_f_with_track()
        plt.axvline(int(t.latency_peak_detect()), color='r', ls='--')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.eps', format='eps')
        plt.close()


def plot_all_theoretical_escape_thresholds():
    mtgs = get_mtgs(LSIE_SNL_KEYS)
    for mtg in mtgs:
        calculate_theoretical_escape_threshold(mtg)


def get_df_non_escape_relative_to_estimated_threshold_mtg(mtg):
    df_dict = {}
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])
    theoretical_escape_threshold = np.mean(
        [t.integral_escape_metric(int(pre_test_latency)) for t in pre_test_trials])


    latencies = []
    speeds = []
    escapes = []
    delta_f_metrics = []
    delta_f_metrics_short = []

    for t in post_test_trials:
        latencies.append(t.metric_functions['latency peak detect samples']())
        speeds.append(t.metric_functions['speed']())
        escapes.append(t.metric_functions['classified as flee']())
        delta_f_metrics.append(np.nanmax(t.integral_downsampled()[:335]))
        delta_f_metrics_short.append(np.nanmax(t.integral_downsampled()[:312]))

    if mtg.mouse_id == '898990':
        escapes = [True] * 3
    theoretical_escape_thresholds = [theoretical_escape_threshold] * len(post_test_trials)

    df_dict.setdefault('escape ∆F threshold', theoretical_escape_thresholds)
    df_dict.setdefault('mouse id', [mtg.mouse_id] * len(post_test_trials))
    df_dict.setdefault('latency', latencies)
    df_dict.setdefault('speed', speeds)
    df_dict.setdefault('escape', escapes)
    df_dict.setdefault('deltaf max in trial', delta_f_metrics)
    df_dict.setdefault('deltaf max in trial up to 5th', delta_f_metrics_short)
    return pd.DataFrame.from_dict(df_dict)


def get_df_non_escape_relative_to_estimated_threshold():
    df_all = pd.DataFrame()
    mtgs = get_mtgs(LSIE_SNL_KEYS)
    for mtg in mtgs:
        df = get_df_non_escape_relative_to_estimated_threshold_mtg(mtg)
        df_all = df_all.append(df, ignore_index=True)
    df_all['exceeds theoretical threshold']  =  df_all['deltaf max in trial'] > df_all['escape ∆F threshold']
    df_all['exceeds theoretical threshold short']  =  df_all['deltaf max in trial up to 5th'] > df_all['escape ∆F threshold']

    df_all.to_csv('/home/slenzi/thesis_latency_plots/df_2.csv')

def replot_lsie():
    mids24 = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[0])
    mids_sameday = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[1])

    fig= photometry_example_traces.plot_LSIE_bars_all_groups(groups=(mids24, mids_sameday))
    fig.savefig('/home/slenzi/thesis_latency_plots/LSIE.eps',format='eps')

def main():
    import seaborn as sns
    sns.set_style("white")
    #get_snl_pre_test_and_high_contrast_trials()
    plot_all_theoretical_escape_thresholds()
    get_df_non_escape_relative_to_estimated_threshold()
    replot_lsie()

if __name__ == '__main__':
    main()
