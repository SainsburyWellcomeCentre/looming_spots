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
        trials = get_high_contrast_naive_trials(mtg)
        all_trials.extend(trials)
    return all_trials


def get_high_contrast_naive_trials(mtg):
    if mtg.exp_key == 'photometry_habituation_tre-GCaMP-contrasts':
        trials = [t for t in mtg.all_trials[:18]]
    else:
        trials = mtg.pre_test_trials()[:3]
    return trials


def get_signal_df_mtgs(groups, timepoint=215):
    all_df = pd.DataFrame()
    for group in groups:
        mtgs = experimental_log.get_mtgs_in_experiment(group)

        for mtg in mtgs:
            mtg_dict = {}
            vals = []
            escapes = []
            contrasts = []
            trials = get_high_contrast_naive_trials(mtg)
            if timepoint is None:
                timepoint = np.mean([t.latency_peak_detect() for t in trials])
            normalising_factor = photometry_example_traces.get_normalisation_factor(mtg, timepoint)

            for t in trials:
                val = (
                        t.integral_escape_metric(int(timepoint))
                        / normalising_factor
                )
                vals.append(val)
                escapes.append(t.is_flee())
                contrasts.append(t.contrast)

            for metric in mtg.analysed_metrics():
                metric_vals = []
                for t in trials:
                    metric_vals.append(t.metric_functions[metric]())
                mtg_dict.setdefault(metric, metric_vals)

            mtg_dict.setdefault("group", [group] * len(trials))
            mtg_dict.setdefault("deltaf metric", vals)
            mtg_dict.setdefault("contrast", contrasts)
            mtg_dict.setdefault("escape", escapes)
            mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(trials))
            mtg_df = pd.DataFrame.from_dict(mtg_dict)
            all_df = all_df.append(mtg_df, ignore_index=True)
    all_df.to_csv(f'/home/slenzi/thesis_latency_plots/signal_df_naive_all_ctst.csv')
    return all_df


def get_snl_pre_test_and_high_contrast_trials():
    mtgs = get_mtgs(['photometry_habituation_tre-GCaMP-contrasts'])
    trials = get_pre_test_and_high_contrast_trials(mtgs)
    for t in trials:
        fig = plt.figure()
        title = f'deltaF_with_track__mouse_{t.mouse_id}__trial_{t.loom_number}'
        plt.title(title)
        t.plot_delta_f_with_track('k')
        plt.axis('off')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.png')
        fig.close()


def get_mtgs(keys):
    mtgs = []
    for key in keys:
        mtgs.extend(experimental_log.get_mtgs_in_experiment(key))
    return mtgs


def calculate_theoretical_escape_threshold(mtg, fig=None, axes=None):
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])

    pre_test_trial_integral_metric_values = [t.integral_escape_metric(int(pre_test_latency)) for t in pre_test_trials]

    normalisation_factor = np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])
    normalisation_factor_trace = np.nanmax([np.nanmax(t.delta_f()[200:350]) for t in mtg.loom_trials()])
    theoretical_escape_threshold = np.mean(pre_test_trial_integral_metric_values) / normalisation_factor
    theoretical_escape_threshold_minimum = np.min(pre_test_trial_integral_metric_values) / normalisation_factor
    theoretical_escape_threshold_maximum = np.max(pre_test_trial_integral_metric_values) / normalisation_factor

    for t in post_test_trials:
        #if not (t.is_flee() or mtg.mouse_id == '898990'):

        latency = t.latency_peak_detect()

        #if fig is None:

        #else:
        fname = f'theoretical_threshold_all_{mtg.mouse_id}_pre'

        #fig, axes = plt.subplots(2, 1)
        #fname = f'theoretical_threshold_{mtg.mouse_id}__loom_number_{t.loom_trial_idx}_avg_latency_metric_{t.is_flee()}_2'
        #title = f'{mtg.mouse_id}__loom_number_{t.loom_trial_idx}'
        #plt.title(title)


        plt.sca(axes[0])
        max_val_reached = np.nanmax((t.integral_downsampled()/normalisation_factor)[:335])
        if max_val_reached > theoretical_escape_threshold:
            color = 'r'
        else:
            color = 'k'

        if (t.is_flee() or mtg.mouse_id == '898990'):
            color='r'
        else:
            color='k'

        plt.axhline(theoretical_escape_threshold, color=color, linewidth=2)
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]

        plt.plot(t.integral_downsampled()/normalisation_factor, color=color)
        plt.xlim([180, 370])
        plt.hlines(0.5, 250, 280)
        plt.vlines(250, 0.5, 0.6)
        #plot_optional_metrics(latency, pre_test_latency, t)
        #plt.axhspan(theoretical_escape_threshold_minimum, theoretical_escape_threshold_maximum, color='r', alpha=0.2)
        #t.plot_stimulus()
        plt.axis('off')

        plt.sca(axes[1])
        #plot_latency(latency)
        t.plot_delta_f_with_track(norm_factor=normalisation_factor_trace)
        plt.ylim([0, 1])
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.xlim([180, 370])

        plt.axis('off')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{fname}.eps', format='eps')


def plot_latency(latency):
    if latency is not None:
        print(f'latency: {latency}')
        if latency < 600:
            plt.axvline(latency, color='r', ls='--')


def plot_optional_metrics(latency, pre_test_latency, t, max_val_reached):
    if latency is not None:
        print(f'latency: {latency}')
        if latency < 600:
            plt.axhline(t.integral_downsampled()[int(latency)], color='b', ls='--')
            plt.axvline(latency, color='b', ls='--')
    plt.axvline(pre_test_latency, color='r', ls='--')
    plt.axhline(max_val_reached, color='b')



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
        plt.axis('off')

        plt.sca(axes[1])
        t.plot_delta_f_with_track()
        plt.axvline(int(t.latency_peak_detect()), color='r', ls='--')
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        title += escape_on(t.latency_peak_detect())
        plt.axis('off')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.eps', format='eps')
        plt.close()


def escape_on(latency):
    if latency < LOOM_ONSETS[1]:
        return '_1st_loom_escape'
    elif LOOM_ONSETS[1] < latency < LOOM_ONSETS[2]:
        return '_2nd_loom_escape'
    elif LOOM_ONSETS[2] < latency < LOOM_ONSETS[3]:
        return '_3rd_loom_escape'
    elif LOOM_ONSETS[3] < latency < LOOM_ONSETS[4]:
        return '_4th_loom_escape'
    else:
        return '_5th_loom_escape'


def plot_all_theoretical_escape_thresholds():
    mtgs = get_mtgs(LSIE_SNL_KEYS[0:1])

    for mtg in mtgs:
        fig, axes = plt.subplots(2, 1)
        calculate_theoretical_escape_threshold(mtg, fig=fig, axes=axes)


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


def plot_all_integrals():
    mtgs = get_mtgs(LSIE_SNL_KEYS)
    for mtg in mtgs:
        plot_pre_post_integral(mtg)


def plot_pre_post_integral(mtg):
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])

    fig = plt.figure()
    for t in pre_test_trials:
        plt.plot(t.integral_downsampled(), color='r')

    for t in post_test_trials:
        plt.plot(t.integral_downsampled(), color='k')

    plt.axvline(int(pre_test_latency), color='r')
    #t.plot_stimulus()
    [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]

    plt.hlines(0.0, 500, 530)
    plt.vlines(500, 0.0, 0.01)
    plt.xlim([0, 600])
    title = f'integral_pre_post__{mtg.mouse_id}'
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.eps', format='eps')


def replot_lsie():
    mids24 = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[0])
    mids_sameday = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[1])

    fig= photometry_example_traces.plot_LSIE_bars_all_groups(groups=(mids24, mids_sameday))

    fig.savefig('/home/slenzi/thesis_latency_plots/LSIE.eps', format='eps')


def plot_snl_signal_escape_latency():
    get_signal_df_mtgs(ALL_SNL_KEYS, 215)


def main():
    import seaborn as sns
    sns.set_style("white")
    #get_snl_pre_test_and_high_contrast_trials()
    plot_all_theoretical_escape_thresholds()
    #plot_all_theoretical_escape_thresholds()
    #plot_snl_signal_escape_latency()
    #get_df_non_escape_relative_to_estimated_threshold()
    #replot_lsie()
    #plot_all_integrals()


if __name__ == '__main__':
    main()
