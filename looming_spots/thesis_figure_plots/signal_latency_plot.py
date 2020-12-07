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
    groups=[]
    for key in keys:
        new_mtgs = experimental_log.get_mtgs_in_experiment(key)
        mtgs.extend(new_mtgs)
        groups.extend([key]*len(new_mtgs))
    return mtgs, groups


def plot_mouse_pre_post_tests(mtg, normalisation_factor,
                              normalisation_factor_trace, post_test_trials, pre_test_trials,
                              theoretical_escape_threshold, pre_test_latency, label='pre_post'):
    fig, axes = plt.subplots(2, 2)
    fname = f'theoretical_threshold_pre_post_{mtg.mouse_id}_{label}'
    for t in pre_test_trials:
        plt.sca(axes[0][0])
        color = 'r' if t.is_flee() else 'k'
        plt.axhline(theoretical_escape_threshold, color=color, linewidth=2)
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.plot(t.integral_downsampled() / normalisation_factor, color=color)
        plt.xlim([180, 370])
        plt.ylim([0, 3*theoretical_escape_threshold])
        plt.hlines(0.5, 250, 280)
        plt.vlines(250, 0.5, 0.6)

        plt.axis('off')
        plt.sca(axes[1][0])
        t.plot_delta_f_with_track(norm_factor=normalisation_factor_trace, color=color)
        plt.ylim([0, 1])
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.xlim([180, 370])
        plt.axis('off')

    for t in post_test_trials:
        plt.sca(axes[0][1])
        if (t.is_flee() or mtg.mouse_id == '898990'):
            color = 'r'
        else:
            color = 'k'
        plt.axhline(theoretical_escape_threshold, color=color, linewidth=2)
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.plot(t.integral_downsampled() / normalisation_factor, color=color)
        plt.xlim([180, 370])
        plt.ylim([0, 3 * theoretical_escape_threshold])
        plt.hlines(0.5, 250, 280)
        plt.vlines(250, 0.5, 0.6)

        plt.axis('off')
        plt.sca(axes[1][1])
        t.plot_delta_f_with_track(norm_factor=normalisation_factor_trace, color=color)
        plt.ylim([0, 1])
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.xlim([180, 370])
        plt.axis('off')
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{fname}.eps', format='eps')


def calculate_theoretical_escape_threshold(mtg, fig=None, axes=None, label=None):
    normalisation_factor, normalisation_factor_trace, \
    post_test_trials, pre_test_latency, pre_test_trials, \
    theoretical_escape_threshold = get_all_variables(mtg)

    fname = f'theoretical_threshold_all_post_{label}_split_by_exceed'

    plot_mouse_trials_separate_scaled(label, mtg, normalisation_factor,
                                      normalisation_factor_trace, pre_test_trials,
                                      theoretical_escape_threshold, pre_test_latency)
    plot_mouse_pre_post_tests(mtg, normalisation_factor, normalisation_factor_trace,
                              post_test_trials, pre_test_trials, theoretical_escape_threshold, pre_test_latency)

    for t in post_test_trials:

        max_val_reached = np.nanmax((t.integral_downsampled() / normalisation_factor)[:335])
        if max_val_reached > theoretical_escape_threshold:
            plot_threshold_and_sub_threshold_trialwise(axes[:2], mtg, normalisation_factor,
                                                       normalisation_factor_trace, t, theoretical_escape_threshold)
        else:
            plot_threshold_and_sub_threshold_trialwise(axes[-2:], mtg, normalisation_factor,
                                                       normalisation_factor_trace, t, theoretical_escape_threshold)
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{fname}.eps', format='eps')


def get_all_variables(mtg):
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])
    pre_test_trial_integral_metric_values = [t.integral_escape_metric(int(pre_test_latency)) for t in pre_test_trials]
    normalisation_factor = np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])
    normalisation_factor_trace = np.nanmax([np.nanmax(t.delta_f()[200:350]) for t in mtg.loom_trials()])
    theoretical_escape_threshold = np.mean(pre_test_trial_integral_metric_values) / normalisation_factor
    theoretical_escape_threshold_minimum = np.min(pre_test_trial_integral_metric_values) / normalisation_factor
    theoretical_escape_threshold_maximum = np.max(pre_test_trial_integral_metric_values) / normalisation_factor

    return normalisation_factor, normalisation_factor_trace, post_test_trials, pre_test_latency, \
           pre_test_trials, theoretical_escape_threshold


def plot_mouse_trials_separate_scaled(label, mtg, normalisation_factor, normalisation_factor_trace,
                                      post_test_trials, theoretical_escape_threshold, pre_test_latency):
    fig2, axes = plt.subplots(2, len(post_test_trials))
    fname = f'theoretical_threshold_all_trials_{label}_{mtg.mouse_id}_to_scale_pre_test'
    row_1_axes = axes[0]
    row_2_axes = axes[1]

    for i, t in enumerate(post_test_trials):
        plt.sca(row_1_axes[i])

        if (t.is_flee() or mtg.mouse_id == '898990'):
            color = 'r'
        else:
            color = 'k'
        plot_optional_metrics(pre_test_latency, t, normalisation_factor)
        plt.axhline(theoretical_escape_threshold, color=color, linewidth=2)
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.plot(t.integral_downsampled()[:int(t.latency_peak_detect())] / normalisation_factor, color=color)
        plt.plot(t.integral_downsampled()[:int(t.latency_peak_detect())] / normalisation_factor, color=color)
        plt.xlim([180, 370])
        plt.ylim([0, 3*theoretical_escape_threshold])
        plt.hlines(0.5, 250, 280)
        plt.vlines(250, 0.5, 0.6)

        plt.axis('off')
        plt.sca(row_2_axes[i])
        t.plot_delta_f_with_track(norm_factor=normalisation_factor_trace, color=color)
        plt.ylim([0, 1])
        [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
        plt.xlim([180, 370])
        plt.axis('off')
    fig2.savefig(f'/home/slenzi/thesis_latency_plots/{fname}.eps', format='eps')


def plot_threshold_and_sub_threshold_trialwise(axes, mtg, normalisation_factor, normalisation_factor_trace,
                                               t, theoretical_escape_threshold):
    plt.sca(axes[0])

    if (t.is_flee() or mtg.mouse_id == '898990'):
        color = 'r'
    else:
        color = 'k'

    plt.axhline(theoretical_escape_threshold / theoretical_escape_threshold, color=color, linewidth=2)
    [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
    plt.plot(t.integral_downsampled() / theoretical_escape_threshold / normalisation_factor, color=color) # normalisation_factor
    plt.xlim([180, 370])
    plt.hlines(0.5, 250, 280)
    plt.vlines(250, 0.5, 0.6)

    plt.axis('off')
    plt.sca(axes[1])
    t.plot_delta_f_with_track(norm_factor=normalisation_factor_trace, color=color)
    plt.ylim([0, 1])
    [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
    plt.xlim([180, 370])
    plt.axis('off')


def plot_all_integrals_normalised_to_threshold(mtgs, label):
    fname = f'all_post_tests_integrals_normalised_to_escape_threshold_{label}__full'
    fig, axes = plt.subplots(2,1)
    for mtg in mtgs:
        normalisation_factor, normalisation_factor_trace, \
        post_test_trials, pre_test_latency, pre_test_trials, \
        theoretical_escape_threshold = get_all_variables(mtg)

        ax=plt.sca(axes[0])

        for t in pre_test_trials:
            if (t.is_flee() or mtg.mouse_id == '898990'):
                color = 'r'
            else:
                color = 'k'

            #plt.axhline(theoretical_escape_threshold / theoretical_escape_threshold, color=color, linewidth=2)
            [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]
            latency = t.latency_peak_detect()
            if latency is None:
                latency = 600
            latency=600
            delta_f_normalised = t.delta_f()[:600] / normalisation_factor_trace
            plt.plot(t.get_integral(delta_f_normalised)[:int(latency)], color=color)

            #plt.plot(t.integral_downsampled()[:int(latency)] / normalisation_factor, color=color) # normalisation_factor
            plt.xlim([180, 370])
            plt.ylim([0, normalisation_factor_trace*20])
            plt.hlines(0.5, 250, 280)
            plt.vlines(250, 0.5, 0.6)

            plt.axis('off')

        plt.sca(axes[1])
        for t in post_test_trials:
            if (t.is_flee() or mtg.mouse_id == '898990'):
                color = 'r'
            else:
                color = 'k'

            #plt.axhline(theoretical_escape_threshold / theoretical_escape_threshold, color=color, linewidth=2)
            [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]

            latency = t.latency_peak_detect()
            if latency is None:
                latency = 600
            latency = 600
            #plt.plot(t.integral_downsampled()[:int(latency)] / normalisation_factor, color=color) # normalisation_factor
            delta_f_normalised = t.delta_f()[:600] / normalisation_factor_trace
            plt.plot(t.get_integral(delta_f_normalised)[:int(latency)], color=color)
            plt.ylim([0, normalisation_factor_trace*20])
            plt.xlim([180, 370])
            plt.hlines(0.5, 250, 280)
            plt.vlines(250, 0.5, 0.6)

            plt.axis('off')
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{fname}_all.eps', format='eps')



def plot_latency(latency):
    if latency is not None:
        print(f'latency: {latency}')
        if latency < 600:
            plt.axvline(latency, color='r', ls='--')


def plot_optional_metrics(pre_test_latency, t, normalisation_factor, max_val_reached=None):
    latency = t.latency_peak_detect()
    if latency is not None:
        print(f'latency: {latency}')
        if latency < 600:
            plt.axhline(t.integral_downsampled()[int(latency)]/ normalisation_factor, color='b', ls='--')
            plt.axvline(latency, color='b', ls='--')
    plt.axvline(pre_test_latency, color='r', ls='--')
    if max_val_reached is not None:
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
    for k in LSIE_SNL_KEYS:
        mtgs = get_mtgs([k])
        plot_all_integrals_normalised_to_threshold(mtgs, k)

        fig, axes = plt.subplots(4, 1)
        for mtg in mtgs:
            calculate_theoretical_escape_threshold(mtg, fig=fig, axes=axes, label=k)

    mtgs = get_mtgs(LSIE_SNL_KEYS)
    plot_all_integrals_normalised_to_threshold(mtgs, 'all')


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
    normalisation_factor, normalisation_factor_trace, \
    post_test_trials, pre_test_latency, pre_test_trials, \
    theoretical_escape_threshold = get_all_variables(mtg)

    colors = ['b', 'g', 'orange']
    for t, c in zip(pre_test_trials, colors):
        latency = t.latency_peak_detect()
        if latency is None:
            latency = 600
        #latency = 600
        delta_f_normalised = t.delta_f()[:600]/normalisation_factor_trace
        plt.plot(t.get_integral(delta_f_normalised)[:int(latency)], color=c)

    for t in post_test_trials:
        latency = t.latency_peak_detect()
        if latency is None:
            latency = 600
        #latency = 600
        delta_f_normalised = t.delta_f()[:600] / normalisation_factor_trace
        plt.plot(t.get_integral(delta_f_normalised)[:int(latency)], color='k')
        #plt.plot(t.integral_downsampled()[:int(latency)], color='k')

    plt.axvline(int(pre_test_latency), color='r')
    #t.plot_stimulus()
    [plt.axvline(x, color='k', ls='--') for x in LOOM_ONSETS]

    plt.hlines(0.0, 500, 530)
    plt.vlines(500, 0.0, 0.01)
    plt.xlim([0, 600])
    title = f'integral_pre_post__{mtg.mouse_id}_to_escape_onset'
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.eps', format='eps')




def replot_lsie():
    mids24 = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[0])
    mids_sameday = experimental_log.get_mouse_ids_in_experiment(LSIE_SNL_KEYS[1])

    fig= photometry_example_traces.plot_LSIE_bars_all_groups(groups=(mids24, mids_sameday))

    fig.savefig('/home/slenzi/thesis_latency_plots/LSIE.eps', format='eps')


def plot_snl_signal_escape_latency():
    get_signal_df_mtgs(ALL_SNL_KEYS, 215)


def plot_895773_latency_escapes():
    mtg = loom_trial_group.MouseLoomTrialGroup('895773')
    fig = plt.figure()
    pre_test_trials = mtg.pre_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])
    for t in pre_test_trials:
        t.plot_track()
        plt.axvline(pre_test_latency)
    t.plot_stimulus()
    fig.savefig('/home/slenzi/thesis_latency_plots/example_avg_latency.eps', format='eps')


def ohda_nmda_first_trial():
    pass


def count_manually_tracked_videos():
    pass


def return_to_shelter_photometry_no_stimulus(mtg):
    pass


def difference_between_expected_threshold_and_actual_signal():
    pass


def plot_pre_test_trials_with_predicted_values(mtg):
    fig = plt.figure()
    pre_test_trials = mtg.pre_test_trials()[:3]
    first_trial = pre_test_trials[0]
    first_latency = first_trial.latency_peak_detect()
    if first_latency is not None:
        first_thresh = first_trial.integral_downsampled()[int(first_latency)]
        for t, color in zip(pre_test_trials, ['b', 'g', 'orange']):
            latency = t.latency_peak_detect()
            if latency is not None:
                val_at_latency = t.integral_downsampled()[int(latency)]
                scale_factor = float(first_thresh) / float(val_at_latency)
                plt.plot(t.integral_downsampled()[:int(latency)], color=color)
                plt.plot(t.integral_downsampled()[:int(latency)+1]*scale_factor, linestyle='dotted', color=color)
                print(f'FIRST: {first_thresh}, TRIAL: {val_at_latency}, SCALE_FACTOR: {scale_factor} NEW MAX: {t.integral_downsampled()[int(latency)]*scale_factor}')
    fname = f'integral_at_latency_with_prediction_{mtg.mouse_id}'
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{fname}.eps', format='eps')


def plot_all_integrals_to_latency_with_predictions():
    mtgs, labels = get_mtgs(LSIE_SNL_KEYS)
    for mtg in mtgs:
        plot_pre_test_trials_with_predicted_values(mtg)


def main():
    import seaborn as sns
    sns.set_style("white")
    plot_all_integrals_to_latency_with_predictions()
    #get_snl_pre_test_and_high_contrast_trials()
    #plot_all_theoretical_escape_thresholds()
    #plot_all_integrals()
    #plot_895773_latency_escapes()
    #plot_all_theoretical_escape_thresholds()
    #plot_snl_signal_escape_latency()
    #get_df_non_escape_relative_to_estimated_threshold()
    #replot_lsie()
    #plot_all_integrals()


if __name__ == '__main__':
    main()
