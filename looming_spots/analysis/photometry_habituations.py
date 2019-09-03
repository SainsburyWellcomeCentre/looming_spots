import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import seaborn as sns

from looming_spots.db import loom_trial_group


def plot_tracks_and_signals(mtgs, test_type='pre_test'):

    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    all_tracks = []
    all_delta_f = []

    for mtg in mtgs:
        trials = mtg.pre_test_trials()[:3] if test_type == 'pre_test' else mtg.post_test_trials()[:3]

        for t in trials:
            t.plot_track(ax1)
            ax2.plot(t.delta_f(), color='k', linewidth=0.5)
            ax2.plot(t.integral_downsampled(), color='grey')
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(t.delta_f())
    trials[0].plot_stimulus()

    avg_track = np.nanmean(all_tracks, axis=0)
    ax1.plot(avg_track, linewidth=3)
    avg_df = np.nanmean(all_delta_f, axis=0)
    ax2.plot(avg_df, linewidth=3)
    ax2.set_ylim([-0.01, 0.2])
    plt.sca(ax2)
    t.plot_stimulus()


def plot_max_integral_bars(mtgs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    group_avg_pre = []
    group_avg_post = []

    for c, mtg in zip(colors, mtgs):
        max_event = max([np.nanmax(t.integral_downsampled) for t in mtg.loom_trials()])
        max_integrals_pre = [np.nanmax(t.integral_downsampled) / max_event for t in mtg.pre_test_trials()[:3]]
        max_integrals_post = [np.nanmax(t.integral_downsampled) / max_event for t in mtg.post_test_trials()[:3]]
        # all_max_integrals.append(max_integrals)

        plt.scatter(np.ones_like(max_integrals_pre), max_integrals_pre, color='w', edgecolor='k')
        plt.scatter(np.ones_like(max_integrals_post) * 2, max_integrals_post, color='w', edgecolor='k')

        plt.scatter(1, np.mean(max_integrals_pre), color=c, zorder=1000, s=60)
        plt.scatter(2, np.mean(max_integrals_post), color=c, zorder=1000, s=60)
        plt.plot([1, 2], [np.mean(max_integrals_pre), np.mean(max_integrals_post)], color='k')
        group_avg_pre.append(np.mean(max_integrals_pre))
        group_avg_post.append(np.mean(max_integrals_post))
    plt.bar(1, np.mean(group_avg_pre), color='k', alpha=0.4)
    plt.bar(2, np.mean(group_avg_post), color='k', alpha=0.4)
    print(scipy.stats.ttest_ind(group_avg_pre, group_avg_post))


def plot_integral_at_latency_bars(mtgs, bar_colors=('k', 'k')):
    group_avg_pre = []
    group_avg_post = []

    for mtg in mtgs:
        pre_test_latency = np.nanmean([t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]])
        normalising_factor = max([np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])])

        max_integrals_pre = [np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor for t in mtg.pre_test_trials()[:3]]
        max_integrals_post = [np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor for t in mtg.post_test_trials()[:3]]
        # all_max_integrals.append(max_integrals)

        plt.scatter(np.ones_like(max_integrals_pre), max_integrals_pre, color='w', edgecolor='k')
        plt.scatter(np.ones_like(max_integrals_post) * 2, max_integrals_post, color='w', edgecolor='k')

        plt.scatter(1, np.mean(max_integrals_pre), color='k', zorder=1000, s=60)
        plt.scatter(2, np.mean(max_integrals_post), color='k', zorder=1000, s=60)
        plt.plot([1, 2], [np.mean(max_integrals_pre), np.mean(max_integrals_post)], color='k')
        plt.xticks([1, 2], ['pre-LSIE protocol', 'post-LSIE protocol'], rotation=90)
        group_avg_pre.append(np.mean(max_integrals_pre))
        group_avg_post.append(np.mean(max_integrals_post))
    plt.bar(1, np.mean(group_avg_pre), color=bar_colors[0], zorder=0)
    plt.bar(2, np.mean(group_avg_post), color=bar_colors[1], zorder=0)
    plt.errorbar(1, np.mean(group_avg_pre), scipy.stats.sem(group_avg_pre, axis=0), color='Grey')
    plt.errorbar(2, np.mean(group_avg_post), scipy.stats.sem(group_avg_post, axis=0), color='Grey')
    print(scipy.stats.ttest_ind(group_avg_pre, group_avg_post))


def get_max_integral_habituations(mtgs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    all_integrals = []

    for c, mtg in zip(colors, mtgs):
        max_event = max([np.nanmax(t.integral_downsampled()) for t in mtg.loom_trials()])
        max_integrals = [np.nanmax(t.integral_downsampled()) / max_event for t in mtg.habituation_trials()[:24]]
        # all_max_integrals.append(max_integrals)

        plt.scatter(np.arange(max_integrals), max_integrals, color='w', edgecolor='k')

        # plt.scatter(1, np.mean(max_integrals_pre), color=c, zorder=1000, s=60)
        # plt.scatter(2, np.mean(max_integrals_post), color=c, zorder=1000, s=60)
        # plt.plot([1, 2], [np.mean(max_integrals_pre), np.mean(max_integrals_post)], color='k')
        all_integrals.append(np.mean(max_integrals))
    return all_integrals


def get_normalised_habituation_heatmap(mtg):
    max_event = max([np.nanmax(t.integral_downsampled()) for t in mtg.loom_trials()])
    hm = np.full((len(mtg.loom_trials()[0].delta_f()), len(mtg.habituation_trials())), np.nan)
    for i, t in enumerate(mtg.habituation_trials()):
        hm[:, i] = t.delta_f()/max_event
    return hm


def plot_habituation_avg_heatmap(mtgs):
    hms = []
    for mtg in mtgs:
        hm = get_normalised_habituation_heatmap(mtg)
        hms.append(hm)
    return hms


def plot_binned_max_integral_habituation_pooled(mtgs, col='k'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    all_integrals = []
    all_integrals_binned = []

    for c, mtg in zip(colors, mtgs):
        max_event = max([np.nanmax(t.integral_downsampled()) for t in mtg.loom_trials()])
        max_integrals = [np.nanmax(t.integral_downsampled()) / max_event for t in mtg.habituation_trials()[:24]]
        all_integrals.append(max_integrals)

        binned_contrast_curve = np.mean(max_integrals.reshape(-1, 3), axis=1)
        all_integrals_binned.append(binned_contrast_curve)

    average_curve = np.array(np.mean(all_integrals, axis=0))

    plt.plot(average_curve, linewidth=3, color=col, zorder=20)
    plt.scatter(np.arange(len(average_curve)), average_curve, color='w', edgecolor=col,
                zorder=100)

    plt.ylim([0, 1])
    return all_integrals, all_integrals_binned


def get_habituation_protocol_responses(mtgs):
    group_integral_escape_metrics = []
    for mtg in mtgs:
        pre_test_latency = np.nanmean([t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]])
        print(pre_test_latency)
        normalising_factor = max([np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])])
        max_integrals = [np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor for t in mtg.habituation_trials()[:24]]
        group_integral_escape_metrics.append(max_integrals)
    return group_integral_escape_metrics


def bin_lsie_contrasts(all_integrals):
    all_binned = []
    for item in all_integrals:
        all_binned.append(np.mean(np.array(item).reshape(-1, 3), axis=1))
    return all_binned


def plot_habituation_curve_with_sem(mtgs, color='k', linewidth=3, plot_dots=True):
    all_groups = get_habituation_protocol_responses(mtgs)
    binned = bin_lsie_contrasts(all_groups)

    group_avg_curve = np.nanmean(binned, axis=0)
    std_escape_curve = scipy.stats.sem(binned, axis=0)

    plt.plot(np.arange(len(group_avg_curve)), group_avg_curve, color=color, linewidth=linewidth)
    if plot_dots:
        plt.scatter(np.arange(len(group_avg_curve)), group_avg_curve, color='w', edgecolor=color, zorder=10)

    for i, (error, value) in enumerate(zip(std_escape_curve, group_avg_curve)):
        plt.errorbar(i, value, error, color=color)


def get_signal_metric_dataframe_variable_contrasts(mtgs, metric):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        vals = []
        signals = []
        event_metric_dict = {}
        #trials = mtg.pre_test_trials()[:3]
        trials = mtg.all_trials[:18]
        norm_factor = max([max(t.integral_downsampled()[200:214]) for t in mtg.all_trials])

        for t in trials:
            val = t.metric_functions[metric]()
            signal = max(t.integral_downsampled()[200:214])/norm_factor
            vals.append(val)
            signals.append(signal)

        mids = [mtg.mouse_id]*len(trials)
        metrics = [metric]*len(trials)
        event_metric_dict.setdefault('mouse id', mids)
        event_metric_dict.setdefault('ca signal', signals)
        event_metric_dict.setdefault(metric, vals)
        event_metric_dict.setdefault('test type', ['variable contrast']*len(trials))
        #event_metric_dict.setdefault('metric', metrics)
        event_metric_dict.setdefault('contrast', [t.contrast for t in trials])
        event_metric_dict.setdefault('escape', [t.is_flee() for t in trials])

        metric_df = pd.DataFrame.from_dict(event_metric_dict)
        all_df = all_df.append(metric_df, ignore_index=True)

    return all_df


def get_signal_metric_dataframe(mtgs, metric):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        norm_factor = max([t.integral_escape_metric() for t in mtg.all_trials]) #t.integral_downsampled()[200:214]

        for trials, test_type in zip([mtg.pre_test_trials()[:3], mtg.post_test_trials()[:3]], ['pre test', 'post test']):
            event_metric_dict = {}
            vals = []
            signals = []
            for t in trials:
                val = t.metric_functions[metric]()
                signal = t.integral_escape_metric()/norm_factor  #t.integral_downsampled()[200:214]
                vals.append(val)
                signals.append(signal)

            mids = [mtg.mouse_id]*len(trials)
            event_metric_dict.setdefault('mouse id', mids)
            event_metric_dict.setdefault('ca signal', signals)
            event_metric_dict.setdefault(metric, vals)
            event_metric_dict.setdefault('test type', [test_type]*len(trials))

            event_metric_dict.setdefault('contrast', [t.contrast for t in trials])
            event_metric_dict.setdefault('escape', [t.is_flee() for t in trials])

            metric_df = pd.DataFrame.from_dict(event_metric_dict)
            all_df = all_df.append(metric_df, ignore_index=True)

    return all_df


def plot_signal_against_escape_metrics(mouse_ids):
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    dfs = []
    for metric in ['latency to escape', 'speed', 'acceleration', 'time in safety zone']:
        df = get_signal_metric_dataframe_variable_contrasts(mtgs, metric)
        dfs.append(df)

    for i, df in enumerate(dfs):
        plt.subplot(2, 2, i + 1)
        sns.regplot('metric value', 'ca signal', df)


def plot_signal_against_metric(mtgs, metric):
    fig = plt.figure()
    for mtg in mtgs:
        plt.subplot(121)
        for t in mtg.pre_test_trials()[:3]:
            val = t.metric_functions[metric]()
            signal = max(t.integral_downsampled()[200:350])
            color = 'r' if t.is_flee() else 'k'
            plt.scatter(val, signal, color=color)

        plt.subplot(122)
        for t in mtg.post_test_trials()[:3]:
            val = t.metric_functions[metric]()
            signal = max(t.integral_downsampled()[200:350])
            color = 'r' if t.is_flee() else 'k'
            plt.scatter(val, signal, color=color)
    return fig


def get_time_series_df(mtgs):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        start = 0
        norm_factor = max([max(t.integral_downsampled()[200:400]) for t in mtg.all_trials]) #t.integral_downsampled()[200:214]
        for trials, test_type in zip([mtg.pre_test_trials()[:3], mtg.post_test_trials()[:3]], ['pre test', 'post test']):
            for i, t in enumerate(trials):
                event_metric_dict = {}
                data = t.integral_downsampled()[200:400] / norm_factor
                timepoints = np.arange(len(data))
                mids = [mtg.mouse_id]*len(data)

                event_metric_dict.setdefault('mouse id', mids)
                event_metric_dict.setdefault('ca signal', data)
                event_metric_dict.setdefault('timepoint', timepoints)
                event_metric_dict.setdefault('test type', [test_type]*len(data))
                event_metric_dict.setdefault('escape', [t.is_flee()]*len(data))
                event_metric_dict.setdefault('trial_number', [i+start]*len(data))
                print(i, start, 'start')

                metric_df = pd.DataFrame.from_dict(event_metric_dict)
                all_df = all_df.append(metric_df, ignore_index=True)
            start += len(trials)
    return all_df


def habituation_df(mtg_groups, mtg_group_labels):
    all_df = pd.DataFrame()
    for mtgs, label in zip(mtg_groups, mtg_group_labels):
        for mtg in mtgs:
            start = 0
            norm_factor = max([max(t.integral_downsampled()[200:214]) for t in mtg.all_trials])
            for trials, test_type in zip([mtg.pre_test_trials()[:3], mtg.habituation_trials()[:23], mtg.post_test_trials()[:3]], ['pre test', 'habituation', 'post test']):
                event_metric_dict = {}
                contrasts = []
                signals = []
                for t in trials:
                    signal = max(t.integral_downsampled()[200:214])/norm_factor
                    contrasts.append(t.contrast)
                    signals.append(signal)

                mids = [mtg.mouse_id]*len(trials)
                event_metric_dict.setdefault('mouse id', mids)
                event_metric_dict.setdefault('group label', [label]*len(trials))
                event_metric_dict.setdefault('ca signal', signals)
                event_metric_dict.setdefault('test type', [test_type]*len(trials))
                event_metric_dict.setdefault('contrast', contrasts)
                event_metric_dict.setdefault('trial number', np.arange(start, len(trials)+start))
                start += len(trials)

                metric_df = pd.DataFrame.from_dict(event_metric_dict)
                all_df = all_df.append(metric_df, ignore_index=True)
    return all_df


def get_behaviour_metric_dataframe(mtgs, metric):
    all_df = pd.DataFrame()
    for mtg in mtgs:
        trials = mtg.all_trials[:18]
        event_metric_dict = {}
        vals = []
        for t in trials:
            val = t.metric_functions[metric]()
            vals.append(val)

        mids = [mtg.mouse_id]*len(trials)
        event_metric_dict.setdefault('mouse id', mids)
        event_metric_dict.setdefault(metric, vals)
        event_metric_dict.setdefault('test type', ['variable contrast'] * len(trials))
        # event_metric_dict.setdefault('metric', metrics)
        event_metric_dict.setdefault('contrast', [t.contrast for t in trials])
        event_metric_dict.setdefault('escape', [t.is_flee() for t in trials])
        metric_df = pd.DataFrame.from_dict(event_metric_dict)
        all_df = all_df.append(metric_df, ignore_index=True)
    return all_df
