# mouse_ids_habituated = ['074744', '074746', '895773']
# mouse_ids_non = ['074743', '898990', '898989']
# for mouse_id in mouse_ids_non:
#     plt.close('all')
#     mtg = loom_trial_group.MouseLoomTrialGroup(mouse_id)
#     pre_trials = mtg.pre_test_trials()[0:3]
#     post_trials = mtg.post_test_trials()[0:3]
#
#     fig, axes = plt.subplots(2, 1)
#     for i, (trials, color) in enumerate(zip([pre_trials, post_trials], ['r', 'k'])):
#         ax = axes[i]
#         plt.ylabel('normalised x position', fontsize=10, fontweight='black', color='#333F4B')
#         plt.xlabel('time (s)', fontsize=10, fontweight='black', color='#333F4B')
#         ax.spines['top'].set_color('none')
#         ax.spines['right'].set_color('none')
#         ax.spines['bottom'].set_position(('axes', -0.04))
#         ax.spines['left'].set_position(('axes', 0.015))
#         fig.subplots_adjust(bottom=0.3)
#         fig.subplots_adjust(wspace=0.8)
#         avg_df = np.mean([t.delta_f() for t in trials], axis=0)
#         avg_tracks = np.mean([t.normalised_x_track for t in trials], axis=0)
#         ax.plot(avg_tracks, color=color)
#         ax.plot(avg_df * 10, color='b')
#         ax.set_ylim([-0.5, 1])
#         t.plot_stimulus()
#         for t in trials:
#             ax.plot(t.normalised_x_track, color='grey')
#     fig.savefig('/home/slenzi/{}_example_habituated.eps'.format(mouse_id), format='eps')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def plot_tracks_and_signals(mtgs):
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    all_tracks = []
    all_delta_f = []

    for mtg in mtgs:
        for t in mtg.post_test_trials()[:3]:
            t.plot_track(ax1)
            ax2.plot(t.delta_f(), color='k', linewidth=0.5)
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(t.delta_f())
    mtgs[0].post_test_trials()[0].plot_stimulus()


    avg_track = np.nanmean(all_tracks, axis=0)
    ax1.plot(avg_track, linewidth=3)
    avg_df = np.nanmean(all_delta_f, axis=0)
    ax2.plot(avg_df, linewidth=3)
    ax2.set_ylim([-0.01, 0.2])
    plt.sca(ax2)
    t.plot_stimulus()

from scipy import stats


def plot_max_integral_bars(mtgs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    group_avg_pre = []
    group_avg_post = []

    for c, mtg in zip(colors, mtgs):
        max_event = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
        max_integrals_pre = [np.nanmax(t.integral) / max_event for t in mtg.pre_test_trials()[:3]]
        max_integrals_post = [np.nanmax(t.integral) / max_event for t in mtg.post_test_trials()[:3]]
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
    print(stats.ttest_ind(group_avg_pre, group_avg_post))


def plot_integral_at_latency_bars(mtgs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    group_avg_pre = []
    group_avg_post = []

    for c, mtg in zip(colors, mtgs):
        pre_test_latency = np.nanmean([t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]])
        normalising_factor = max([np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])])

        max_integrals_pre = [np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor for t in mtg.pre_test_trials()[:3]]
        max_integrals_post = [np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor for t in mtg.post_test_trials()[:3]]
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
    plt.errorbar(1, np.mean(group_avg_pre), scipy.stats.sem(group_avg_pre, axis=0), color='k')
    plt.errorbar(2, np.mean(group_avg_post), scipy.stats.sem(group_avg_post,axis=0), color='k')
    print(stats.ttest_ind(group_avg_pre, group_avg_post))


def get_max_integral_habituations(mtgs):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    all_integrals = []

    for c, mtg in zip(colors, mtgs):
        max_event = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
        max_integrals = [np.nanmax(t.integral) / max_event for t in mtg.habituation_trials()[:24]]
        # all_max_integrals.append(max_integrals)

        plt.scatter(np.arange(max_integrals), max_integrals, color='w', edgecolor='k')

        # plt.scatter(1, np.mean(max_integrals_pre), color=c, zorder=1000, s=60)
        # plt.scatter(2, np.mean(max_integrals_post), color=c, zorder=1000, s=60)
        # plt.plot([1, 2], [np.mean(max_integrals_pre), np.mean(max_integrals_post)], color='k')
        all_integrals.append(np.mean(max_integrals))
    return all_integrals


def get_normalised_habituation_heatmap(mtg):
    max_event = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
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
        max_event = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
        max_integrals = [np.nanmax(t.integral) / max_event for t in mtg.habituation_trials()[:24]]
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


def plot_habituation_curve_with_sem(mtgs):
    all_groups = get_habituation_protocol_responses(mtgs)
    binned = bin_lsie_contrasts(all_groups)

    group_avg_curve = np.nanmean(binned, axis=0)
    std_escape_curve = scipy.stats.sem(binned, axis=0)

    plt.plot(np.arange(len(group_avg_curve)), group_avg_curve, color='k', linewidth=3)
    plt.scatter(np.arange(len(group_avg_curve)), group_avg_curve, color='w', edgecolor='k', zorder=10)

    for i, (error, value) in enumerate(zip(std_escape_curve, group_avg_curve)):
        plt.errorbar(i, value, error, color='k')

# plt.figure()
# ax = plt.subplot(121)
# plot_max_integral_bars(hab_mtgs)
# ax2 = plt.subplot(122, sharey=ax)
# plot_max_integral_bars(non_hab_mtgs)
# plt.ylim([0, 1.1])
#
# ax.set_xticklabels([None, 'pre_test', '', 'post_test'], rotation=90)
# ax2.set_xticklabels([None, 'pre_test', '', 'post_test'], rotation=90)

# from looming_spots.analysis import contrast_experiments
# control_mids = ['898992', '916063',  '907822', '921000', 'CA439_5', '276585D', '276585E', 'CA452_1']
# contrast_experiments.plot_block_1_escape_curves_with_avg(control_mids)
# nmda_mids = ['CA439_1', 'CA439_4', '276585A', '276585B']
# contrast_experiments.plot_block_1_escape_curves_with_avg(nmda_mids)
