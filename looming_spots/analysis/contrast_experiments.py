import numpy as np
import matplotlib.pyplot as plt

from looming_spots.analysis import plotting
from looming_spots.db import loom_trial_group, experimental_log


def get_trials_of_contrast(trials, c):
    return [t for t in trials if t.contrast == c]


def get_binned_escape_probabilities_at_contrast(trials, contrasts, c, group_by_idx=2):
    grouped_by = get_grouped_trials_for_contrast(c, contrasts, group_by_idx, trials)

    all_escape_probabilities = []
    for group in grouped_by:
        classified_as_flee = []
        for t in group:
            classified_as_flee.append(t.is_flee())
        all_escape_probabilities.append(np.mean(classified_as_flee))
    return all_escape_probabilities


def get_binned_escapes_at_contrast(trials, contrasts, c, group_by_idx=2):
    grouped_by = get_grouped_trials_for_contrast(c, contrasts, group_by_idx, trials)

    all_escapes = []
    for group in grouped_by:
        classified_as_flee = []
        for t in group:
            classified_as_flee.append(t.is_flee())
        all_escapes.append(classified_as_flee)
    return all_escapes


def get_block_escape_probability_contrast_curve(trials, contrasts, group_by_idx=2, block_id=0):
    all_probabilities = []
    contrast_order = np.unique(contrasts)
    for c in contrast_order:
        grouped_by = list(get_grouped_trials_for_contrast(c, contrasts, group_by_idx, trials))
        group = grouped_by[block_id]

        classified_as_flee = []

        for t in group:
            classified_as_flee.append(t.is_flee())

        all_probabilities.append(np.mean(classified_as_flee))
    return all_probabilities, contrast_order


def get_grouped_trials_for_contrast(c, contrasts, group_by_idx, trials):
    for t, contrast in zip(trials, contrasts):
        t.set_contrast(contrast)
    trials = sorted(get_trials_of_contrast(trials, c))
    grouped_by = chunks(trials, group_by_idx)
    return grouped_by


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def plot_block_bars(all_groups_escape_probabilities):
    fig = plt.figure()
    for i, item in enumerate(np.array(all_groups_escape_probabilities).T):
        plt.bar(np.arange(i + i*len(all_groups_escape_probabilities), i*len(item), 1), item, edgecolor='k')
        plt.ylim([0, 1.1])
    return fig

#
# contrasts = io.loadmat('/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/CA452_1/20190613_15_07_24/contrasts.mat')['contrasts']
# contrast_experiments.get_binned_escape_probabilities_at_contrast(mtg.all_trials, )


# import scipy
# from scipy import io
# from looming_spots.analysis import contrast_experiments
# import numpy as np
# from looming_spots.db import loom_trial_group
#
# mtg = loom_trial_group.MouseLoomTrialGroup('CA452_1')
# contrasts = io.loadmat(
#     '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/CA452_1/20190613_15_07_24/contrasts.mat')[
#     'contrasts'][0]
#
# all_groups_escape_probabilities = []
# for c in np.unique(contrasts):
#     group_by_idx = 6 if c == 0 else 2
#     escape_probabilities = contrast_experiments.get_binned_escape_probabilities_at_contrast(mtg.all_trials[:-1],
#                                                                                             contrasts, c, group_by_idx)
#     print(escape_probabilities)
#     all_groups_escape_probabilities.append(escape_probabilities)

# fig, axes = plt.subplots(7, 1)
#
# for i, (group, label) in enumerate(zip(all_groups_escape_probabilities, [0, 0.1007, 0.1107, 0.1207, 0.1307, 0.1407, 0.1507])):
#     axes[i].plot(np.arange(len(group)),group)
#     axes[i].set_ylabel(label)
#     axes[i].set_ylim([-0.01,1.1])
#     axes[i].set_xlabel('block number')
#
# [ax.spines['right'].set_visible(False) for ax in axes]
# [ax.spines['top'].set_visible(False) for ax in axes]

def get_pooled_escape_probilities_all_contrasts_block(mouse_ids, contrast_set, block_id=0):
    escape_curve = []
    for c in contrast_set:
        trials = get_trials_of_contrast_mouse_group(mouse_ids, c, start=block_id*18, end=(block_id+1)*18)
        avg_contrast_probability = np.nanmean([t.is_flee() for t in trials])
        escape_curve.append(avg_contrast_probability)

    return escape_curve


def get_trials_of_contrast_mouse_group(mids, c, start=0, end=18):
    from looming_spots.db import loom_trial_group
    all_trials = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        trials = [t for t in mtg.all_trials[start:end] if t.contrast == c]
        all_trials.extend(trials)
    return all_trials


def get_pooled_escape_probilities_all_contrasts(mouse_ids, contrast_set, n_blocks=2):
    block_1_escape_curve = []
    block_2_escape_curve = []
    for c in contrast_set:
        block_1_summary = []
        block_2_summary = []
        for mid in mouse_ids:
            mtg = loom_trial_group.MouseLoomTrialGroup(mid)
            groupbyidx = 6 if c == 0 else 2
            block_1, block_2 = get_binned_escapes_at_contrast(mtg.all_trials[:18*n_blocks], mtg.contrasts(), c, group_by_idx=groupbyidx)
            block_1_summary.extend(block_1)
            block_2_summary.extend(block_2)

        percent_at_contrast_block1 = np.count_nonzero(block_1_summary) / len(block_1_summary)
        percent_at_contrast_block2 = np.count_nonzero(block_2_summary) / len(block_2_summary)
        block_1_escape_curve.append(percent_at_contrast_block1)
        block_2_escape_curve.append(percent_at_contrast_block2)
    return block_1_escape_curve, block_2_escape_curve


def plot_all_blocks_scatterbar(all_groups_escape_probabilities, contrasts, fig=None, axes=None):
    import pandas as pd

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if fig is None:
        fig, axes = plt.subplots(len(all_groups_escape_probabilities[0]), 1, figsize=(4, 5))
    contrast_set = np.unique(contrasts)

    for i, item in enumerate(np.array(all_groups_escape_probabilities).T):
        ax = axes[i]
        plt.sca(ax)
        percentages = pd.Series(item, index=contrast_set)
        df = pd.DataFrame({'percentage': percentages})

        ax.plot(contrast_set, df['percentage'], "o", markersize=5, alpha=0.2, color=colors[i])

        plt.ylabel('escape probability', fontsize=10, fontweight='black', color='#333F4B')
        plt.xlabel('contrast', fontsize=10, fontweight='black', color='#333F4B')
        # ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)

        # set the spines position
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position(('axes', -0.04))
        ax.spines['left'].set_position(('axes', 0.015))
        ax.set_ylim([0, 1.1])

        ax.set_xticks(contrast_set)
        ax.set_xticklabels(contrast_set, rotation=90)
        fig.subplots_adjust(bottom=0.3)
        fig.subplots_adjust(hspace=0.9)
        fig.subplots_adjust(wspace=0.8)

    return fig


def plot_block(block_escape_probabilities, contrasts, colors=None):

    import pandas as pd

    prop_cycle = plt.rcParams['axes.prop_cycle']
    if colors is None:
        colors = prop_cycle.by_key()['color']
    fig, axes = plt.subplots(2, 1, figsize=(4, 5))
    contrast_set = np.unique(contrasts)


    ax = axes[0]
    plt.sca(ax)
    percentages = pd.Series(block_escape_probabilities, index=contrast_set)
    df = pd.DataFrame({'percentage': percentages})

    plt.vlines(x=contrast_set, ymin=0, ymax=df['percentage'], alpha=0.2, linewidth=5, color=colors[0])
    ax.plot(contrast_set, df['percentage'], "o", markersize=5, alpha=0.2, color=colors[0])

    plt.ylabel('escape probability', fontsize=10, fontweight='black', color='#333F4B')
    plt.xlabel('contrast', fontsize=10, fontweight='black', color='#333F4B')
    #ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    # set the spines position
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))
    ax.set_ylim([0, 1.1])

    ax.set_xticks(contrast_set)
    ax.set_xticklabels(contrast_set, rotation=90)
    fig.subplots_adjust(bottom=0.3)
    fig.subplots_adjust(hspace=0.9)
    fig.subplots_adjust(wspace=0.8)

    return fig, axes


def plot_all_blocks(all_groups_escape_probabilities, contrasts, colors=None):

    import pandas as pd

    prop_cycle = plt.rcParams['axes.prop_cycle']
    if colors is None:
        colors = prop_cycle.by_key()['color']
    fig, axes = plt.subplots(len(all_groups_escape_probabilities[0]), 1, figsize=(4, 5))
    contrast_set = np.unique(contrasts)

    for i, item in enumerate(np.array(all_groups_escape_probabilities).T):
        ax = axes[i]
        plt.sca(ax)
        percentages = pd.Series(item, index=contrast_set)
        df = pd.DataFrame({'percentage': percentages})

        plt.vlines(x=contrast_set, ymin=0, ymax=df['percentage'], alpha=0.2, linewidth=5, color=colors[i])
        ax.plot(contrast_set, df['percentage'], "o", markersize=5, alpha=0.2, color=colors[i])


        plt.ylabel('escape probability', fontsize=10, fontweight='black', color='#333F4B')
        plt.xlabel('contrast', fontsize=10, fontweight='black', color='#333F4B')
        #ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)

        # set the spines position
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position(('axes', -0.04))
        ax.spines['left'].set_position(('axes', 0.015))
        ax.set_ylim([0, 1.1])

        ax.set_xticks(contrast_set)
        ax.set_xticklabels(contrast_set, rotation=90)
        fig.subplots_adjust(bottom=0.3)
        fig.subplots_adjust(hspace=0.9)
        fig.subplots_adjust(wspace=0.8)

    return fig, axes


def plot_photometry_by_contrast(mtg):
    for contrast in np.unique(mtg.contrasts()):  #TODO: extract to mtg?
        tracks = []
        avg_df = []
        fig, axes = plt.subplots(2, 1)

        for t in mtg.all_trials:
            if t.contrast == contrast:
                tracks.append(t.normalised_x_track)
                avg_df.append(t.delta_f())
                axes[0].plot(t.normalised_x_track, color='grey', linewidth=0.5)
                axes[1].plot(t.delta_f(), color='grey', linewidth=0.5)

        axes[0].plot(np.mean(tracks, axis=0), color='r')
        axes[1].plot(np.mean(avg_df, axis=0), color='k')
        t.plot_stimulus()


def plot_delta_f_max_integral_against_contrast(mtgs):
    plt.figure()

    for mtg in mtgs:
        trials = mtg.all_trials
        normalising_factor = max([np.nanmax(t.integral) for t in trials])
        for t in trials:
            ca_response = np.nanmax(t.integral) / normalising_factor
            color = 'r' if t.is_flee() else 'k'
            plt.plot(t.contrast, ca_response, 'o', color=color)


def plot_delta_f_at_latency_against_contrast(mtgs):
    plt.figure()

    for mtg in mtgs:
        test_contrast_trials = [t for t in mtg.loom_trials() if t.contrast == 0]
        pre_test_latency = np.nanmean([t.estimate_latency(False) for t in test_contrast_trials])
        normalising_factor = max([np.nanmax([t.integral_escape_metric(int(pre_test_latency)) for t in mtg.loom_trials()])])

        for i, t in zip(np.arange(18), mtg.all_trials):
            ca_response = np.nanmax(t.integral_escape_metric(int(pre_test_latency))) / normalising_factor
            color = 'r' if t.is_flee() else 'k'
            plt.plot(t.contrast, ca_response, 'o', color=color)

# import scipy
# from scipy import io
# from looming_spots.analysis import contrast_experiments
# import numpy as np
# from looming_spots.db import loom_trial_group
#
# nmda_lesion_group = []
#
# for mid in ['CA439_1', 'CA439_4', '276585A']:
#     mtg = loom_trial_group.MouseLoomTrialGroup(mid)
#     contrasts = mtg.contrasts()
#     n_blocks = 2
#     all_groups_escape_probabilities = []
#     for c in np.unique(contrasts):
#         group_by_idx = 6 if c == 0 else 2
#         escape_probabilities = contrast_experiments.get_binned_escape_probabilities_at_contrast(
#             mtg.all_trials[:18 * n_blocks], contrasts, c, group_by_idx)
#         print(escape_probabilities)
#         all_groups_escape_probabilities.append(escape_probabilities)
#     nmda_lesion_group.append(all_groups_escape_probabilities)
#     fig = contrast_experiments.plot_all_blocks(all_groups_escape_probabilities, np.unique(contrasts))
#     fig.savefig('/home/slenzi/first_two_blocks_NMDA_mice_variable_contrasts_{}.eps'.format(mid), format='eps')

#
# from looming_spots.db import loom_trial_group
# from looming_spots.analysis import contrast_experiments
# import numpy as np
# import matplotlib.pyplot as plt
#
# mouse_group = ['CA439_1', 'CA439_4', '276585A', '276585B']
# ctrl_mouse_group = ['CA439_5', 'CA452_1']
# plt.figure(figsize=(10, 3))
# ax=plt.subplot(111)
# all_tracks = []
#
# for mid in ctrl_mouse_group:
#
#     mtg = loom_trial_group.MouseLoomTrialGroup(mid)
#     contrasts = mtg.contrasts()
#     for t in mtg.all_trials[:18]:
#         if t.contrast == 0.0:
#             all_tracks.append(t.normalised_x_track)
#             plt.plot(t.normalised_x_track, color='grey', linewidth = 0.5)
# plt.plot(np.mean(all_tracks,axis=0), color='r')
# t.plot_stimulus()

import scipy.stats


def plot_block_1_escape_curves_with_avg(mids, color='k'):
    mtg = loom_trial_group.MouseLoomTrialGroup(mids[0])
    contrast_set = np.unique(mtg.contrasts())
    all_escape_curves = []
    for mid in mids:
        escape_curve = get_pooled_escape_probilities_all_contrasts_block([mid], contrast_set)
        all_escape_curves.append(escape_curve)
        #plt.plot(contrast_set, escape_curve, linewidth=0.5, color=color, alpha=0.3)

    avg_escape_curve = np.nanmean(all_escape_curves, axis=0)
    sem_escape_curve = scipy.stats.sem(all_escape_curves, axis=0)

    plt.plot(np.unique(mtg.contrasts()), avg_escape_curve, color=color, linewidth=3)
    plt.scatter(np.unique(mtg.contrasts()), avg_escape_curve, color='w', edgecolor=color, zorder=10)

    for i, (contrast, error, value) in enumerate(zip(contrast_set, sem_escape_curve, avg_escape_curve)):
        plt.errorbar(contrast, value, error, color=color)

    plt.ylim([-0.01, 1.1])
    plt.xlim([0.165, -0.01])
    plt.xlabel('contrast', fontsize=10, fontweight='black', color='#333F4B')
    plt.ylabel('escape %', fontsize=10, fontweight='black', color='#333F4B')


def plot_all_pre_tests(mids):
    mids = ['074744', '074746', '074743', '898989', '898990', '895773',  '898992', '916063', '921000', '907822']

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    all_tracks = []
    all_delta_f = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        norm_factor = max([np.nanmax(t.delta_f()) for t in mtg.loom_trials()])
        if len(mtg.contrasts()) == 0:
            pre_trials = mtg.loom_trials()[:3]
        else:
            pre_trials = [t for t in mtg.all_trials if t.contrast == 0][:3]

        for t in pre_trials:
            norm_df = t.delta_f()/norm_factor

            t.plot_track(ax1)
            ax2.plot(norm_df, color='grey', linewidth=0.5)
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(norm_df)
    ax1.plot(np.mean(all_tracks, axis=0), color='k', linewidth=3)
    ax2.plot(np.mean(all_delta_f, axis=0), color='k', linewidth=3)


def plot_all_post_tests(mids):
    fig=plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    all_tracks = []
    all_delta_f = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        norm_factor = max([np.nanmax(t.delta_f()) for t in mtg.loom_trials()])
        for t in mtg.post_test_trials()[:3]:
            norm_df = t.delta_f()/norm_factor

            t.plot_track(ax1)
            ax2.plot(norm_df, color='grey', linewidth=0.5)
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(norm_df)

    ax1.plot(np.mean(all_tracks, axis=0), color='k', linewidth=3)
    ax2.plot(np.mean(all_delta_f, axis=0), color='k', linewidth=3)
    ax2.set_ylim([-0.1, 1.1])
    return fig


def plot_pre_test_max_integral():
    mids = ['074744', '074746', '074743', '898989', '898990', '895773', '898992', '916063', '921000', '907822']
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    all_max_integrals = []
    for mtg in mtgs:
        norm_factor = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
        if len(mtg.contrasts()) == 0:
            pre_trials = mtg.loom_trials()[:3]
        else:
            pre_trials = [t for t in mtg.all_trials if t.contrast == 0][:3]
        max_integrals = [np.nanmax(t.integral) for t in pre_trials] / norm_factor
        all_max_integrals.append(max_integrals)

    plt.bar(1, np.mean(all_max_integrals), alpha=0.2)
    plt.scatter(np.ones_like(np.array(all_max_integrals).flatten()), np.array(all_max_integrals).flatten(), color='w',
                edgecolor='k')


def plot_pre_post_photometry_LSIE(mtg):

    fig1 = plt.figure()
    ax1 = plt.subplot(211)
    avg_df = []

    for t in mtg.pre_test_trials()[:3]:
        t.plot_track()
    mtg.pre_test_trials()[0].plot_stimulus()
    ax2 = plt.subplot(212)
    for t in mtg.pre_test_trials()[:3]:
        plt.plot(t.delta_f())
        avg_df.append(t.delta_f())
    mtg.pre_test_trials()[0].plot_stimulus()
    plt.plot(np.mean(avg_df, axis=0), linewidth=4)
    plt.ylim([-0.01, 0.15])

    avg_df = []
    fig2 = plt.figure()
    ax3 = plt.subplot(211)
    for t in mtg.post_test_trials()[:3]:
        t.plot_track()
    mtg.pre_test_trials()[0].plot_stimulus()

    ax4 = plt.subplot(212)
    for t in mtg.post_test_trials()[:3]:
        avg_df.append(t.delta_f())
        plt.plot(t.delta_f())
    plt.plot(np.mean(avg_df, axis=0), linewidth=4)

    plt.ylim([-0.01, 0.15])
    mtg.pre_test_trials()[0].plot_stimulus()


    fig1.savefig('/home/slenzi/pre_post_LSIE_photometry/{}_pre_test.eps'.format(mtg.mouse_id), format='eps')
    fig2.savefig('/home/slenzi/pre_post_LSIE_photometry/{}_post_test.eps'.format(mtg.mouse_id), format='eps')


def plot_pre_post_photometry_trials_LSIE(mtg):
    for t in mtg.pre_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig('/home/slenzi/pre_post_LSIE_photometry/loom_{}_{}_pre_test.eps'.format(t.loom_number, mtg.mouse_id), format='eps')
        plt.close('all')
    for t in mtg.post_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig('/home/slenzi/pre_post_LSIE_photometry/loom_{}_{}_post_test.eps'.format(t.loom_number, mtg.mouse_id), format='eps')
        plt.close('all')


def get_all_trials(experimental_group_label):
    mids = experimental_log.get_mouse_ids_in_experiment(experimental_group_label)
    all_trials = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        trials = mtg.pre_test_trials()[:3]
        all_trials.extend(trials)
    return all_trials


def get_cossel_curve(experimental_group_label, subtract_val=None):
    all_trials = get_all_trials(experimental_group_label)
    contrasts = [t.contrast for t in all_trials]
    all_contrasts = np.unique(contrasts)
    avg_curve = []
    sem_curve = []

    for c in all_contrasts:
        trials_of_contrast = get_trials_of_contrast(all_trials, c)
        avg_curve.append(np.mean([t.is_flee() for t in trials_of_contrast]))
        sem_curve.append(scipy.stats.sem([t.is_flee() for t in trials_of_contrast]))

    if subtract_val is not None:
        all_contrasts = subtract_val - np.array(all_contrasts)

    return avg_curve, sem_curve, all_contrasts


def plot_cossel_curves_pooled_trials():
    plt.figure()
    curve, sems, contrasts = get_cossel_curve('spot_contrast_cossel_curve', 0.1607)
    plt.errorbar(contrasts, curve, sems)
    plt.plot(contrasts, curve, 'k', linewidth=4)

    curve, sems, contrasts = get_cossel_curve('background_contrast_cossel_curve')
    plt.errorbar(contrasts, curve, sems)
    plt.plot(contrasts, curve, 'k', linewidth=4)

    plt.xlabel('contrast a.u.', fontsize=10, fontweight='black', color='#333F4B')
    plt.ylabel('escape probability', fontsize=10, fontweight='black', color='#333F4B')
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')


def plot_cossell_curves_by_mouse(exp_group_label, subtract_val=None):
    mids = experimental_log.get_mouse_ids_in_experiment(exp_group_label)
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        t = mtg.pre_test_trials()[0]
        escape_rate = np.mean([t.is_flee() for t in mtg.pre_test_trials()[:3]])

        contrast = (subtract_val - float(t.contrast)) if subtract_val is not None else float(t.contrast)
        plt.plot(contrast, escape_rate, 'o', color='k', alpha=0.2)
