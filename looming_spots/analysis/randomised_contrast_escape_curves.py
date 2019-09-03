import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt

from looming_spots.analysis.analysis import plot_trials
from looming_spots.db import loom_trial_group, experimental_log


def get_trials_of_contrast(trials, c):
    return [t for t in trials if t.contrast == c]


# def get_binned_escape_probabilities_at_contrast(trials, contrasts, c, group_by_idx=2):
#     grouped_by = get_grouped_trials_for_contrast(c, contrasts, group_by_idx, trials)
#
#     all_escape_probabilities = []
#     for group in grouped_by:
#         classified_as_flee = []
#         for t in group:
#             classified_as_flee.append(t.is_flee())
#         all_escape_probabilities.append(np.mean(classified_as_flee))
#     return all_escape_probabilities


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


# def get_pooled_escape_probilities_all_contrasts(mouse_ids, contrast_set, n_blocks=2):
#     block_1_escape_curve = []
#     block_2_escape_curve = []
#     for c in contrast_set:
#         block_1_summary = []
#         block_2_summary = []
#         for mid in mouse_ids:
#             mtg = loom_trial_group.MouseLoomTrialGroup(mid)
#             groupbyidx = 6 if c == 0 else 2
#             block_1, block_2 = get_binned_escapes_at_contrast(mtg.all_trials[:18*n_blocks], mtg.contrasts(), c, group_by_idx=groupbyidx)
#             block_1_summary.extend(block_1)
#             block_2_summary.extend(block_2)
#
#         percent_at_contrast_block1 = np.count_nonzero(block_1_summary) / len(block_1_summary)
#         percent_at_contrast_block2 = np.count_nonzero(block_2_summary) / len(block_2_summary)
#         block_1_escape_curve.append(percent_at_contrast_block1)
#         block_2_escape_curve.append(percent_at_contrast_block2)
#     return block_1_escape_curve, block_2_escape_curve


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


def get_all_trials(experimental_group_label):
    mids = experimental_log.get_mouse_ids_in_experiment(experimental_group_label)
    all_trials = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        trials = mtg.pre_test_trials()[:3]
        all_trials.extend(trials)
    return all_trials


def get_contrast_escape_curve_from_group_label(experimental_group_label, subtract_val=None):
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


def plot_cossell_curves_by_mouse(exp_group_label, subtract_val=None):
    mids = experimental_log.get_mouse_ids_in_experiment(exp_group_label)
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        t = mtg.pre_test_trials()[0]
        escape_rate = np.mean([t.is_flee() for t in mtg.pre_test_trials()[:3]])

        contrast = (subtract_val - float(t.contrast)) if subtract_val is not None else float(t.contrast)
        plt.plot(contrast, escape_rate, 'o', color='k', alpha=0.2)


def plot_all_contrasts(trials):
    grouped_trials = sort_trials_by_contrast(trials)
    fig, axes = plt.subplots(len(grouped_trials), 1)
    for gt, ax in zip(grouped_trials, axes):
        plt.sca(ax)
        plot_trials(gt)


def sort_trials_by_contrast(trials):

    grouped_trials = []
    all_contrasts = sorted(set([t.contrast for t in trials]))
    for contrast in all_contrasts:
        all_in_condition = [t for t in trials if t.contrast == contrast]
        grouped_trials.append(all_in_condition)
    return grouped_trials
