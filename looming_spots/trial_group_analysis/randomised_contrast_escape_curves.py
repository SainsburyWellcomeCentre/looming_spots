import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt

# from looming_spots.deprecated.analysis import plot_trials
from looming_spots.db import loom_trial_group, experimental_log


"""

functions relating to pseudorandom presentations of low-or-high contrast looms

"""


def plot_block_escape_curves_with_avg(mids, color="k", block_id=0):
    mtg = loom_trial_group.MouseLoomTrialGroup(mids[0])
    contrast_set = np.unique(mtg.contrasts())
    all_escape_curves = []
    for mid in mids:
        escape_curve = get_pooled_escape_probilities_all_contrasts_block(
            [mid], contrast_set, block_id=0
        )
        all_escape_curves.append(escape_curve)
        # plt.plot(contrast_set, escape_curve, linewidth=0.5, color=color, alpha=0.3)

    avg_escape_curve = np.nanmean(all_escape_curves, axis=0)
    sem_escape_curve = scipy.stats.sem(all_escape_curves, axis=0)

    plt.plot(
        np.unique(mtg.contrasts()), avg_escape_curve, color=color, linewidth=3
    )
    plt.scatter(
        np.unique(mtg.contrasts()),
        avg_escape_curve,
        color="w",
        edgecolor=color,
        zorder=10,
    )

    for i, (contrast, error, value) in enumerate(
        zip(contrast_set, sem_escape_curve, avg_escape_curve)
    ):
        plt.errorbar(contrast, value, error, color=color)

    plt.ylim([-0.01, 1.1])
    plt.xlim([0.165, -0.01])
    plt.xlabel("contrast", fontsize=10, fontweight="black", color="#333F4B")
    plt.ylabel("escape %", fontsize=10, fontweight="black", color="#333F4B")


def get_pooled_escape_probilities_all_contrasts_block(
    mouse_ids, contrast_set, block_id=0
):
    escape_curve = []
    for c in contrast_set:
        trials = get_trials_of_contrast_mouse_group(
            mouse_ids, c, start=block_id * 18, end=(block_id + 1) * 18
        )
        avg_contrast_probability = np.nanmean([t.is_flee() for t in trials])
        escape_curve.append(avg_contrast_probability)

    return escape_curve


def get_trials_of_contrast(trials, c):
    return [t for t in trials if t.contrast == c]


def get_trials_of_contrast_mouse_group(mids, c, start=0, end=18):
    from looming_spots.db import loom_trial_group

    all_trials = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        trials = [t for t in mtg.all_trials[start:end] if t.contrast == c]
        all_trials.extend(trials)
    return all_trials


def get_all_trials(experimental_group_label):
    mids = experimental_log.get_mouse_ids_in_experiment(
        experimental_group_label
    )
    all_trials = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        trials = mtg.pre_test_trials()[:3]
        all_trials.extend(trials)
    return all_trials


def get_contrast_escape_curve_from_group_label(
    experimental_group_label, subtract_val=None
):
    all_trials = get_all_trials(experimental_group_label)
    contrasts = [t.contrast for t in all_trials]
    all_contrasts = np.unique(contrasts)
    avg_curve = []
    sem_curve = []

    for c in all_contrasts:
        trials_of_contrast = get_trials_of_contrast(all_trials, c)
        avg_curve.append(np.mean([t.is_flee() for t in trials_of_contrast]))
        sem_curve.append(
            scipy.stats.sem([t.is_flee() for t in trials_of_contrast])
        )

    if subtract_val is not None:
        all_contrasts = subtract_val - np.array(all_contrasts)

    return avg_curve, sem_curve, all_contrasts


def sort_trials_by_contrast(trials):

    grouped_trials = []
    all_contrasts = sorted(set([t.contrast for t in trials]))
    for contrast in all_contrasts:
        all_in_condition = [t for t in trials if t.contrast == contrast]
        grouped_trials.append(all_in_condition)
    return grouped_trials


def plot_all_contrasts(trials):
    grouped_trials = sort_trials_by_contrast(trials)
    fig, axes = plt.subplots(len(grouped_trials), 1)
    for gt, ax in zip(grouped_trials, axes):
        plt.sca(ax)
        # plot_trials(gt)
