import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


from looming_spots.analysis import plotting
from looming_spots.db.loom_trial_group import LoomTrialGroup

from looming_spots.db.experimental_log import get_mouse_ids_in_experiment
from looming_spots.db.session_group import MouseSessionGroup


def get_flee_probabilities_from_trials(trials):
    all_mouse_ids = set([t.mouse_id for t in trials])
    flee_probabilities = []
    n_flees_all = []
    n_non_flees_all = []

    for id in all_mouse_ids:
        mouse_trial_results = [t.is_flee() for t in trials if t.mouse_id == id]
        n_flees = np.count_nonzero(mouse_trial_results)
        n_non_flees = len(mouse_trial_results) - n_flees
        flee_probability = n_flees / len(mouse_trial_results)
        flee_probabilities.append(flee_probability)
        n_flees_all.append(n_flees)
        n_non_flees_all.append(n_non_flees)
    return n_flees_all, n_non_flees_all, flee_probabilities


def plot_flee_probabilities(experimental_labels, trial_type='pre_test', ax=None):
    df = pd.DataFrame()
    for experimental_group_label in experimental_labels:
        df_dict = {}
        trials = load_trials_from_label(experimental_group_label, trial_type)
        n_flees, n_non_flees, flee_probabilities = get_flee_probabilities_from_trials(trials)

        df_dict.setdefault('experimental group', [experimental_group_label]*len(flee_probabilities))
        df_dict.setdefault('n flees', n_flees)
        df_dict.setdefault('n non flees', n_non_flees)
        df_dict.setdefault('flee probability', flee_probabilities)

        group_df = pd.DataFrame.from_dict(df_dict)
        df = df.append(group_df, ignore_index=True)
    df.boxplot(by='experimental group', ax=ax, rot=90, grid=False)
    plotting.format_plots(ax)
    return df


def plot_all_metrics_mouse_avgs(experimental_labels, trial_type='pre_test', ax=None):
    all_df = pd.DataFrame()
    for experimental_group_label in experimental_labels:
        trials = load_trials_from_label(experimental_group_label, trial_type)
        ltg = LoomTrialGroup(trials, experimental_group_label)
        df = ltg.get_mouse_avg_df(trials, experimental_group_label)
        all_df = all_df.append(df)
    all_df.boxplot(by='experimental group', ax=ax, rot=90, grid=False)
    plotting.format_plots(ax)
    return all_df


def plot_all_metrics_trials(experimental_group_labels, trial_type='pre_test'):
    all_dfs = pd.DataFrame()
    for experimental_group_label in experimental_group_labels:
        trials = load_trials_from_label(experimental_group_label, trial_type)
        ltg = LoomTrialGroup(trials, experimental_group_label)

        df = ltg.to_df()
        all_dfs = all_dfs.append(df, ignore_index=True)

    fig, axes = plt.subplots(1, len(LoomTrialGroup.analysed_metrics()))

    all_dfs.boxplot(column=LoomTrialGroup.analysed_metrics(), by='condition', ax=axes, grid=False, rot=90)
    plotting.format_plots(axes)
    return all_dfs


def get_trials(mouse_ids, trial_type):
    trials = []
    for mid in mouse_ids:
        msg = MouseSessionGroup(mid)
        pre, post = msg.get_pre_and_post_test_trials()
        if trial_type == 'pre_test':
            trials.extend(pre)
        elif trial_type == 'post_test':
            trials.extend(post)
    return trials


def load_trials_from_label(experimental_group_label, trial_type='pre_test'):
    trials = []
    mouse_ids = get_mouse_ids_in_experiment(experimental_group_label)
    return get_trials(mouse_ids, trial_type)


def plot_trials_with_habituation(mtg, trial_type='pre_test', habit_limit=5):
    """
    
    :param MouseTrialGroup mtg:  
    :param trial_type: 
    :param habit_limit: 
    :return: 
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 1))
    for t in mtg.get_trials_of_type(trial_type):
        plt.sca(axes[1])
        color = 'r' if t.is_flee() else 'k'
        plt.plot(t.normalised_x_track, color=color)
    for t in mtg.get_trials_of_type('habituation')[0:habit_limit]:
        plt.sca(axes[0])
        t.plot_track_on_image(180, -200)

        plt.sca(axes[2])

    plotting.plot_looms_ax(axes[1])
