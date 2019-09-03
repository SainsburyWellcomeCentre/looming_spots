import numpy as np
from matplotlib import pyplot as plt

from looming_spots.analysis.randomised_contrast_escape_curves import get_contrast_escape_curve_from_group_label
from looming_spots.db import experimental_log, loom_trial_group
from looming_spots.preprocess import photodiode
import pandas as pd
import seaborn as sns


A10_naive_pre_tests = {'CA285_3', 'CA389_5', 'CA330_2', 'CA389_4', 'CA412_3', 'CA389_2', 'CA305_1', 'CA412_1', 'CA330_3', 'CA389_3', 'CA408_5', 'CA409_2', 'CA414_2', 'CA408_4'}


def plot_cossell_curves_pooled_trials():
    # TODO: implement in seaborn/pandas.. much simpler
    # i.e. with trials based metric dataframe the following will suffice:
    # sns.lineplot(x='contrast', y='escape', data=df, err_style='bars')


    plt.figure()
    plt.title('Escape probability (pooled trials) vs. contrast \n '
              'for spot or background luminance \n '
              'at low contrasts (n=4 mice per contrast)')

    curve, sems, contrasts = get_contrast_escape_curve_from_group_label('spot_contrast_cossel_curve', 0.1607)
    plt.errorbar(contrasts, curve, sems)

    curve, sems, contrasts = get_contrast_escape_curve_from_group_label('background_contrast_cossel_curve')
    plt.errorbar(contrasts, curve, sems)

    plt.xlabel('contrast a.u.', fontsize=10, fontweight='black', color='#333F4B')
    plt.ylabel('escape probability', fontsize=10, fontweight='black', color='#333F4B')
    plt.legend(['spot_luminance_cossel_curve', 'background_luminance_cossel_curve'])

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')


def plot_all_naive_test_trials():
    pass


def plot_all_lsie():
    pass


def plot_cossell_curves_by_mouse(exp_group_label, subtract_val=None):
    mids = experimental_log.get_mouse_ids_in_experiment(exp_group_label)
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        t = mtg.pre_test_trials()[0]
        escape_rate = np.mean([t.is_flee() for t in mtg.pre_test_trials()[:3]])

        contrast = (subtract_val - float(t.contrast)) if subtract_val is not None else float(t.contrast)
        plt.plot(contrast, escape_rate, 'o', color='k', alpha=0.2)


def plot_pre_test_effect():
    labels = ['pre_hab_post_immediate', 'pre_hab_post_24hr']
    for label in labels:

        mouse_ids = experimental_log.get_mouse_ids_in_experiment(label)

        fig, axes = plt.subplots(2, 1)
        plt.title(label)
        for mid in mouse_ids:
            try:
                mtg = loom_trial_group.MouseLoomTrialGroup(mid)
                ax = plt.sca(axes[0])
                for t in mtg.pre_test_trials()[:3]:
                    t.plot_track(ax)

                ax = plt.sca(axes[1])

                for t in mtg.post_test_trials()[:3]:
                    t.plot_track(ax)
            except photodiode.LoomNumberError as e:
                print(e)

        for ax in axes:
            plt.sca(ax)
            t.plot_stimulus()
            plt.subplots_adjust(hspace=0.6)

    plt.figure(figsize=(4, 7))
    plt.title('Post-LSIE Escape Probability \n (n=7 mice)')
    ltg = loom_trial_group.ExperimentalConditionGroup(labels)
    df = ltg.to_df('post_test', True)
    df = df.rename(columns={'classified as flee': 'escape probability'})
    sns.barplot(x='experimental condition', y="escape probability", data=df, errwidth=1)

    ax=plt.gca()
    plt.xticks(rotation=90)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1.2])
    plt.subplots_adjust(bottom=0.35, left=0.3, right=0.8)


def plot_pre_test_effect_post_test_only():
    labels = ['pre_hab_post_immediate', 'pre_hab_post_24hr']
    for label in labels:

        mouse_ids = experimental_log.get_mouse_ids_in_experiment(label)

        fig = plt.figure(figsize=(7, 2))
        plt.title(label)
        ax = plt.subplot(111)

        for mid in mouse_ids:
            try:
                mtg = loom_trial_group.MouseLoomTrialGroup(mid)
                for t in mtg.post_test_trials()[:3]:
                    t.plot_track(ax)
            except photodiode.LoomNumberError as e:
                print(e)

        t.plot_stimulus()
        plt.subplots_adjust(hspace=0.6)

    plt.figure(figsize=(4, 7))
    plt.title('Post-LSIE Escape Probability \n (n=7 mice)')
    ltg = loom_trial_group.ExperimentalConditionGroup(labels)
    df = ltg.to_df('post_test', True)
    df = df.rename(columns={'classified as flee': 'escape probability'})
    sns.barplot(x='experimental condition', y="escape probability", data=df, errwidth=1)

    ax=plt.gca()
    plt.xticks(rotation=90)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1.2])
    plt.subplots_adjust(bottom=0.35, left=0.3, right=0.8)


def get_group_avg_df(df, key='experimental condition'):
    all_groups_df = pd.DataFrame()

    for label in df[key].unique():
        sub_df = df[df[key] == label].mean()
        b = pd.DataFrame([list(sub_df.mean().keys()),
                          list(sub_df.mean().values),
                          [label] * len(sub_df.mean().values)
                          ],
                         index=['metric', 'value', 'experimental group'])
        all_groups_df = all_groups_df.append(b)
        # sub_df_mean = pd.concat([sub_df_mean,
        #                         pd.Series([label, 'mean'], index=['experimental condition', 'metric'])])
        # all_groups_df = all_groups_df.append(sub_df_mean, ignore_index=True)
        #
        # sub_df_std = df[df[key] == label].std()
        # sub_df_std = pd.concat([sub_df_std,
        #                        pd.Series([label, 'std'], index=['experimental condition', 'metric'])])
        #
        # all_groups_df = all_groups_df.append(sub_df_std, ignore_index=True)
    return all_groups_df


def compare_groups_lsie_exploration(groups=('pre_hab_post_immediate', 'pre_hab_post_24hr')):
    import matplotlib.pyplot as plt
    from looming_spots.db import loom_trial_group, experimental_log
    from looming_spots.analysis.trial_group_analysis import make_trial_heatmap_location_overlay

    immediate = experimental_log.get_mouse_ids_in_experiment(groups[0])
    day_before = experimental_log.get_mouse_ids_in_experiment(groups[1])

    plt.close('all')
    for mids, group in zip([immediate, day_before], [groups[0], groups[1]]):
        plt.figure()
        for i, mid in enumerate(mids):
            trials = []
            mtg = loom_trial_group.MouseLoomTrialGroup(mid)
            for t in mtg.habituation_trials():
                trials.extend([t])
            hm = make_trial_heatmap_location_overlay(trials)
            ax = plt.subplot(2, 7, i+1)
            ax.title.set_text('{} {}'.format(group, mid))
            plt.imshow(hm, aspect='auto', vmax=2, vmin=0, interpolation='bilinear')
            ax.axis('off')
            plt.ylim(0, 300)
            plt.xlim(0, 400)
            ax2=plt.subplot(2, 7, i+8)
            for t in mtg.post_test_trials()[:3]:
                t.plot_track()
            t.plot_stimulus()

    df = pd.DataFrame()
    plt.figure()
    for mids, group in zip([immediate, day_before], ['immediate', 'day before']):
        for i, mid in enumerate(mids):
            event_metric_dict = {}
            mtg = loom_trial_group.MouseLoomTrialGroup(mid)
            event_metric_dict.setdefault('mouse id', [mid])
            event_metric_dict.setdefault('group', [group])
            event_metric_dict.setdefault('percentage time in tz middle', [mtg.percentage_time_in_tz_middle()])
            df = df.append(pd.DataFrame.from_dict(event_metric_dict), ignore_index=True)

    df.boxplot(by='group', rot=90, grid=False)

