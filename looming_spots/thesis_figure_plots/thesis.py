import itertools
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import pingouin as pg
import numpy as np
import seaborn as sns

from looming_spots.db import loom_trial_group
from looming_spots.thesis_figure_plots import randomised_contrast_escape_curves_lesions
from looming_spots.thesis_figure_plots.randomised_contrast_escape_curves_lesions import get_behaviour_metrics_df, \
    get_escape_curve_df
from looming_spots.trial_group_analysis.escape_metric_dataframes import get_behaviour_metric_dataframe

from looming_spots.trial_group_analysis.photometry_habituations import get_signal_metric_dataframe_variable_contrasts
from looming_spots.trial_group_analysis import escape_metric_dataframes


def plot_example_tracks_var_contrast(mids=('CA452_1',)):
    """"""
    norm = mpl.colors.Normalize(vmin=0, vmax=0.1607)
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        for t in mtg.all_trials[:18]:
            c = tuple([norm(t.contrast)]*3)
            t.plot_track(color=c)
        plt.xlim([0, 600])
    t.plot_stimulus()


def plot_all_metrics(mids=('CA452_1',), metrics=('speed', 'acceleration', 'time in safety zone')):
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    fig, axes = plt.subplots(1, len(metrics)+1)
    all_df = pd.DataFrame()
    for metric, ax in zip(metrics, axes):
        plt.sca(ax)
        df = get_behaviour_metric_dataframe(mtgs, metric, "variable_contrast")
        sns.scatterplot(data=df, x='contrast', y='metric value', hue='contrast', palette='Greys_r', edgecolor='k', linewidth=0.5)
        sns.lineplot(data=df, x='contrast', y='metric value', err_style='bars', color='k')
        ax.invert_xaxis()
        ax.set_ylabel(f'{metric}')
        ax.set_xlabel('spot luminance (a.u.)')
        all_df = all_df.append(df)

    plt.sca(axes[-1])
    ax = sns.lineplot(data=df, x='contrast', y='escape', err_style='bars', color='k')
    ax.invert_xaxis()
    ax.set_ylabel('escape (%)')
    ax.set_ylabel('spot luminance (a.u.)')
    return all_df


def make_multivariable_plotting_df(df):
    df_multivar = pd.DataFrame()
    for metric in df['metric'].unique():
        values = df[df['metric'] == metric]['metric value'].values
        df_multivar[metric] = values
        df_multivar['experimental group'] = df[df['metric'] == metric]['experimental group'].values
        df_multivar['contrast'] = df[df['metric'] == metric]['contrast'].values
        df_multivar['escape'] = df[df['metric'] == metric]['escape'].values
        df_multivar['loom number'] = df[df['metric'] == metric]['loom number'].values
    return df_multivar


def get_main_metric_df_multiple_groups(group_keys=('CONTROL', 'OHDA'),
                                       metrics=('speed',
                                                'acceleration',
                                                'time in safety zone',
                                                'latency to escape',
                                                'reaction time', 'loom number'),):

    main_df = pd.DataFrame()

    for metric in metrics:
        df = get_behaviour_metrics_df(metric, group_keys=group_keys)
        main_df = main_df.append(df, ignore_index=True)

    return main_df


def plot_lesion_group_comparison(group_keys=('CONTROL', 'OHDA'), reload=False,
                                 save_dir='/home/slenzi/looming/dataframes/',
                                 figure_dir='/home/slenzi/looming/figures/',
                                 fmt='png', fname='chapter_3_raw.csv', stats=False):

    set_plotting_globals()
    flatui = ["#34495e", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71"]
    metric_labels = [ 'escape','speed', 'latency peak detect', 'reaction time', 'time in safety zone']

    def format_plots(title, flip_x=True):
        plt.title(title)
        if flip_x:
            plt.xlim(reversed(plt.xlim()))
        sns.despine(offset=10, trim=True)

    save_path = Path(save_dir) / fname

    if not os.path.isfile(str(save_path)) or reload:
        print('recalculating...')
        main_metric_df_all_groups = get_main_metric_df_multiple_groups(
            metrics=('latency peak detect', 'speed',
                     'time in safety zone', 'time to reach shelter stimulus onset',
                     'time to reach shelter detection', 'reaction time'),
            group_keys=group_keys)

        main_metric_df_all_groups.to_csv(str(save_path))

    main_metric_df_all_groups = pd.DataFrame.from_csv(save_path)
    main_metric_df_all_groups = main_metric_df_all_groups[main_metric_df_all_groups['experimental group'].isin(group_keys)]

    df = make_multivariable_plotting_df(main_metric_df_all_groups)
    df['contrast'] = df['contrast'].astype(float)
    df['loom number'] = df['loom number'].astype(int)

    # plot figures
    plot_all_metrics_against_contrast(df, figure_dir, flatui, format_plots, metric_labels, fmt=fmt)
    plot_all_metric_distributions(df, figure_dir, flatui, metric_labels, fmt=fmt)

    first_label = main_metric_df_all_groups['metric'].unique()[0]
    single_metric_df = main_metric_df_all_groups[main_metric_df_all_groups['metric'] == first_label]

    plot_test_contrast_trials_in_order(figure_dir, flatui, single_metric_df, fmt=fmt)
    plot_all_metrics_bars_at_test_contrast(figure_dir, flatui, format_plots, main_metric_df_all_groups, metric_labels, fmt=fmt)


    if stats:
        # produce stats table and save
        for combination in itertools.combinations(group_keys, 2):
            compute_stats_mixed_anova(combination[0], combination[1], main_df_name=fname)

        for metric in ['speed', 'reaction time', 'time in safety zone', 'latency peak detect']:
            subdf = main_metric_df_all_groups[main_metric_df_all_groups['metric'] == metric]
            a = pg.pairwise_ttests(
                dv="metric value",
                between="experimental group",
                subject="mouse id",
                data=subdf,
            )
            print(a.to_latex())

            a.to_csv(str(Path(save_dir) / f'stats_ttest_test_contrast_{metric}.csv'))

    return main_metric_df_all_groups


def set_plotting_globals():
    sns.set_style("white")
    sns.set_style("ticks")

    mpl.rcParams['axes.linewidth'] = 3
    mpl.rcParams['axes.titlesize'] = 24
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.minor.width'] = 3
    mpl.rcParams['xtick.major.width'] = 3
    mpl.rcParams['ytick.major.width'] = 3
    mpl.rcParams['ytick.major.width'] = 3
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['font.weight'] = 'bold'


def plot_all_metrics_bars_at_test_contrast(figure_dir,
                                           flatui,
                                           format_plots,
                                           main_metric_df_all_groups,
                                           metric_labels,
                                           fmt='eps'
                                           ):
    fig = plt.figure()
    title = 'all metrics at test contrast'
    barplot_df = main_metric_df_all_groups[main_metric_df_all_groups['metric'].isin(metric_labels)]
    sns.barplot('metric value', 'metric', data=barplot_df, hue='experimental group', palette=flatui)
    format_plots(title, flip_x=False)
    plt.subplots_adjust(left=0.35)
    fig.savefig(str(Path(figure_dir) / (title + f'.{fmt}')))


def plot_test_contrast_trials_in_order(figure_dir, flatui, single_metric_df, fmt='eps'):
    fig = plt.figure()
    title = 'Escape (%) at Test Contrast'
    plt.title(title)
    single_metric_df = single_metric_df[single_metric_df['contrast'] == 0]
    b=sns.pointplot(x='loom number',
                    y='escape',
                    hue='experimental group',
                    data=single_metric_df,
                    palette=flatui,
                    scale=2)

    [print('escape', line.get_data()) for line in b.get_lines()]
    single_metric_df['loom number'] = single_metric_df['loom number'].astype(str)
    single_metric_df['escape'] = single_metric_df['escape'].astype(int)
    aov = pg.mixed_anova(
        dv="escape",
        within="loom number",
        between="experimental group",
        subject="mouse id",
        data=single_metric_df,
    )
    print(aov)

    for i, group in enumerate(single_metric_df['experimental group'].unique()):
        plt.figure()
        plt.title(group)
        groupdf = single_metric_df[single_metric_df['experimental group'] == group]
        sns.swarmplot(x='loom number', y='escape', hue='mouse id', data=groupdf)

    sns.despine(trim=True)
    fig.savefig(str(Path(figure_dir) / (title + f'.{fmt}')))


def plot_all_metric_distributions(df, figure_dir, flatui, metric_labels, xlim_upper=120, fmt='eps'):
    for metric in metric_labels:
        if metric == 'escape':
            continue
        fig = plt.figure()
        title = f"{metric} distribution all contrasts"
        plt.title(title)
        y_lim = 0
        # df0 = df[df['contrast']==0]
        df0 = df
        for i, group_df_label in enumerate(df['experimental group'].unique()):
            group_df = df0[df0['experimental group'] == group_df_label]
            n_bins = abs((np.nanmax(group_df[metric]) - np.nanmin(group_df[metric])) / 3)
            print(f'number of bins = {n_bins}')

            sns.distplot(group_df[metric], bins=int(n_bins), norm_hist=True, color=flatui[i], hist_kws={'linewidth': 0})
            y_lim = max(max(plt.ylim()), y_lim)
        plt.ylim([0, y_lim])
        plt.xlim([-20, xlim_upper])
        sns.despine(offset=10, trim=True)
        fig.savefig(str(Path(figure_dir) / (title + f'.{fmt}')))


def plot_all_metrics_against_contrast(df, figure_dir, flatui, format_plots, metric_labels, fmt='eps'):
    for metric in metric_labels:
        fig = plt.figure()
        title = f"{metric} vs. contrast"
        b=sns.pointplot(x='contrast', y=metric, hue='experimental group', data=df, palette=flatui, scale=2)
        format_plots(title, flip_x=True)
        fig.savefig(str(Path(figure_dir) / (title + f'.{fmt}')))
        #[print(metric,  line.get_data()) for line in b.get_lines()]



def plot_reaction_time_at_test_contrast(df):
    sub_df = make_multivariable_plotting_df(df)
    sns.barplot('experimental group', 'reaction time', data=sub_df)


def compute_stats_mixed_anova(group_label1,
                              group_label2,
                              save_dir='/home/slenzi/looming/dataframes/',
                              main_df_name='chapter_3_raw.csv'):

    save_dir = Path(save_dir)
    main_df_path = save_dir / main_df_name
    mixed_anova_result_path = save_dir / f'mixed_anova_{group_label1}_vs_{group_label2}.csv'
    mixed_anova_post_hocs_path = save_dir / f'mixed_anova_post_hocs_{group_label1}_vs_{group_label2}.csv'

    main_metric_df_all_groups = pd.read_csv(str(main_df_path))

    first_label = main_metric_df_all_groups['metric'].unique()[0]
    single_metric_df = main_metric_df_all_groups[main_metric_df_all_groups['metric'] == first_label]
    two_groups_df = single_metric_df[single_metric_df['experimental group'].isin([group_label1, group_label2])]
    escape_curve_df = randomised_contrast_escape_curves_lesions.get_escape_curve_df(two_groups_df)

    # assumes all categories are categorical/iterable types
    escape_curve_df['contrast'] = escape_curve_df.astype(str)

    aov = pg.mixed_anova(
        dv="escape",
        within="contrast",
        between="experimental group",
        subject="mouse id",
        data=escape_curve_df,
    )

    aov.to_csv(str(mixed_anova_result_path))
    print(aov.to_latex())

    summary_table = pg.pairwise_ttests(
        dv="escape",
        within="contrast",
        between="experimental group",
        subject="mouse id",
        data=escape_curve_df,
    )

    summary_table.to_csv(str(mixed_anova_post_hocs_path))
    print(summary_table[summary_table['Contrast'] == 'contrast * experimental group'].to_latex())


def two_way_mixed_anova(df, post_hoc=False):
    """
    perform two way mixed ANOVA on two groups of mice with variable contrast curve experiment

    :param str group_label_1:
    :param str group_label_2:
    :return:
    """

    print(df.unique('experimental group'))

    escape_curve_df = get_escape_curve_df(df)

    aov = pg.mixed_anova(
        dv="escape",
        within="contrast",
        between="experimental group",
        subject="mouse id",
        data=escape_curve_df,
    )
    pg.print_table(aov)

    if post_hoc:
        pg.pairwise_ttests(
            dv="escape",
            within="contrast",
            between="experimental group",
            subject="mouse id",
            data=escape_curve_df,
        )
    return escape_curve_df, aov


def get_group_track_statistics(mids, bins):
    all_speeds = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        t = mtg.all_trials[0]
        x, y = t.session.track()
        s = t.sample_number - (30 * 5 * 60)
        e = t.sample_number
        speed = np.abs(np.diff((1 - x[s:e])))
    all_speeds.extend(speed)
    plt.hist(all_speeds, bins=bins)
    return all_speeds


def chapter_4_LSIE():
    GROUPS = {
        "naive": [],
        "lsie": [],
    }

    pass


def chapter_5():

    GROUPS = {
              "d1 photometry": [],
              "d1 photometry_var_contrast": [],
              "d2 photometry": [],
              "d2 photometry_var_contrast": [],
              "d1 caspase": [],
              "d2 caspase": [],
              }
    pass



def ohda_imaging():
    from looming_spots.db import loom_trial_group
    mids = ['FJ5_2', 'FJ5_3']
    import matplotlib.pyplot as plt
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)

        plt.figure()
        avg_loom = []
        for i, t in enumerate(mtg.loom_trials()):
            if t.contrast == 0:
                color = 'r' if t.is_flee() else 'k'
                plt.plot(t.delta_f()[:600] * 20, color=color, linestyle='-')
                t.plot_track()
                t.plot_stimulus()

        plt.figure()
        for i, t in enumerate(mtg.auditory_trials()):
            color = 'r' if t.is_flee() else 'k'
            plt.plot(t.delta_f()[:600] * 20, color=color, linestyle='-')
            t.plot_track()
            t.plot_stimulus()


def get_time_series_df(mid):
    mtg = loom_trial_group.MouseLoomTrialGroup(mid)
    df_dict={}
    trial_number = []
    signal = []
    track = []
    timepoint = []
    trial_type = []

    for t in mtg.loom_trials():
        if t.contrast == 0:
            sig = t.delta_f()[:600]
            signal.extend(sig)
            track.extend(t.normalised_x_track[:600])
            trial_number.extend([t.loom_number]*len(sig))
            trial_type.extend(['looming']*len(sig))
            timepoint.extend(list(range(len(sig))))

    for t in mtg.auditory_trials():
        sig = t.delta_f()[:600]
        signal.extend(sig)
        track.extend(t.normalised_x_track[:600])
        trial_number.extend([t.auditory_number] * len(sig))
        trial_type.extend(['auditory'] * len(sig))
        timepoint.extend(list(range(len(sig))))

    df_dict.setdefault('trial number', trial_number)
    df_dict.setdefault('signal', signal)
    df_dict.setdefault('track', track)
    df_dict.setdefault('timepoint', timepoint)
    df_dict.setdefault('trial type', trial_type)
    return pd.DataFrame.from_dict(df_dict)


def plot_scatter_by_contrast(mids, group_label,
                             figure_dir='/home/slenzi/looming/figures/photometry_var_contrast',
                             fmt='png'):

    set_plotting_globals()
    figure_dir = Path(figure_dir)
    mtgs = [
        loom_trial_group.MouseLoomTrialGroup(mid)
        for mid in mids
    ]

    all_plots = []
    metrics = [
        "speed",
        "acceleration",
        "latency peak detect",
        "time in safety zone",
    ]
    for metric in metrics:
        df = get_signal_metric_dataframe_variable_contrasts(mtgs, metric)
        g = sns.lmplot(
            metric,
            "ca signal",
            data=df,
            hue="contrast",
            fit_reg=False,
            palette=sns.cubehelix_palette(7, hue=0)[::-1],
            scatter_kws={"s": 100},
        )

        xlim_upper = df[metric].quantile(0.95)
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, xlim_upper])
        g.savefig(str(figure_dir / f'contrast_signal_{metric}_{group_label}.{fmt}'), fmt=fmt)
        all_plots.append(g)

    df = df.sort_values(by='contrast')
    df['contrast'] = df['contrast'].astype(str)
    g = sns.lmplot(
        "contrast",
        "ca signal",
        data=df,
        hue='escape',
        fit_reg=False,
        scatter_kws={"s": 100},
    )
    sns.pointplot('contrast', 'ca signal', data=df, err_style='bars', scale=2)
    sns.pointplot('contrast', 'escape', color='k', data=df, err_style='bars', scale=2)

    g.savefig(str(figure_dir / f'contrast_signal_escape_{group_label}.{fmt}'), fmt=fmt)
    fig=plt.figure()
    n_bins = 10
    df_escape = df[df['escape']]
    df_no_escape = df[~df['escape']]
    sns.distplot(df_no_escape['ca signal'], bins=int(n_bins), norm_hist=True, hist_kws={'linewidth': 0})
    sns.distplot(df_escape['ca signal'], bins=int(n_bins), norm_hist=True, hist_kws={'linewidth': 0})
    fig.savefig(str(figure_dir / f'escape_displot_signal_{group_label}.{fmt}'), fmt=fmt)
    return all_plots, df, mtgs


def plot_ohda_with_photometry(mid):
    df = get_time_series_df(mid)
    sns.lineplot(data=df, x='timepoint', y='signal', style='trial type', hue='trial type')
    return df


def plot_snl_laser_stim(mids):
    pass


def get_df_metrics_as_columns(mids,
                              metrics = ('speed', 'latency peak detect', 'time to reach shelter stimulus onset'),
                              experimental_group_label='control'):
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]

    df = escape_metric_dataframes.get_behaviour_metrics_dataframe(mtgs,
                                                                  metrics,
                                                                  'variable_contrast',
                                                                  experimental_group_label)
    return df


def get_track_df(mids):
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    df = escape_metric_dataframes.get_track_dataframe(mtgs, 'variable_contrast')
    return df


def get_track_df_all(groups):
    all_df = pd.DataFrame()
    for k, mids in groups.items():
        df = get_track_df(mids)
        all_df = all_df.append(df, ignore_index=True)
    return all_df


