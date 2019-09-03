import numpy as np
from looming_spots.db.constants import FIGURE_DIRECTORY
from looming_spots.db import loom_trial_group
from looming_spots.analysis import photometry_habituations
import seaborn as sns

import matplotlib.pyplot as plt

pre_test_24hr = ['074743', '074744', '074746', '895773', '953828', '953829']
pre_test_sameday = ['898989', '898990']
contrast_curves = ['898992', '916063', '921000', '907822']

any_escape = ['074743', '895773', '953828', '898989', '898990']
no_escape = ['074746', '957073']


def plot_pre_post_photometry_trials_lsie(mtg):
    for t in mtg.pre_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig('/{}/loom_{}_{}_pre_test.eps'.format(FIGURE_DIRECTORY, t.loom_number, mtg.mouse_id), format='eps')
        plt.close('all')
    for t in mtg.post_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig('/{}/loom_{}_{}_post_test.eps'.format(FIGURE_DIRECTORY, t.loom_number, mtg.mouse_id), format='eps')
        plt.close('all')


def plot_pre_post_photometry_lsie(mtg):

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


def plot_LSIE_evoked_signals(groups=(pre_test_sameday, pre_test_24hr), colors=('b', 'k'), labels=('same day pre-test (n={})', '24 hr pre-test n={}')):
    plt.figure()
    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]

        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
    plt.legend([labels[0].format(len(groups[0])), labels[1].format(len(groups[1]))])
    plt.title('loom evoked DA signal in ToS during LSIE protocol')
    plt.xlabel('contrast (binned 3 trials each)')
    plt.ylabel('integral of dF/F at avg. pre-test escape latency')
    plt.show()


def plot_LSIE_evoked_signals_all_mice(groups=(pre_test_sameday, pre_test_24hr), colors=('b', 'k'), labels=('same day pre-test (n={})', '24 hr pre-test n={}')):
    plt.figure()
    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
    plt.legend([labels[0].format(len(groups[0])), labels[1].format(len(groups[1]))])  # two loops because legend

    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        for mtg in mtgs:
            photometry_habituations.plot_habituation_curve_with_sem([mtg], color, linewidth=0.5, plot_dots=False)

    plt.title('loom evoked DA signal in ToS during LSIE protocol')
    plt.xlabel('contrast (binned 3 trials each)')
    plt.ylabel('integral of dF/F at avg. pre-test escape latency')
    plt.show()


def plot_LSIE_bars(groups=(pre_test_sameday, pre_test_24hr)):
    #pre_test_24hr.remove('074744')
    mtgs_sup = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in groups[1]]
    mtgs_non_sup = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in groups[0]]

    fig = plt.figure(figsize=(3, 5))
    plt.title('ToS DA signal before and after LSIE protocol')
    plt.subplot(121)
    plt.ylabel('integral dF/F at avg. escape latency in pre-test')
    photometry_habituations.plot_integral_at_latency_bars(mtgs_sup, ('Grey', 'Grey'))
    plt.subplot(122)
    photometry_habituations.plot_integral_at_latency_bars(mtgs_non_sup, ('b', 'b'))
    for ax in fig.axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1.2])
        plt.sca(ax)
        plt.subplots_adjust(bottom=0.35, left=0.3, right=0.8)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_escape_metrics_variable_contrast_experiments():
    from looming_spots.db import loom_trial_group
    from looming_spots.analysis import photometry_habituations
    import seaborn as sns

    all_dfs = []
    mouse_ids = ['898992', '916063', '921000', '907822']
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]

    for metric in ['latency to escape', 'speed', 'acceleration', 'time in safety zone']:
        df = photometry_habituations.get_signal_metric_dataframe_variable_contrasts(mtgs, metric)
        all_dfs.append(df)
    metrics = ['latency to escape', 'speed', 'acceleration', 'time in safety zone']

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(metric, 'ca signal', df, fit_reg=True,
                         palette=sns.cubehelix_palette(8)[::-1], legend=False)

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(metric, 'ca signal', df, hue='escape', fit_reg=False, legend=True)

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(metric, 'ca signal', df, hue='contrast', fit_reg=False,
                         palette=sns.cubehelix_palette(8)[::-1], legend=False)


def plot_ca_vs_metric_signal_before_after_lsie():
    all_dfs = []
    mouse_ids = ['074743', '074746', '895773', '953828', '953829']

    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in ['latency to escape', 'speed', 'acceleration', 'time in safety zone']:
        df = photometry_habituations.get_signal_metric_dataframe(mtgs, metric)
        all_dfs.append(df)

    metrics = ['latency to escape', 'speed', 'acceleration', 'time in safety zone']
    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        g = sns.lmplot(metric, 'ca signal', data=df, hue='escape', fit_reg=False)
        sns.regplot(x=metric, y='ca signal', data=df, scatter=False, ax=g.axes[0, 0], color='k')


def plot_lsie_suppression_over_variable_contrast(hue='test type'):
    all_dfs = []
    mouse_ids = ['898992', '916063', '921000', '907822']
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in ['latency to escape', 'speed', 'acceleration', 'time in safety zone']:
        df = photometry_habituations.get_signal_metric_dataframe_variable_contrasts(mtgs, metric)
        all_dfs.append(df)
    all_dfs_pre_post = []

    mouse_ids = ['074743', '074746', '895773', '953828', '953829']

    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in ['latency to escape', 'speed', 'acceleration', 'time in safety zone']:
        df = photometry_habituations.get_signal_metric_dataframe(mtgs, metric)
        all_dfs_pre_post.append(df)

    metrics = ['latency to escape', 'speed', 'acceleration', 'time in safety zone']
    for dfa, dfb, metric in zip(all_dfs, all_dfs_pre_post, metrics):
        joined = dfa.append(dfb)
        g = sns.lmplot(metric, 'ca signal', data=joined, hue=hue, fit_reg=False)
        sns.regplot(x=metric, y='ca signal', data=joined, scatter=False, ax=g.axes[0, 0], color='k')
        g.savefig('/home/slenzi/figures/photometry_contrasts/pre_post_24hr_group_on_contrast_ca_curve_{}_hue_{}.eps'.format(metric, hue.replace(' ', '_')), format='eps')
        g.savefig('/home/slenzi/figures/photometry_contrasts/pre_post_24hr_group_on_contrast_ca_curve_{}_hue_{}.png'.format(metric, hue.replace(' ', '_')), format='png')


def plot_habituation_trialwise_with_lowess_fit():
    pre_test_24hr = ['074743', '074746', '895773', '953828', '953829']
    pre_test_sameday = ['898989', '898990']
    mtgs_24 = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in (pre_test_24hr)]
    mtgs_imm = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in (pre_test_sameday)]

    df =photometry_habituations.habituation_df([mtgs_24, mtgs_imm], ['24 hr', 'same day'])
    plt.figure()
    a = sns.lmplot('trial number', 'ca signal', hue='test type', data=df[df['group label']=='24 hr'], lowess=True)
    b = sns.lmplot('trial number', 'ca signal', hue='test type', data=df[df['group label']=='same day'], lowess=True)
    c = sns.lmplot('trial number', 'ca signal', hue='group label', data=df[df['test type'] == 'habituation'], lowess=True)
