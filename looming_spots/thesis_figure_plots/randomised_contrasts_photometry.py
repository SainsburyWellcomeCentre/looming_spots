from looming_spots.trial_group_analysis.photometry_habituations import (
    get_signal_metric_dataframe_variable_contrasts,
)
from looming_spots.db import trial_group
from looming_spots.thesis_figure_plots import photometry_example_traces
from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


snl_contrast_curve_mids = ["898992", "916063", "921000", "907822"]


def plot_first_loom_raw_signal():
    df = photometry_example_traces.get_first_loom_response_by_contrast()

    sns.set(style="whitegrid")

    pal = sns.cubehelix_palette(7)[::-1]

    # Plot the responses for different events and regions
    g = sns.lineplot(
        x="timepoint",
        y="signal",
        hue="contrast",
        data=df,
        palette=pal,
        linewidth=2,
    )
    r1 = patches.Rectangle((0, 0.06), 14, height=0.005, color="k")
    g.add_patch(r1)
    plt.ylim([-0.005, 0.07])
    plt.xlim([-5, 60])
    return g


def plot_scatter_by_contrast(mids=snl_contrast_curve_mids):
    mtgs = [
        trial_group.MouseLoomTrialGroup(mid)
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
        g=sns.lmplot(
            metric,
            "ca signal",
            data=df,
            hue="contrast", #contrast
            fit_reg=False,
            palette=sns.cubehelix_palette(7, hue=0)[::-1],
        )

        xlim_upper = df[metric].quantile(0.95)
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, xlim_upper])

        all_plots.append(g)

    df = df.sort_values(by='contrast')
    df['contrast'] = df['contrast'].astype(str)
    sns.lmplot(
        "contrast",
        "ca signal",
        data=df,
        hue='escape',
        fit_reg=False,
    )
    plt.figure()
    n_bins=10
    df_escape = df[df['escape']]
    df_no_escape = df[~df['escape']]
    sns.distplot(df_no_escape['ca signal'], bins=int(n_bins), norm_hist=True, hist_kws={'linewidth': 0})
    sns.distplot(df_escape['ca signal'], bins=int(n_bins), norm_hist=True, hist_kws={'linewidth': 0})
    return all_plots, df, mtgs


def get_trials_of_contrast(trials, contrast):
    return [t for t in trials if t.contrast==contrast]


def waveform_comparison(ctst, mid):

    mtg = trial_group.MouseLoomTrialGroup(mid)

    trials = get_trials_of_contrast(mtg.loom_trials(), ctst)
    test_ctst_trials=get_trials_of_contrast(mtg.loom_trials(), 0)
    avg_waveform=np.nanmean([t.delta_f()[:600] for t in trials], axis=0)
    avg_test_waveform=np.nanmean([t.delta_f()[:600] for t in test_ctst_trials],axis=0)
    return avg_waveform, avg_test_waveform
