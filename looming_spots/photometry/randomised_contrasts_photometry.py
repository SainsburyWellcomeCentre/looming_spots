import matplotlib as mpl
import matplotlib.colors
from looming_spots.constants import LOOM_ONSETS, FRAME_RATE
from matplotlib import cm

import pandas as pd
from looming_spots.db import loom_trial_group
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


snl_contrast_curve_mids = ["898992", "916063", "921000", "907822"]


def plot_first_loom_raw_signal(mouse_ids, df=None, trace_type='raw'):
    if df is None:
        if trace_type == 'raw':
            df = get_average_deltaf_trace_first_loom_by_contrast(mouse_ids)
        elif trace_type == 'integral':
            df = get_average_deltaf_integral_trace_first_loom_by_contrast(mouse_ids)

    sns.set(style="whitegrid")

    norm = mpl.colors.Normalize(vmin=0.08, vmax=0.1507)
    cmap = cm.Greys_r

    for contrast in sorted(df['contrast'].unique())[::-1]:
        line_df = df[df['contrast'] == contrast]
        if contrast == 0:
            c = norm(contrast + 0.05)
        else:
            c = norm(contrast)

        plt.plot(line_df['timepoint']/FRAME_RATE, line_df['signal'], color=cm.Greys_r(c))

    plt.ylim([-0.01, 1.1])


def get_average_deltaf_trace_first_loom_by_contrast(
    contrast_curve_mids=snl_contrast_curve_mids, n_samples=150, start=200
):
    mtgs = [
        loom_trial_group.MouseLoomTrialGroup(mid, photometry=True)
        for mid in contrast_curve_mids
    ]
    df_all = pd.DataFrame()
    for contrast in np.unique(mtgs[0].contrasts()):
        contrast_response_dict = {}
        print(contrast)
        avg_df = []
        pooled_trials_at_contrast = get_first_n_trials_of_contrast(mtgs, contrast, 4)

        for t in pooled_trials_at_contrast:
            avg_df.append(t.delta_f()[start : start + n_samples])

        avg_response_at_contrast = np.mean(avg_df, axis=0)
        contrast_response_dict.setdefault("signal", avg_response_at_contrast)
        contrast_response_dict.setdefault(
            "contrast", [contrast] * len(avg_response_at_contrast)
        )
        contrast_response_dict.setdefault(
            "timepoint", np.arange(len(avg_response_at_contrast))
        )
        df = pd.DataFrame.from_dict(contrast_response_dict)
        df_all = df_all.append(df)
    return df_all

def get_average_deltaf_integral_trace_first_loom_by_contrast(
    contrast_curve_mids=snl_contrast_curve_mids, n_samples=150, start=200
):
    mtgs = [
        loom_trial_group.MouseLoomTrialGroup(mid, photometry=True)
        for mid in contrast_curve_mids
    ]
    df_all = pd.DataFrame()
    for contrast in np.unique(mtgs[0].contrasts()):
        contrast_response_dict = {}
        print(contrast)
        avg_df = []
        pooled_trials_at_contrast = get_first_n_trials_of_contrast(mtgs, contrast, 4)

        for t in pooled_trials_at_contrast:
            avg_df.append(t.integral_downsampled()[start : start + n_samples])

        avg_response_at_contrast = np.mean(avg_df, axis=0)
        contrast_response_dict.setdefault("signal", avg_response_at_contrast)
        contrast_response_dict.setdefault(
            "contrast", [contrast] * len(avg_response_at_contrast)
        )
        contrast_response_dict.setdefault(
            "timepoint", np.arange(len(avg_response_at_contrast))
        )
        df = pd.DataFrame.from_dict(contrast_response_dict)
        df_all = df_all.append(df)
    return df_all


def load_snl_photometry_data():

    df = pd.DataFrame()
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid, photometry=True) for mid in snl_contrast_curve_mids]

    for mtg in mtgs:
        df = df.append(mtg.to_df('snl_photometry_variable_contrast', n_trials=18), ignore_index=True)
    return df


def plot_scatter_by_contrast(df=None):
    if df is None:
        df = load_snl_photometry_data()

    all_plots = []
    metrics = [
        "peak_speed",
    ]
    for metric in metrics:
        norm = mpl.colors.Normalize(vmin=0.01, vmax=0.2)
        cmap = cm.Greys_r
        c = []
        for x in df['contrast']:
            if x == 0:
                c.append(norm(x + 0.09007))
            else:
                c.append(norm(x))

        plt.scatter(df[metric], df['normalised_delta_f_0.5s'], c=c, cmap=cmap)
        plt.ylim([-0.01, 1.1])
        plt.xlim([-0.2, 100])

    # df = df.sort_values(by="contrast")
    # df["contrast"] = df["contrast"].astype(str)
    # sns.lmplot(
    #     "contrast",
    #     "normalised_delta_f_0.5s",
    #     data=df,
    #     hue="is_flee",
    #     fit_reg=False,
    # )
    #
    # plt.figure()
    # n_bins = 10
    # df_escape = df[df["is_flee"]]
    # df_no_escape = df[~df["is_flee"]]
    # sns.distplot(
    #     df_no_escape["normalised_delta_f_0.5s"],
    #     bins=int(n_bins),
    #     norm_hist=True,
    #     hist_kws={"linewidth": 0},
    # )
    # sns.distplot(
    #     df_escape["normalised_delta_f_0.5s"],
    #     bins=int(n_bins),
    #     norm_hist=True,
    #     hist_kws={"linewidth": 0},
    # )


def get_first_n_trials_of_contrast(mtgs, contrast, n_trials_to_take):
    pooled_trials_at_contrast = []

    for mtg in mtgs:
        trials_at_contrast = [
            t for t in mtg.all_trials[:18] if t.contrast == contrast
        ]
        pooled_trials_at_contrast.extend(trials_at_contrast[:n_trials_to_take])

    return pooled_trials_at_contrast
