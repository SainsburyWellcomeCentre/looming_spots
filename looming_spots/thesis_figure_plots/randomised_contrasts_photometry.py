from looming_spots.analysis.photometry_habituations import get_signal_metric_dataframe_variable_contrasts
from looming_spots.db import loom_trial_group
from looming_spots.thesis_figure_plots import photometry_example_traces
from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns

contrast_curve_mids = ['898992', '916063', '921000', '907822']


def plot_first_loom_raw_signal():
    df = photometry_example_traces.get_first_loom_response_by_contrast()

    sns.set(style='whitegrid')

    pal = sns.cubehelix_palette(7)[::-1]

    # Plot the responses for different events and regions
    g=sns.lineplot(x="timepoint", y="signal", hue="contrast",
                 data=df, palette=pal, linewidth=2)
    r1 = patches.Rectangle((0, 0.06), 14, height=0.005, color='k')
    g.add_patch(r1)
    plt.ylim([-0.005, 0.07])
    plt.xlim([-5, 60])
    return g


def plot_scatter_by_contrast():
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in contrast_curve_mids]
    all_plots = []
    metrics = ['speed', 'acceleration', 'latency to escape', 'time in safety zone', 'time to reach safety']
    for metric in metrics:
        df = get_signal_metric_dataframe_variable_contrasts(mtgs, metric)
        g = sns.lmplot(metric, 'ca signal', data=df, hue='contrast', fit_reg=False,
                       palette=sns.cubehelix_palette(7)[::-1])
        sns.lmplot(metric, 'ca signal', data=df, scatter=False, lowess=True,
                   palette=sns.cubehelix_palette(7)[::-1])
        all_plots.append(g)
    return all_plots
