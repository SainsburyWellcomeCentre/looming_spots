import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt, patches as patches

from matplotlib.collections import LineCollection

from looming_spots.analysis.tracks import STIMULUS_ONSETS, NORM_FRONT_OF_HOUSE_B, NORM_FRONT_OF_HOUSE_A, NORM_FRONT_OF_HOUSE_A9


def plot_looms(fig):
    for ax in fig.axes:
        for loom in [create_loom_patch(stim) for stim in STIMULUS_ONSETS]:
            ax.add_patch(loom)
    return fig


def plot_home(fig, context):
    for ax in fig.axes:
        plt.sca(ax)
        #home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
        home_front = NORM_FRONT_OF_HOUSE_A9
        plt.axhline(home_front, 0, 400, ls='--')


def plot_looms_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    looms = [create_loom_patch(stim) for stim in STIMULUS_ONSETS]
    for loom in looms:
        ax.add_patch(loom)


def create_loom_patch(start):
    return patches.Rectangle((start, -0.2), 14, 1.3, alpha=0.1, color='k')


def plot_line_with_color_variable(x, y, color_variable_array, start=None, normalising_factor=None):

    """
    modified from: https://stackoverflow.com/questions/10252412/matplotlib-varying-color-of-line-to-capture-natural-time-parameterization-in-da
    :param x:
    :param y:
    :param color_variable_array:
    :return:
    """

    points = np.array([x, y]).transpose().reshape(-1, 1, 2)
    color_variable_array = np.array([val for (p, val) in zip(points, color_variable_array) if not np.isnan(p).any()])
    points = np.array([p for p in points if not np.isnan(p).any()])

    if start is None:
        norm_factor = 1
    else:
        norm_factor = max(color_variable_array)
        points = points[start:]
        color_variable_array = color_variable_array[start:]

    points = np.array([p for (p, val) in zip(points, color_variable_array) if not np.isnan(val)])
    color_variable_array = np.array([val for val in color_variable_array if not np.isnan(val)])

    if normalising_factor is not None:
        norm_factor = normalising_factor

    color_variable_array /= norm_factor

    segs = np.concatenate([points[:-1], points[1:]], axis=1)


    lc = LineCollection(segs, cmap=plt.get_cmap('inferno'), norm=matplotlib.colors.Normalize(vmin=0, vmax=0.8))
    lc.set_array(color_variable_array)

    plt.gca().add_collection(lc)

    plt.plot(color_variable_array)
