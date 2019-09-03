import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from looming_spots.db.constants import CLASSIFICATION_WINDOW_START, \
    CLASSIFICATION_WINDOW_END

from looming_spots.preprocess import photodiode
from looming_spots.preprocess.normalisation import load_normalised_track, load_raw_track


def get_mouse_position_at_loom_onset(loom_folder):
    x, y = load_raw_track(loom_folder)
    x_at_loom_onset, y_at_loom_onset = x[CLASSIFICATION_WINDOW_START], y[CLASSIFICATION_WINDOW_START]
    return x_at_loom_onset, y_at_loom_onset


def get_track_corrections(session_folder):
    manual_looms_mtd = photodiode.get_manual_looms_from_metadata(session_folder)
    manual_looms_raw = photodiode.get_manual_looms_raw(session_folder)
    return manual_looms_mtd - manual_looms_raw


def get_peak_speed_and_latency_deprecated(loom_folder, context):
    """

    :param loom_folder:
    :param context:
    :return peak_speed:
    :return arg_peak: the frame number of the peak speed
    """
    track = load_normalised_track(loom_folder, context)
    filtered_track = gaussian_filter(track, 3)
    distances = np.diff(filtered_track)
    peak_speed = np.nanmin(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    arg_peak = np.argmin(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    return -peak_speed, arg_peak + CLASSIFICATION_WINDOW_START


def get_peak_speed_and_latency(normalised_track):
    """
    :return peak_speed:
    :return arg_peak: the frame number of the peak speed
    """
    filtered_track = gaussian_filter(normalised_track, 3)
    distances = np.diff(filtered_track)
    peak_speed = np.nanmin(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    arg_peak = np.argmin(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    return -peak_speed, arg_peak + CLASSIFICATION_WINDOW_START


def plot_track(loom_folder, context, color, zorder=0, smooth=True, alpha=1, label=None):
    track = load_normalised_track(loom_folder, context)
    if smooth:
        plt.plot(gaussian_filter(track, 3), color=color, zorder=zorder, alpha=alpha, label=label)
    else:
        plt.plot(track, color=color, zorder=zorder, alpha=alpha, label=label)

