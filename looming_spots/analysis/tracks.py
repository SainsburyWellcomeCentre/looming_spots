import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from looming_spots.db import constants
from looming_spots.db.constants import STIMULUS_ONSETS, CLASSIFICATION_WINDOW_START, \
    CLASSIFICATION_WINDOW_END, CLASSIFICATION_SPEED, SPEED_THRESHOLD, CLASSIFICATION_LATENCY, FRAME_RATE

from looming_spots.preprocess import photodiode


def load_raw_track(loom_folder, name='tracks.csv'):
    track_path = os.path.join(loom_folder, name)
    df = pd.read_csv(track_path, sep='\t')
    x_pos = np.array(df['x_position'])
    y_pos = np.array(df['y_position'])
    return x_pos, y_pos


def normalise_x_track(x_track, context):

    left_wall_pixel = constants.context_params[context].left
    right_wall_pixel = constants.context_params[context].right

    arena_length = right_wall_pixel - left_wall_pixel
    normalised_track = (x_track - left_wall_pixel) / arena_length

    if constants.context_params[context].flip:
        return 1 - normalised_track

    return normalised_track


def load_normalised_track(loom_folder, context):
    x_track, _ = load_raw_track(loom_folder)
    norm_x = normalise_x_track(x_track, context=context)
    return norm_x


def load_normalised_speeds(loom_folder, context):
    x_track = load_normalised_track(loom_folder, context)
    norm_speeds = np.diff(x_track)
    return norm_speeds


def normalised_home_front(context):
    house_front_raw = constants.context_params[context].house_front
    house_front_normalised = normalise_x_track(house_front_raw, context)
    print(house_front_normalised)
    return house_front_normalised


def classify_flee(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    speed = np.diff(track)

    if fast_enough(speed) and reaches_home(track, context):

        return True

    print('fast enough: {}, reaches home: {}'.format(fast_enough(speed), reaches_home(track, context)))
    return False


def time_spent_hiding(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    stimulus_relevant_track = track[CLASSIFICATION_WINDOW_START:]

    home_front = normalised_home_front(context)
    safety_zone_border_crossings = np.where(np.diff(stimulus_relevant_track < home_front))

    if len(safety_zone_border_crossings[0]) == 0:  # never runs away
        return 0
    elif len(safety_zone_border_crossings[0]) == 1:  # never comes back out
        print('this mouse never leaves {}'.format(loom_folder))
        print(safety_zone_border_crossings)
        return int(len(stimulus_relevant_track) - int(safety_zone_border_crossings[0]))/FRAME_RATE
    else:
        return int(safety_zone_border_crossings[0][1])/FRAME_RATE


def fast_enough(speed):
    return any([x < CLASSIFICATION_SPEED for x in speed[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]])


def reaches_home(track, context):
    house_front = normalised_home_front(context)
    return any([x < house_front for x in track[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]])


def retreats_rapidly_at_onset(track):
    latency, _ = estimate_latency(track)
    if latency < CLASSIFICATION_LATENCY:
        return True


def estimate_latency(track, start=CLASSIFICATION_WINDOW_START, end=CLASSIFICATION_WINDOW_END, threshold=SPEED_THRESHOLD):
    speeds = np.diff(track)
    for i, speed in enumerate(speeds[start:end]):
        if speed < threshold:
            return start + i, track[start+i]
    return np.nan


def get_flee_duration(loom_folder, context):
    track = load_normalised_track(loom_folder, context)
    house_front = normalised_home_front(context)

    for i, x in enumerate(track[STIMULUS_ONSETS[0]:]):
        if x < house_front:
            return i
    return np.nan


def get_mouse_position_at_loom_onset(loom_folder):
    x, y = load_raw_track(loom_folder)
    x_at_loom_onset, y_at_loom_onset = x[CLASSIFICATION_WINDOW_START], y[CLASSIFICATION_WINDOW_START]
    return x_at_loom_onset, y_at_loom_onset


def get_track_corrections(session_folder):
    manual_looms_mtd = photodiode.get_manual_looms_from_metadata(session_folder)
    manual_looms_raw = photodiode.get_manual_looms_raw(session_folder)
    return manual_looms_mtd - manual_looms_raw


def get_peak_speed_and_latency(loom_folder, context):
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


def plot_track(loom_folder, context, color, zorder=0, smooth=True, alpha=1, label=None):
    track = load_normalised_track(loom_folder, context)
    if smooth:
        plt.plot(gaussian_filter(track, 3), color=color, zorder=zorder, alpha=alpha, label=label)
    else:
        plt.plot(track, color=color, zorder=zorder, alpha=alpha, label=label)

