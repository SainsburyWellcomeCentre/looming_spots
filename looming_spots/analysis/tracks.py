import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from looming_spots.db import constants
from looming_spots.db.constants import STIMULUS_ONSETS, NORM_FRONT_OF_HOUSE_A, NORM_FRONT_OF_HOUSE_A9, \
    NORM_FRONT_OF_HOUSE_B, FRAME_RATE, CLASSIFICATION_WINDOW_START, CLASSIFICATION_WINDOW_END, CLASSIFICATION_SPEED, \
    SPEED_THRESHOLD, CLASSIFICATION_LATENCY

from looming_spots.preprocess import photodiode
from zarchive.track.retrack_variables import convert_tracks_from_dat

BOX_BOUNDARIES = {
                  'A':     (143, 613),
                  'B':     (39, 600),
                  'split': (30, 550),
                  'A9':    (32, 618),
                  'C':     (0, 615)
                   }

HOME_FRONTS = {
               'A': NORM_FRONT_OF_HOUSE_A,
               'B': NORM_FRONT_OF_HOUSE_B,
               'A9': NORM_FRONT_OF_HOUSE_A9,
               }


def load_raw_track(loom_folder, name='tracks.csv'):
    track_path = os.path.join(loom_folder, name)
    if not os.path.isfile(track_path):  # TODO: remove this
        convert_tracks_from_dat(loom_folder)
    df = pd.read_csv(track_path, sep='\t')
    x_pos = np.array(df['x_position'])  # FIXME: pyper saving issue?
    y_pos = np.array(df['y_position'])
    return x_pos, y_pos


def load_normalised_speeds(loom_folder, context):
    x_track = load_normalised_track(loom_folder, context)
    norm_speeds = np.diff(x_track)
    return norm_speeds


def load_normalised_track(loom_folder, context):
    x_track, _ = load_raw_track(loom_folder)
    norm_x = normalise_track(x_track, context=context)
    return norm_x


def normalised_home_front(context):
    house_front_raw = constants.context_params[context].house_front
    house_front_normalised = normalise_track(house_front_raw, context)
    print(house_front_normalised)
    return house_front_normalised


def normalise_track(x_track, context, image_shape=(480, 640)):

    left_wall_pixel = constants.context_params[context].left
    right_wall_pixel = constants.context_params[context].right

    arena_length = right_wall_pixel - left_wall_pixel
    normalised_track = (x_track - left_wall_pixel) / arena_length

    if constants.context_params[context].flip:
        return 1 - normalised_track

    return normalised_track


def classify_flee(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    print(context)
    speed = np.diff(track)

    house_front = normalised_home_front(context)

    fast_enough = any([x < CLASSIFICATION_SPEED for x in speed[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]])  # -0.031
    reaches_home = any([x < house_front for x in track[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]])
    print('fast enough: {}, reaches home: {}'.format(fast_enough, reaches_home))
    if fast_enough and reaches_home:
        return True

    return False


def fast_enough(track, start=CLASSIFICATION_WINDOW_START,
                end=CLASSIFICATION_WINDOW_END, threshold=CLASSIFICATION_SPEED):
    track = gaussian_filter(track, 3)
    speed = np.diff(track)
    return any([x < threshold for x in speed[start:end]])


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


def any_flee(s):
    return True in s.trials_results


def n_flees_all_sessions(sessions):
    n_flees_total = [s.n_flees for s in sessions]
    return n_flees_total


def n_non_flees_all_sessions(sessions):
    n_non_flees_total = [s.n_non_flees for s in sessions]
    return n_non_flees_total


def get_mean_track_and_speed(sessions):
    distances, tracks, _ = get_tracks_and_speeds(sessions)
    avg_track = np.mean(tracks, axis=0)
    avg_speed = np.mean(distances, axis=0)
    return avg_track, avg_speed


def get_tracks_and_speeds(sessions, smooth=False):
    tracks = get_all_tracks(sessions, smooth=smooth, corrected=True)
    speeds = np.diff(tracks, axis=1)

    flees = get_all_classifications(sessions)
    return tracks, speeds, flees


def get_tracks_and_speeds_nth(sessions, smooth=False, n=0):
    tracks = get_nth_track(sessions, smooth=smooth, corrected=True, n=n)
    speeds = np.diff(tracks, axis=1)
    flees = get_all_classifications_nth(sessions, n=n)
    return tracks, speeds, flees


def get_all_tracks(sessions, smooth=False, corrected=False):
    tracks = []
    for s in sessions:
        if corrected:
            track_corrections = get_track_corrections(s.path)

        for i, loom_folder in enumerate(s.loom_paths):
            track = load_normalised_track(loom_folder, s.context)
            if len(track) == 601:
                track = track[:-1]
            print('{} track shape {}'.format(s.path, track.shape))

            if corrected:
                track = np.roll(track, track_corrections[i])
                track[0:track_corrections[i]] = np.nan
            if smooth:
                track = gaussian_filter(track, 3)
            tracks.append(track)

    return np.array(tracks)


def get_nth_track(sessions, smooth=False, corrected=False, n=0):
    tracks = []
    for s in sessions:
        if corrected:
            track_corrections = get_track_corrections(s.path)

        track = load_normalised_track(s.loom_paths[n], s.context)

        if len(track) == 601:
            track = track[:-1]

        print('{} track shape {}'.format(s.path, track.shape))

        if corrected:
            track = np.roll(track, track_corrections[n])
            track[0:track_corrections[n]] = np.nan

        if smooth:
            track = gaussian_filter(track, 3)
        tracks.append(track)
    return tracks


def get_all_speeds(sessions, smooth=False):
    speeds = []
    for s in sessions:
        for loom_folder in s.loom_paths:
            track = load_normalised_track(loom_folder, s.context)
            if smooth:
                track = gaussian_filter(track, 3)
            speed = np.diff(track)
            speeds.append(speed)
    return np.array(speeds)


def get_all_accelerations(sessions, smooth=False):
    accelerations = []
    for s in sessions:
        for loom_folder in s.loom_paths:
            track = load_normalised_track(loom_folder, s.context)
            if smooth:
                track = gaussian_filter(track, 3)
            speed = np.diff(track)
            acceleration = np.diff(speed)
            accelerations.append(acceleration)
    return np.array(accelerations)


def get_all_peak_speeds(sessions):
    speeds, peak_speeds_args = [], []
    for s in sessions:
        session_speeds, session_arg_speeds = get_speeds_all_trials(s)
        speeds.extend(session_speeds)
        peak_speeds_args.extend(session_arg_speeds)
    return speeds, peak_speeds_args


def get_all_peak_speeds_nth(sessions, n=0):
    speeds, peak_speeds_args = [], []
    for s in sessions:
        session_speeds, session_arg_speeds = get_speeds_nth_trial(s, n)
        speeds.extend([session_speeds])
        peak_speeds_args.extend([session_arg_speeds])
    return speeds, peak_speeds_args


def get_speeds_all_trials(s):
    peak_speeds, peak_speeds_args = [], []
    for loom_folder in s.loom_paths:
        peak_speed, arg_peak_speed = get_peak_speed_and_latency(loom_folder, s.context)
        peak_speeds.append(peak_speed)
        peak_speeds_args.append(arg_peak_speed)
    return peak_speeds, peak_speeds_args


def get_speeds_nth_trial(s, n):
    peak_speed, arg_peak_speed = get_peak_speed_and_latency(s.loom_paths[n], s.context)
    return peak_speed, arg_peak_speed


def get_all_classifications(sessions):
    flee_outcomes = [s.trials_results for s in sessions]
    return np.array(flee_outcomes)


def get_all_classifications_nth(sessions, n=0):
    flee_outcomes = [s.trials_results[n] for s in sessions]
    return np.array(flee_outcomes)


def get_flee_durations(sessions):
    durations = []
    for s in sessions:
        for loom_folder in s.loom_paths:
            duration = get_flee_duration(loom_folder, s.context)
            durations.append(duration)
    return np.array(durations)


def get_flee_duration(loom_folder, context):
    track = load_normalised_track(loom_folder, context)
    home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
    for i, x in enumerate(track[STIMULUS_ONSETS[0]:]):
        if x < home_front:
            return i
    return np.nan


def get_experiment_hour_all_sessions(sessions):
    hours = []
    for s in sessions:
        for _ in s.loom_paths:
            hours.append(s.hour)
    return hours


def get_flee_classification_colors_all_sessions(sessions):
    colors = []
    for s in sessions:
        for loom_folder in s.loom_paths:
            color = 'r' if classify_flee(loom_folder, s.context) else 'k'
            colors.extend(color)
    return colors


def get_mouse_position_at_loom_onset(loom_folder):
    x, y = load_raw_track(loom_folder)
    x_at_loom_onset, y_at_loom_onset = x[CLASSIFICATION_WINDOW_START], y[CLASSIFICATION_WINDOW_START]
    return x_at_loom_onset, y_at_loom_onset


def get_loom_position_all_sessions(sessions):
    all_xs, all_ys = [], []
    for s in sessions:
        xs, ys = get_loom_positions(s.path)
        all_xs.extend(xs)
        all_ys.extend(ys)
    return all_xs, all_ys


def get_loom_positions(s):
    xs, ys = [], []
    for loom_folder in s.loom_paths:
        x_at_loom_onset, y_at_loom_onset = get_mouse_position_at_loom_onset(loom_folder)
        xs.extend([x_at_loom_onset])
        ys.extend([y_at_loom_onset])
    return xs, ys


def get_avg_speed_and_latency(s):
    speeds, latencies = get_all_peak_speeds([s])
    return np.mean(speeds), np.mean(latencies)


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
    peak_speed = min(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    arg_peak = np.argmin(distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END])
    return -peak_speed, arg_peak + CLASSIFICATION_WINDOW_START


def plot_track_and_loom_position_all_sessions(sessions):
    for s in sessions:
        plot_session_tracks_and_loom_positions(s)


def plot_avg_track_and_std(sessions, color='b', label='', filt=False):
    track, speeds, flees = get_tracks_and_speeds(sessions)
    if filt:
        track = np.array(track)[np.array(flees) == 1]
    avg_track = np.nanmean(track, axis=0)
    std_track = np.nanstd(track, axis=0)
    plt.plot(avg_track, color=color, linewidth=3, label=label, zorder=0)
    plt.fill_between(np.arange(0, 399), avg_track + std_track, avg_track - std_track, alpha=0.5, color=color, zorder=0)
    plt.plot(avg_track + std_track, color='k', alpha=0.4, linewidth=0.25)
    plt.plot(avg_track - std_track, color='k', alpha=0.4, linewidth=0.25)


def plot_session_avg(s, color='b', label=''):
    track, speeds, _ = get_tracks_and_speeds([s])
    avg_track = np.nanmean(track, axis=0)
    plt.plot(avg_track, linewidth=3, label=label, zorder=0, color=color, alpha=0.7)
    #std_track = np.nanstd(track, axis=0)
    #plt.fill_between(np.arange(0, 399), avg_track + std_track, avg_track - std_track, alpha=0.5, color=color, zorder=0)
    #plt.plot(avg_track + std_track, color='k', alpha=0.4, linewidth=0.25)
    #plt.plot(avg_track - std_track, color='k', alpha=0.4, linewidth=0.25)


def plot_each_mouse(sessions, color=None, label=None):
    for i, s in enumerate(sessions):
        if color is not None:
            plot_session_avg(s, label=label, color=color)
        else:
            plot_session_avg(s, label='mouse {}'.format(str(i)))


def plot_avg_speed_latency_time_of_day(sessions, color=None, label=None):
    flee_times = []
    avg_speeds = []
    avg_latencies = []
    for s in sessions:
        avg_speed, avg_latency = get_avg_speed_and_latency(s)
        flee_times.extend([s.dt.hour+s.dt.minute/60])
        avg_speeds.append(avg_speed)
        avg_latencies.append(avg_latency)

    plt.subplot(211)
    plt.scatter(flee_times, avg_speeds, color=color, label=label)
    plt.xlabel('time of day (hr)')
    plt.ylabel('peak speed (a.u.)')

    plt.subplot(212)
    plt.scatter(flee_times, avg_latencies, color=color, label=label)
    plt.xlabel('time of day (hr)')
    plt.ylabel('time of peak speed (frame number)')


def plot_all_sessions(sessions, smooth=False, alpha=1, manual_color=None, label=''):
    for s in sessions:
        plot_flees(s, smooth=smooth, alpha=alpha, label=label, manual_color=manual_color)
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_all_sessions_mouse_colors(sessions, smooth=False, alpha=1):
    color_space = np.linspace(0, 1, len(sessions))
    for i, s in enumerate(sessions):
        color = plt.cm.Spectral(color_space[i])
        plot_flees(s, smooth=smooth, alpha=alpha, suppress_non_flees=True, label=s.mouse_name, manual_color=color)
    plt.legend()
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_flees(session, smooth=False, alpha=1, label=None, suppress_non_flees=False, manual_color=None):
    for loom_folder in session.loom_paths:
        track_is_flee = classify_flee(loom_folder, session.context)
        if track_is_flee or (not suppress_non_flees):
            zorder = 1 if track_is_flee else 0
            if manual_color is not None:
                color = manual_color
            else:
                color = 'r' if track_is_flee else 'k'

            plot_track(loom_folder, session.context, color=color, zorder=zorder,
                       smooth=smooth, alpha=alpha, label=label)


def plot_flees_corrected(session, alpha=1, label=None):
    track_corrections = get_track_corrections(session.path)
    print('track corrections: {}'.format(track_corrections))
    for i, loom_folder in enumerate(session.loom_paths):
        track_is_flee = classify_flee(loom_folder, session.context)
        color = 'r' if track_is_flee else 'k'
        track = load_normalised_track(loom_folder, session.context)
        corrected_track = np.roll(track, track_corrections[i])
        corrected_track[0:track_corrections[i]] = np.nan
        plt.plot(corrected_track, color=color, alpha=alpha, label=label)


def get_track_corrections(directory):
    manual_looms_mtd = photodiode.get_manual_looms_from_metadata(directory)
    manual_looms_raw = photodiode.get_manual_looms_raw(directory)
    return manual_looms_mtd - manual_looms_raw


def plot_speeds_and_latencies(sessions, ax, colors=None, label=''):
    speeds, arg_speeds = get_all_peak_speeds(sessions)
    ax.scatter(arg_speeds, speeds, c=colors, edgecolor='None', s=45, label=label)
    plt.ylim([-0.01, 0.12])
    plt.xlabel('frame number')
    plt.ylabel('peak speed')


def plot_track(loom_folder, context, color, zorder=0, smooth=True, alpha=1, label=None):
    track = load_normalised_track(loom_folder, context)
    if smooth:
        plt.plot(gaussian_filter(track, 3), color=color, zorder=zorder, alpha=alpha, label=label)
    else:
        plt.plot(track, color=color, zorder=zorder, alpha=alpha, label=label)


def plot_durations(sessions, ax, color='r', label='', highlight_flees=False):
    speeds, _ = get_all_peak_speeds(sessions)
    colors = get_flee_classification_colors_all_sessions(sessions)
    durations_in_frames = get_flee_durations(sessions)

    if len(durations_in_frames) == 0:
        return
    durations_in_seconds = durations_in_frames / FRAME_RATE

    if highlight_flees:
        ax.scatter(durations_in_seconds, speeds, c=colors, edgecolor='None', s=45, label=label)
    else:
        ax.scatter(durations_in_seconds, speeds, color=color, edgecolor='None', s=45, label=label)
    plt.xlabel('flee duration in seconds')
    plt.ylabel('speed of flee a.u.')
    plt.xlim([0, 7])
    plt.ylim([0, 0.1])


def plot_session_tracks_and_loom_positions(session, color='r'):
    plt.imshow(session.reference_frame, cmap='Greys', vmin=0, vmax=110, aspect='auto')
    for loom_folder in session.loom_paths:
        x, y = load_raw_track(loom_folder)
        plt.plot(x[CLASSIFICATION_WINDOW_START], y[CLASSIFICATION_WINDOW_START],
                 'o', markersize=8, color=color, zorder=1000, alpha=0.7)
        plt.plot(x[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END],
                 y[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END], color=color)
