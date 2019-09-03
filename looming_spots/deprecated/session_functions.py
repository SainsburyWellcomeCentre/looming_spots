import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

import looming_spots.analysis.escape_classifiers
import looming_spots.preprocess.normalisation
from looming_spots.analysis import tracks
from looming_spots.db.constants import FRAME_RATE


def get_speeds_all_trials(s):
    peak_speeds, peak_speeds_args = [], []
    for loom_folder in s.loom_paths:
        peak_speed, arg_peak_speed = tracks.get_peak_speed_and_latency(loom_folder, s.context)
        peak_speeds.append(peak_speed)
        peak_speeds_args.append(arg_peak_speed)
    return peak_speeds, peak_speeds_args


def get_speeds_nth_trial(s, n):
    peak_speed, arg_peak_speed = tracks.get_peak_speed_and_latency(s.loom_paths[n], s.context)
    return peak_speed, arg_peak_speed


def get_loom_positions(s):
    xs, ys = [], []
    for loom_folder in s.loom_paths:
        x_at_loom_onset, y_at_loom_onset = tracks.get_mouse_position_at_loom_onset(loom_folder)
        xs.extend([x_at_loom_onset])
        ys.extend([y_at_loom_onset])
    return xs, ys


def get_avg_speed_and_latency(s):
    speeds, latencies = get_all_peak_speeds([s])
    return np.mean(speeds), np.mean(latencies)


def plot_session_avg(s, color='b', label=''):
    track, speeds, _ = get_tracks_and_speeds([s])
    avg_track = np.nanmean(track, axis=0)
    plt.plot(avg_track, linewidth=3, label=label, zorder=0, color=color, alpha=0.7)


def plot_flees_corrected(session, alpha=1, label=None):
    track_corrections = tracks.get_track_corrections(session.path)
    print('track corrections: {}'.format(track_corrections))
    for i, loom_folder in enumerate(session.loom_paths):
        track_is_flee = looming_spots.analysis.escape_classifiers.classify_flee(loom_folder, session.context)
        color = 'r' if track_is_flee else 'k'
        track = looming_spots.preprocess.normalisation.load_normalised_track(loom_folder, session.context)
        corrected_track = np.roll(track, track_corrections[i])
        corrected_track[0:track_corrections[i]] = np.nan
        plt.plot(corrected_track, color=color, alpha=alpha, label=label)


def plot_flees(session, smooth=False, alpha=1, label=None, suppress_non_flees=False, manual_color=None):
    for loom_folder in session.loom_paths:
        track_is_flee = looming_spots.analysis.escape_classifiers.classify_flee(loom_folder, session.context)
        if track_is_flee or (not suppress_non_flees):
            zorder = 1 if track_is_flee else 0
            if manual_color is not None:
                color = manual_color
            else:
                color = 'r' if track_is_flee else 'k'

                tracks.plot_track(loom_folder, session.context, color=color, zorder=zorder,
                       smooth=smooth, alpha=alpha, label=label)


def get_x_length(ax):
    line = ax.lines[0]
    xdata = line.get_xdata()
    return len(xdata)


def n_flees_all_sessions(sessions):
    n_flees_total = [s.n_flees for s in sessions if s.get_trials('test')]
    return n_flees_total


def n_non_flees_all_sessions(sessions):
    n_non_flees_total = [s.n_non_flees for s in sessions if s.get_trials('test')]
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
            track_corrections = tracks.get_track_corrections(s.path)

        for i, loom_folder in enumerate(s.loom_paths):
            track = tracks.load_normalised_track(loom_folder, s.context)
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
            track_corrections = tracks.get_track_corrections(s.path)

        track = tracks.load_normalised_track(s.loom_paths[n], s.context)

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
            track = looming_spots.preprocess.normalisation.load_normalised_track(loom_folder, s.context)
            if smooth:
                track = gaussian_filter(track, 3)
            speed = np.diff(track)
            speeds.append(speed)
    return np.array(speeds)


def get_all_accelerations(sessions, smooth=False):
    accelerations = []
    for s in sessions:
        for loom_folder in s.loom_paths:
            track = looming_spots.preprocess.normalisation.load_normalised_track(loom_folder, s.context)
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
            duration = looming_spots.analysis.escape_classifiers.get_flee_duration(loom_folder, s.context)
            durations.append(duration)
    return np.array(durations)


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
            color = 'r' if looming_spots.analysis.escape_classifiers.classify_flee(loom_folder, s.context) else 'k'
            colors.extend(color)
    return colors


def get_loom_position_all_sessions(sessions):
    all_xs, all_ys = [], []
    for s in sessions:
        xs, ys = get_loom_positions(s.path)
        all_xs.extend(xs)
        all_ys.extend(ys)
    return all_xs, all_ys


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
        plot_flees(s, smooth=smooth, alpha=alpha, suppress_non_flees=True, label=s.mouse_id, manual_color=color)
    plt.legend()
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_speeds_and_latencies(sessions, ax, colors=None, label=''):
    speeds, arg_speeds = get_all_peak_speeds(sessions)
    ax.scatter(arg_speeds, speeds, c=colors, edgecolor='None', s=45, label=label)
    plt.ylim([-0.01, 0.12])
    plt.xlabel('frame number')
    plt.ylabel('peak speed')


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
