import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

STIMULUS_ONSETS = [200, 228, 256, 284, 312]
NORM_FRONT_OF_HOUSE_A = 0.1
NORM_FRONT_OF_HOUSE_B = 0.135
FRAME_RATE = 30


def plot_tracks(directory, fig, color='k'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            track, _ = load_track(loom_folder)
            plt.plot(track, color=color)
    return fig


def plot_track(loom_folder, context, color, zorder=0, smooth=True, alpha=1, label = None):
    track = load_normalised_track(loom_folder, context)
    if smooth:
        plt.plot(gaussian_filter(track, 3), color=color, zorder=zorder, alpha=alpha, label=label)
    else:
        plt.plot(track, color=color, zorder=zorder, alpha=alpha, label=label)


def plot_speed(loom_folder, context, color):
    distances = get_smooth_speed(loom_folder, context)
    plt.plot(distances, color=color)


def load_distances(loom_folder, context):
    x_track, _ = load_track(loom_folder)
    x_track = normalise_track(x_track, context)
    return np.diff(x_track)


# def plot_distances(directory, fig, color='b'):
#     for name in os.listdir(directory):
#         loom_folder = os.path.join(directory, name)
#         if os.path.isdir(loom_folder):
#             distances = load_distances(loom_folder)
#             plt.plot(distances, color=color)
#     return fig


def plot_looms(fig):
    for ax in fig.axes:
        for loom in [loom_patch(stim) for stim in STIMULUS_ONSETS]:
            ax.add_patch(loom)
    return fig


def plot_home(context):
    ax = plt.gca()
    home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
    ax.hlines(home_front, 0, 400, linestyles='dashed')


def plot_looms_ax(ax):
    looms = [loom_patch(stim) for stim in STIMULUS_ONSETS]
    for loom in looms:
        ax.add_patch(loom)


def load_track(loom_folder, name='tracks.csv'):
    track_path = os.path.join(loom_folder, name)
    if not os.path.isfile(track_path):
        convert_tracks_from_dat(loom_folder)
    df = pd.read_csv(track_path, sep='\t')
    x_pos = np.array(df['x_position'])  # FIXME: pyper saving issue?
    y_pos = np.array(df['y_position'])
    return x_pos, y_pos


def load_distances_from_file(path, name='distances.dat'):
    path = os.path.join(path, name)
    print(path)
    df = pd.read_csv(path, sep='\t')
    return np.array(df)


def get_n_flees_all_sessions(session_list):
    n_flees_total = []
    for s in session_list:
        n_flees_session, _ = get_n_flees(s.path, s.context)
        n_flees_total.append(n_flees_session)
    return n_flees_total


def get_n_flees(session_folder, context):
    n_trials = 0
    results = []
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            results.append(classify_flee(loom_folder, context))
            n_trials += 1
    return np.count_nonzero(results), n_trials


def get_flee_rate(session_folder, context):
    flees, trials = get_n_flees(session_folder, context)
    return flees/trials


def loom_patch(start):
    return patches.Rectangle((start, -0.2), 14, 800, alpha=0.1, color='k')


def get_smooth_speed(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    return np.diff(track)


def classify_flee(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    speed = np.diff(track)

    home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
    fast_enough = any([x < -0.027 for x in speed[200:345]])  # -0.031
    reaches_home = any([x < home_front for x in track[200:345]])
    #print('fast enough: {}, reaches home: {}'.format(fast_enough, reaches_home))

    if fast_enough and reaches_home:
        return True


def plot_all_sessions(session_list, smooth=False, alpha=1, color=None, label=''):
    for s in session_list:
        plot_all(s.path, s.context, smooth=smooth, alpha=alpha, color=color, label=label)
    plot_home(s.context)
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_all_sessions_headpost_house(session_list, smooth=False, alpha=1):
    from looming_spots.db import experiment_metadata

    for s in session_list:
        mtd = experiment_metadata.load_metadata(s.path)
        sub_context = mtd['sub_context']
        print(sub_context, type(sub_context))
        if sub_context == 'None':
            color = 'r'
            label = 'with headpost round house'
        elif sub_context == '2':
            color = 'b'
            label = 'no headpost'
        elif sub_context == '3':
            color = 'g'
            label = 'square house'
        plot_all(s.path, s.context, smooth=smooth, alpha=alpha, color=color, label=label)
    plot_home(s.context)
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_all_sessions_mouse_colors(session_list, smooth=False, alpha=1):
    color_space = np.linspace(0, 1, len(session_list))
    for i, s in enumerate(session_list):
        color = plt.cm.Spectral(color_space[i])
        plot_all(s.path, s.context, smooth=smooth, alpha=alpha, color=color, suppress_non_flees=True, label=s.mouse_name)
    plt.legend()
    plot_home(s.context)
    plt.xlabel('frame number')
    plt.ylabel('normalised x position')


def plot_all(directory, context, smooth=False, alpha=1, color=None, suppress_non_flees=False, label=None):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(directory, name)
            if classify_flee(loom_folder, context):
                c = 'r' if color is None else color
                zorder = 10000
                plot_track(loom_folder, context, color=c, zorder=zorder, smooth=smooth, alpha=alpha, label=label)
                #plot_speed(loom_folder, context, color='b')
            elif not suppress_non_flees:
                c = 'k' if color is None else color
                print(directory)
                zorder = 0
                plot_track(loom_folder, context, color=c, zorder=zorder, smooth=smooth, alpha=alpha, label=label)
                #plot_speed(loom_folder, context, color)


def plot_flees(directory, context, color='r'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(directory, name)
            print(loom_folder)
            if classify_flee(loom_folder, context):
                plot_track(loom_folder, context, color=color)
                #plot_speed(loom_folder, context, color='b')


def plot_non_flees(directory, context, color='k'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(directory, name)
            print(loom_folder)
            if not classify_flee(loom_folder, context):
                plot_track(loom_folder, context, color=color)
                plot_speed(loom_folder, context, color='b')


def load_normalised_track(loom_folder, context):
    x_track, _ = load_track(loom_folder)
    return normalise_track(x_track, context=context)


def load_normalised_distances(loom_folder, context):
    norm_speeds = np.diff(load_normalised_track(loom_folder, context))
    return norm_speeds


def get_mean_track_and_speed(session_list):
    distances, tracks, _ = get_tracks_and_speeds(session_list)
    avg_track = np.mean(tracks, axis=0)
    avg_speed = np.mean(distances, axis=0)
    return avg_track, avg_speed


def get_tracks_and_speeds(session_list):
    tracks = []
    distances = []
    flees = []
    for s in session_list:
        for name in os.listdir(s.path):
            loom_folder = os.path.join(s.path, name)
            if os.path.isdir(loom_folder):
                track = load_normalised_track(loom_folder, s.context)
                speeds = load_normalised_distances(loom_folder, s.context)
                tracks.append(track)
                distances.append(speeds)  # FIXME: naming
                if classify_flee(loom_folder, s.context):
                    flees.extend([1])
                else:
                    flees.extend([0])
    return tracks, distances, flees


def get_flee_durations(session_list):
    durations = []
    for s in session_list:
        for name in os.listdir(s.path):
            loom_folder = os.path.join(s.path, name)
            if os.path.isdir(loom_folder):
                #if classify_flee(loom_folder, s.context):
                duration = get_flee_duration(loom_folder, s.context)
                durations.append(duration)
    return np.array(durations)


def get_experiment_hour_all_sessions(session_list):
    hours = []
    for s in session_list:
        for name in os.listdir(s.path):
            loom_folder = os.path.join(s.path, name)
            if os.path.isdir(loom_folder):
                hour = s.dt.hour + s.dt.minute/60
                hours.append(hour)
    return hours


def get_flee_duration(loom_folder, context):
    track = load_normalised_track(loom_folder, context)
    home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
    for i, x in enumerate(track[200:]):
        if x < home_front:
            return i
    return 250


def plot_durations(session_list, ax, color='r', label='', highlight_flees=False):
    speeds, _, colors = get_speeds_all_sessions(session_list)
    durations_in_frames = get_flee_durations(session_list)

    if len(durations_in_frames) == 0:
        return
    durations_in_seconds = durations_in_frames/FRAME_RATE

    if highlight_flees:
        ax.scatter(durations_in_seconds, speeds, c=colors, edgecolor='None', s=45, label=label)
    else:
        ax.scatter(durations_in_seconds, speeds, color=color, edgecolor='None', s=45, label=label)
    plt.xlabel('flee duration in seconds')
    plt.ylabel('speed of flee a.u.')
    plt.xlim([0, 7])
    plt.ylim([0, 0.1])


def normalise_track(x_track, context):
    if context == 'A':
        x_track = 640 - x_track
        return (x_track - 143)/(613-143)  # FIXME: remove magic numbers
    elif context == 'B':
        return (x_track - 39)/(600-39)


def plot_condition_flees(sessions, timepoints, context, threshold=0):
    for s, tp in zip(sessions, timepoints):
        if tp >= threshold:
            plot_flees(s.path, context=context)


def plot_condition_non_flees(sessions, timepoints, context, threshold=0):
    for s, tp in zip(sessions, timepoints):
        if tp >= threshold:
            plot_non_flees(s.path, context=context)


def needs_retrack(loom_directory, region_start=120, region_end=300, n_critical_fails_threshold=10):
    failed_tracking_frames = get_failed_tracking_frames(loom_directory)
    n_critical_fails = get_n_critical(failed_tracking_frames, region_end, region_start)
    if n_critical_fails > n_critical_fails_threshold:
        return True


def get_failed_tracking_frames(loom_directory):
    x, y = load_track(loom_directory)
    diff = np.diff(np.array(x).astype(float))
    same_position_idx = np.where(diff == 0)[0]
    no_track_idx = np.where(x == -1)[0]
    #origin = np.where(x == 0)[0]
    #both = np.concatenate([no_track_idx, same_position_idx, origin])
    return np.unique(same_position_idx)


def get_important_failed_tracking_frames(session, loom_number):
    path = os.path.join(session.path, 'loom{}'.format(loom_number))
    x, y = load_track(path)
    x = np.diff(np.array(x).astype(float))
    return np.where(x == 0)[0]


def get_n_critical(failed_tracking_frames, region_start=120, region_end=300):
    n_critical_fails = np.count_nonzero(np.logical_and(failed_tracking_frames > region_start,  #TODO: refactor
                                                       failed_tracking_frames < region_end))
    return n_critical_fails


def get_critical_frame_ids(failed_tracking_frames, region_start=120, region_end=300):
    return failed_tracking_frames[np.logical_and(failed_tracking_frames > region_start, failed_tracking_frames < region_end)]


def update_track(directory, x, y):
    path = os.path.join(directory, 'data.dat')
    df = pd.read_csv(path, sep='\t')
    df['frame_num'] = x
    df['centre_x'] = y
    print(df)
    df.to_csv(path, sep='\t')


def convert_tracks_from_dat(loom_folder):
    df = pd.read_csv(os.path.join(loom_folder, 'data.dat'), sep='\t')
    x = df['frame_num']
    x.name = 'x_position'
    y = df['centre_x']
    y.name = 'y_position'
    df_tracks = pd.DataFrame(x, index=x.index)
    df_tracks[y.name] = y
    df_tracks.to_csv(os.path.join(loom_folder, 'tracks.csv'), sep='\t', index=False)


def any_flee(session_folder, context):
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(session_folder, name)
            print(loom_folder)
            if classify_flee(loom_folder, context):
                return True


def flee_twice(session_folder, context):
    running = 0
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(session_folder, name)
            print(loom_folder)
            if classify_flee(loom_folder, context):
                running += 1
                if running == 2:
                    return True


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
    peak_speed = min(distances[200:350])  # FIXME: remove magic numbers
    arg_peak = np.argmin(distances[200:350])
    return -peak_speed, arg_peak+200


def get_speeds_all_trials(session_folder, context):
    speeds, locs, colors = [], [], []
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            peak_speed, arg_peak_speed = get_peak_speed_and_latency(loom_folder, context)
            speeds.append(peak_speed)
            locs.append(arg_peak_speed)

            color = 'r' if classify_flee(loom_folder, context) else 'k'
            colors.extend(color)

    return speeds, locs, colors


def get_speeds_all_sessions(session_list):
    speeds, locs, colors = [], [], []
    for s in session_list:
        session_speeds, session_locs, session_colors = get_speeds_all_trials(s.path, s.context)
        speeds.extend(session_speeds)
        locs.extend(session_locs)
        colors.extend(session_colors)
    return speeds, locs, colors


def plot_speeds_and_latencies(session_list, ax, colors=None, label=''):
    speeds, arg_speeds, c = get_speeds_all_sessions(session_list)
    if colors is not None:
        c = colors
    ax.scatter(arg_speeds, speeds, c=c, edgecolor='None', s=45, label=label)
    plt.ylim([-0.01, 0.12])
    plt.xlabel('frame number')
    plt.ylabel('peak speed')


def plot_track_and_loom_position(session_folder, context):
    path = os.path.join(session_folder, 'ref.npy')
    if not os.path.isfile(path):
        path = '/home/slenzi/spine_shares/loomer/data_working_copy/ref.npy'
    img = np.load(path)
    plt.imshow(img, cmap='Greys', vmin=0, vmax=110, aspect='auto')
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            color = 'r'
            x, y = load_track(loom_folder)
            plt.plot(x[200], y[200], 'o', markersize=8, color=color, zorder=1000, alpha=0.7)
            plt.plot(x[200:350], y[200:350], color=color)


def get_loom_positions(session_folder):
    xs, ys = [], []
    for name in os.listdir(session_folder):
        loom_folder = os.path.join(session_folder, name)
        if os.path.isdir(loom_folder):
            x, y = load_track(loom_folder)
            xs.extend([x[STIMULUS_ONSETS[0]]])  # FIXME: ugly code
            ys.extend([y[STIMULUS_ONSETS[0]]])
    return xs, ys


def get_loom_position_all_sessions(session_list):
    all_xs, all_ys = [], []
    for s in session_list:
        xs, ys = get_loom_positions(s.path)
        all_xs.extend(xs)
        all_ys.extend(ys)
    return all_xs, all_ys


def plot_track_and_loom_position_all_sessions(session_list):
    for s in session_list:
        plot_track_and_loom_position(s.path, s.context)


def plot_avg_track_and_std(sessions_list, color='b', label='', filt=False):
    track, speeds, flees = get_tracks_and_speeds(sessions_list)
    if filt:
        track = np.array(track)[np.array(flees) == 1]
    avg_track = np.nanmean(track, axis=0)
    std_track = np.nanstd(track, axis=0)
    plt.plot(avg_track, color=color, linewidth=3, label=label, zorder=0)
    plt.fill_between(np.arange(0, 399), avg_track + std_track, avg_track - std_track, alpha=0.5, color=color, zorder=0)
    plt.plot(avg_track + std_track, color='k', alpha=0.4, linewidth=0.25)
    plt.plot(avg_track - std_track, color='k', alpha=0.4, linewidth=0.25)


def plot_session_avg(session, color='b', label=''):
    track, speeds, _ = get_tracks_and_speeds([session])
    avg_track = np.nanmean(track, axis=0)
    plt.plot(avg_track, linewidth=3, label=label, zorder=0, color=color, alpha=0.7)
    #std_track = np.nanstd(track, axis=0)
    #plt.fill_between(np.arange(0, 399), avg_track + std_track, avg_track - std_track, alpha=0.5, color=color, zorder=0)
    #plt.plot(avg_track + std_track, color='k', alpha=0.4, linewidth=0.25)
    #plt.plot(avg_track - std_track, color='k', alpha=0.4, linewidth=0.25)


def plot_each_mouse(session_list, color=None, label=None):
    for i, s in enumerate(session_list):
        if color is not None:
            plot_session_avg(s, label=label, color=color)
        else:
            plot_session_avg(s, label='mouse {}'.format(str(i)))


def get_avg_speed_and_latency(s):
    speeds = []
    latencies = []
    print(s.path)
    for name in os.listdir(s.path):
        loom_folder = os.path.join(s.path, name)
        if os.path.isdir(loom_folder):
            speed, arg_speed = get_peak_speed_and_latency(loom_folder, s.context)
            speeds.append(speed)
            latencies.append(arg_speed)
    return np.mean(speeds), np.mean(latencies)


def plot_avg_speed_latency_time_of_day(session_list, color=None, label=None):
    flee_times = []
    avg_speeds = []
    avg_latencies = []
    for s in session_list:
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
