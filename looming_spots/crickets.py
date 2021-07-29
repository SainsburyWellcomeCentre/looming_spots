import math
import os
import pathlib

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from looming_spots.tracking_dlc import process_DLC_output
from matplotlib import pyplot as plt
import more_itertools as mit
from scipy.ndimage import gaussian_filter1d


def plot_tracks_all_mice(axes, cricket_bouts_x, cricket_bouts_y, i, n_bouts, x_tracks, y_tracks):
    shelters=[plt.Circle((0.2, 0.2), 0.08, fill=False, color='k', zorder=1000, linewidth=2, edgecolor='k',
                          linestyle='--') for ax in axes]

    [axes[i].plot(gaussian_filter1d(x, 2), gaussian_filter1d(y, 2), color='r', alpha=0.5) for x, y in
     zip(x_tracks[:n_bouts], y_tracks[:n_bouts])]
    [axes[i].plot(gaussian_filter1d(x, 2), gaussian_filter1d(y, 2), color='b', alpha=0.5) for x, y in
     zip(cricket_bouts_x[:n_bouts], cricket_bouts_y[:n_bouts])]
    [axes[i].plot(x.values[0], y.values[0], 'o', color='y', alpha=0.6) for x, y in
     zip(x_tracks[:n_bouts], y_tracks[:n_bouts])]
    [ax.set_xlim([0, 1]) for ax in axes]

    for ax, s in zip(axes, shelters):
        plt.sca(ax)
        plt.axis('off')
        ax.add_patch(s)
        plt.xlim([0,1])
        plt.ylim([0,0.4])


def plot_bouts(n_bouts, house, tz, x_tracks, y_tracks, end = False):
    x_tracks_to_show = x_tracks[-n_bouts:] if end else x_tracks[:n_bouts]
    y_tracks_to_show = y_tracks[-n_bouts:] if end else y_tracks[:n_bouts]

    for x, y in zip(x_tracks_to_show, y_tracks_to_show):
        plt.plot(x['body'], y['body'], alpha=0.2, color='k')
        plt.plot(x['cricket'].iloc()[0], y['cricket'].iloc()[0], 'o', alpha=0.2, color='b')
        plt.plot(x['body'].iloc()[0], y['body'].iloc()[0], 'o', alpha=0.2, color='k')
        plt.plot(x['body'].iloc()[-1], y['body'].iloc()[-1], 'o', alpha=0.2, color='r')
    plt.xlim([0, 1])
    plt.axvline([house], linestyle='--')
    plt.axvline([tz], linestyle='--')
    plt.axis('off')


def get_x_and_y_tracks(fname, n_frames_to_show, max_n_samples=30000):

    df = get_dlc_tracks(fname)
    body_part_labels = ['body', 'cricket']

    body_parts = {body_part_label: df[body_part_label] for body_part_label in body_part_labels}
    df_y = pd.DataFrame(
        {body_part_label: (1 - (body_part["y"] / 240)) * 0.4 for body_part_label, body_part in body_parts.items()})
    df_x = pd.DataFrame(
        {body_part_label: 1 - (body_part["x"] / 600) for body_part_label, body_part in body_parts.items()})

    bout_idx = get_bouts_minimum_distance(df_x, df_y, distance_threshold=0.2)

    x_tracks = [df_x['body'][entry:entry + n_frames_to_show] for entry in bout_idx]
    y_tracks = [df_y['body'][entry:entry + n_frames_to_show] for entry in bout_idx]

    x_tracks = [x for x in x_tracks if len(x) > 5]
    y_tracks = [y for y in y_tracks if len(y) > 5]

    cricket_x_tracks = [df_x['cricket'][entry:entry + n_frames_to_show] for entry in bout_idx]
    cricket_y_tracks = [df_y['cricket'][entry:entry + n_frames_to_show] for entry in bout_idx]

    cricket_x_tracks = [x for x in cricket_x_tracks if len(x) > 5]
    cricket_y_tracks = [y for y in cricket_y_tracks if len(y) > 5]

    distances = get_distance_between_cricket_and_mouse_centre(df_x, df_y)
    approach_distances = [distances[x] for x in bout_idx]

    return x_tracks, y_tracks, bout_idx, distances, cricket_x_tracks, cricket_y_tracks, approach_distances


def get_dlc_tracks(fname):
    df_name = fname.split('/')[0] + '_' + fname.split('/')[1]
    df_path = f'Z:\\margrie\\slenzi\\cricket_dfs\\{df_name}.h5'
    p = pathlib.Path('Z:\\margrie\\glusterfs\\imaging\\l\\loomer\\processed_data\\') / fname
    if not os.path.isfile(df_path):

        df = pd.read_hdf(str(p))
        df = df[df.keys()[0][0]]
        start, end = process_DLC_output.get_first_and_last_likely_frame(df, 'cricket')
        print(f'{fname}, start:{start}, end: {end}')

        df = process_DLC_output.replace_low_likelihood_as_nan(df)[start:end]

        df.to_hdf(df_path, key='df')
    else:
        df = pd.read_hdf(df_path)
        print(fname, len(df))
    return df


def get_peak_speed(x_track, y_track, frame_rate=50, ARENA_LENGTH_CM=50):
    distances = np.array(get_distances(x_track, y_track))
    return np.nanmax(np.abs(np.diff(distances))) * frame_rate * ARENA_LENGTH_CM


def get_distances(x_track, y_track):
    distances =[]
    coordinates = np.array(x_track, y_track)
    for this_point, next_point in zip(coordinates[:-1], coordinates[1:]):
        distances.append(np.linalg.norm(next_point-this_point))
    return distances


def get_total_distance_travelled_in_bout(x_track, y_track):
    pt1 = np.array(x_track.iloc()[0], y_track.iloc()[0])
    pt2 = np.array(x_track.iloc()[-1], y_track.iloc()[-1])
    return np.linalg.norm(pt1-pt2)


def get_heading(df_x, df_y):
    left_ear = np.array([df_x['L_ear'], df_y['L_ear']]).T
    right_ear = np.array([df_x['R_ear'], df_y['R_ear']]).T
    body = np.array([df_x['body'], df_y['body']]).T
    head_middle = (left_ear + right_ear) / 2
    delta_point = np.array(head_middle - body)
    theta_rad = np.arccos((delta_point [:,1]) / (np.sqrt(np.nansum(np.square(delta_point), 1))* np.linalg.norm([1,0])) )
    return theta_rad


def plot_polar(theta):
    theta = np.array(theta)[~np.isnan(theta)]
    N = 10
    bottom = 8
    bar_heights = np.histogram(theta, bins=N, normed=True)
    width = (2 * np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(np.linspace(0,np.pi*2, N),bar_heights[0], width=36)
    plt.show()


def get_cricket_bout_start_idx(data_x, data_y, distance_threshold=0.25, inter_bout_interval=50):
    distances = get_distance_between_cricket_and_mouse_centre(data_x, data_y)
    mouse_within_range_of_cricket = np.array(distances) < distance_threshold
    encounters = np.where(np.diff(mouse_within_range_of_cricket))[0]
    y=0
    encounter_starts = []
    for x in encounters:
        if (x - y < inter_bout_interval) and (y != 0):
            continue
        else:
            encounter_starts.append(x)
            y=x

    return encounter_starts


def get_bouts_minimum_distance(data_x, data_y, distance_threshold=0.2, inter_bout_interval=50, minimum_bout_length=10):
    distances = get_distance_between_cricket_and_mouse_centre(data_x, data_y)
    mouse_within_range_of_cricket = np.array(distances) < distance_threshold
    encounters = np.where(mouse_within_range_of_cricket)[0]
    bouts = get_continuous_segments(list(encounters))

    closest_pos = []

    for bout in bouts:
        if len(bout) > minimum_bout_length:
            bout_distances = distances[bout[0]:bout[-1]]
            minimum_pos = np.argmin(bout_distances)
            closest_pos.append(bout[minimum_pos])
    y=0
    closest_pos_removed_interval = []
    for x in closest_pos:
        if x - y > inter_bout_interval:
            closest_pos_removed_interval.append(x)
        y = x
    return closest_pos_removed_interval


def get_continuous_segments(data):
    return [list(group) for group in mit.consecutive_groups(list(data))]


def get_distance_between_cricket_and_mouse_centre(data_x, data_y):
    body = np.array([data_x['body'], data_y['body']]).T
    cricket = np.array([data_x['cricket'], data_y['cricket']]).T
    distances = []
    for pt1, pt2 in zip(body, cricket):
        euc_dist = np.linalg.norm(pt1 - pt2)
        distances.extend([euc_dist])
    return distances


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang  if ang < 0 else ang


def get_inner_angle(p1, p2, p3):
    p12 = get_length(p1, p2)
    p13 = get_length(p1, p3)
    p23 = get_length(p2, p3)
    ang = np.arccos((p12**2 + p13**2 - p23**2)/ (2* p12 * p13))
    return ang


def get_length(p1, p2):
    p12 = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return p12


def get_angle_between_mouse_cricket_and_shelter(mouse_pos, cricket_pos, shelter_pos):
    angle = get_inner_angle(cricket_pos,mouse_pos, shelter_pos)
    if mouse_pos[1] > cricket_pos[1]:
        angle = -angle
    return angle


def get_positions(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y):
    mouse_coords = []
    cricket_coords = []
    for bout in zip(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y):
        mouse_pos = [bout[0].values[0], bout[1].values[0]]
        cricket_pos = [bout[2].values[0], bout[3].values[0]]
        mouse_coords.append(mouse_pos)
        cricket_coords.append(cricket_pos)
    return mouse_coords, cricket_coords


def get_all_angles(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y, shelter_pos=(0.1, 0.2)):
    mouse_bout_positions, cricket_bout_positions = get_positions(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x,
                                                                 cricket_bouts_y)

    all_bout_angles = []
    all_bout_distances = []
    for mouse_pos, cricket_pos in zip(mouse_bout_positions, cricket_bout_positions):
        all_bout_angles.append(get_angle_between_mouse_cricket_and_shelter(mouse_pos, cricket_pos, shelter_pos))
        distance_between_cricket_and_mouse = np.linalg.norm(np.array(cricket_pos) - np.array(mouse_pos))
        all_bout_distances.append(distance_between_cricket_and_mouse)
    return all_bout_angles, all_bout_distances


def verify_angle_calculations(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y,shelter_location=(0.1,0.2)):
    all_bout_angles, all_bout_distances = get_all_angles(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y)
    for boutx, bouty,boutxcrick,boutycrick, angle, distance in zip(mouse_bouts_x[:15], mouse_bouts_y, cricket_bouts_x, cricket_bouts_y ,all_bout_angles, all_bout_distances):
        plt.figure()
        plt.plot(boutx, bouty, color='r')
        plt.plot(boutxcrick, boutycrick, color='b')
        plt.plot([boutx.values[0], boutxcrick.values[0]], [bouty.values[0], boutycrick.values[0]], color='g', linestyle='-')
        plt.plot([shelter_location[0], boutxcrick.values[0]], [shelter_location[1], boutycrick.values[0]], color='y', linestyle='-')
        plt.text(0.2,0.2, np.rad2deg(angle))
        plt.xlim([0,1])
        plt.ylim([0,0.4])


def compute_stats(summary_df, label1='naive', label2='LSIE', metric='count', normal=True):
    grp1 = summary_df[summary_df['group'] == label1][metric].values
    grp2 = summary_df[summary_df['group'] == label2][metric].values
    if not normal:
        stat = scipy.stats.mannwhitneyu(grp1, grp2)
    else:
        stat = scipy.stats.ttest_ind(grp1, grp2)
    print(f'{metric} comparison of {label1} vs. {label2}: {stat}')
    return stat


def plot_stats(summary_df, axes):
    #plt.figure(figsize=(1, 5))
    plt.sca(axes[1][2])
    ax=plt.gca()
    sns.barplot(data=summary_df, x='group', y='count', zorder=0, palette=['k', 'b'])
    sns.scatterplot(data=summary_df, x='group', y='count', hue='group', s=100, palette=['k', 'b'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    stat = compute_stats(summary_df, normal=False)
    plt.title(f'retreats to shelter p={stat[1].round(3)}, MWU')
    change_width(ax, 0.2)




def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        patch.set_width(new_value)

        patch.set_x(patch.get_x() + diff * .5)


def generate_summary_df(df):
    summary_df_dict = {}
    mids = []
    counts = []
    median_speeds = []
    groups = []
    for mid in df['mouse_id'].unique():
        mouse_df = df[df['mouse_id'] == mid]
        mids.append(mid)
        counts.append(len(mouse_df))
        median_speeds.append(mouse_df['speeds'].median())
        groups.append(mouse_df['group'].unique()[0])
    summary_df_dict.setdefault('group', groups)
    summary_df_dict.setdefault('mid', mids)
    summary_df_dict.setdefault('count', counts)
    summary_df_dict.setdefault('median_speed', median_speeds)
    summary_df = pd.DataFrame.from_dict(summary_df_dict)
    return summary_df
