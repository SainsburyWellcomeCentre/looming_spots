import math
import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from looming_spots.track_analysis import arena_region_crossings
from looming_spots.tracking_dlc import process_DLC_output
import more_itertools as mit

plt.close('all')


LSIE_fpaths = ['1114170/20210408_15_49_13/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               '1114177/20210421_11_48_27/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               '1114178/20210417_14_41_09/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               '1114179/20210415_13_08_19/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               '1114309/20210423_14_28_47/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               '1114302/20210429_10_41_59/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
               ]

naive_fpaths = ['1114171/20210410_12_52_20/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114174/20210408_11_25_40/4_label_and_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114176/20210416_13_18_24/4_label_with_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114175/20210421_15_34_16/4_label_with_cricket/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114306/20210427_14_22_57/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114303/20210429_16_04_42/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                ]

new_paths = ['/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114309/20210423_14_28_47/camera.avi',
             '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114306/20210427_14_22_57/camera.avi',
             '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114302/20210429_10_41_59/camera.avi',
             '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114303/20210429_16_04_42/camera.avi',
             ]

vpaths= ['/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114178/20210417_14_41_09/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114170/20210408_15_49_13/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114177/20210421_11_48_27/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114179/20210415_13_08_19/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114171/20210410_12_52_20/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114174/20210408_11_25_40/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114176/20210416_13_18_24/camera.avi',
'/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114175/20210421_15_34_16/camera.avi',]

from scipy.ndimage import gaussian_filter1d
def plot_crickets(pathlist, max_n_samples):
    n_frames_to_show = 100
    eventlist = []
    events_pooled = []
    speeds = []
    n_bouts=-1
    x_tracks_all = []

    for i, fname in enumerate(pathlist):
        speeds = []
        x_tracks, y_tracks, bout_idx, distances, \
        cricket_x_tracks, cricket_y_tracks = get_x_and_y_tracks(fname, n_frames_to_show, max_n_samples)

        fig=plt.figure(figsize=(10,3))
        plt.title(f'{fname} MID')

        [plt.plot(gaussian_filter1d(x,2), gaussian_filter1d(y,2), color='r', alpha=0.5) for x, y in zip(x_tracks[:n_bouts], y_tracks[:n_bouts])]
        [plt.plot(gaussian_filter1d(x,2), gaussian_filter1d(y,2), color='b', alpha=0.5) for x, y in zip(cricket_x_tracks[:n_bouts], cricket_y_tracks[:n_bouts])]
        [plt.plot(x.values[0], y.values[0], 'o', color='y', alpha=0.6) for x, y in zip(x_tracks[:n_bouts], y_tracks[:n_bouts])]
        #fig.savefig('/home/slenzi/figures/cricket_examples_10min/' + fname.split('/')[0] + '.eps',fmt='eps')
        # plt.figure()
        # plt.title(f'{fname} MID')
        # plt.plot(np.arange(len(df_x['cricket'])), df_x['cricket'], color='b')
        # # plt.plot(np.arange(len(df_x['body'])), df_x['body'], color='k')
        # plt.plot(np.arange(len(distances)), distances, color='y')

        # [plt.plot(np.arange(len(x)), x, color='r') for x in x_tracks]

        eventlist.append(bout_idx)
        events_pooled.extend(bout_idx)
        print(fname, len(x_tracks))
        x_tracks_all.extend(x_tracks)
        for x, y in zip(x_tracks,y_tracks):
            speeds.extend([get_peak_speed(x, y)])
    return eventlist, events_pooled, speeds, x_tracks_all


    # plt.figure()
    # plt.eventplot(eventlist)
    # plt.figure()
    # plt.hist(events_pooled, histtype='step', bins=15)
    # plt.figure()
    # plt.hist(speeds, histtype='step', bins=15,color='r')
        # for x,y in zip(x_tracks[:n_bouts], y_tracks[:n_bouts]):
        #     distances.append(get_total_distance_travelled_in_bout(x, y))
        # for x,y in zip(x_tracks[:n_bouts], y_tracks[:n_bouts]):
        #
        #     d = get_peak_speed(x['body'], y['body'])
        #     speeds.append(d)
        # plt.scatter(np.arange(len(speeds)), speeds)

    #plt.hist(distances, bins=60,histtype='step')



    #
    # axes = plt.subplots(3, 4)
    # for i, fname in enumerate(pathlist):
    #     x_tracks, y_tracks = get_x_and_y_tracks(fname, n_frames_to_show)
    #
    #     ax = plt.sca(axes[1][0][i])
    #     plt.title(fname)
    #
    #     plot_bouts(len(x_tracks), house, tz, x_tracks, y_tracks, end=False)
    #
    #     ax = plt.sca(axes[1][1][i])
    #     plot_bouts(n_bouts, house, tz, x_tracks, y_tracks, end=False)
    #
    #     ax = plt.sca(axes[1][2][i])
    #     plot_bouts(n_bouts, house, tz, x_tracks, y_tracks, end=True)




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
    df_name = fname.split('/')[0] + '_' + fname.split('/')[1]
    df_path = f'/home/slenzi/winstor/margrie/slenzi/cricket_dfs/{df_name}.h5'
    p = pathlib.Path('/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/processed_data/') / fname
    if not os.path.isfile(df_path):


        df = pd.read_hdf(str(p))
        df = df[df.keys()[0][0]]
        start, end = process_DLC_output.get_first_and_last_likely_frame(df, 'cricket')
        print(f'{fname}, start:{start}, end: {end}')


        df = process_DLC_output.replace_low_likelihood_as_nan(df)[start:end]

        df.to_hdf(df_path,key='df')
    else:
        df = pd.read_hdf(df_path)
        print(fname, len(df))



    body_part_labels = ['body', 'cricket']
    body_parts = {body_part_label: df[body_part_label] for body_part_label in body_part_labels}
    df_y = pd.DataFrame(
        {body_part_label: 1 - (body_part["y"] / 200) for body_part_label, body_part in body_parts.items()})
    df_x = pd.DataFrame(
        {body_part_label: 1 - (body_part["x"] / 600) for body_part_label, body_part in body_parts.items()})

    bout_idx = get_bouts_minimum_distance(df_x, df_y, distance_threshold=0.25)
    x_tracks = [df_x['body'][entry:entry + n_frames_to_show] for entry in bout_idx]
    y_tracks = [df_y['body'][entry:entry + n_frames_to_show] for entry in bout_idx]

    x_tracks = [x for x in x_tracks if len(x) > 5]
    y_tracks = [y for y in y_tracks if len(y) > 5]

    cricket_x_tracks = [df_x['cricket'][entry:entry + n_frames_to_show] for entry in bout_idx]
    cricket_y_tracks = [df_y['cricket'][entry:entry + n_frames_to_show] for entry in bout_idx]

    cricket_x_tracks = [x for x in cricket_x_tracks if len(x) > 5]
    cricket_y_tracks = [y for y in cricket_y_tracks if len(y) > 5]

    distances = get_distance_between_cricket_and_mouse_centre(df_x, df_y)


    return x_tracks, y_tracks, bout_idx, distances, cricket_x_tracks, cricket_y_tracks


def get_peak_speed(x_track, y_track):
    distances = get_distances(x_track, y_track)
    return np.nanmax(np.abs(np.diff(distances)))

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
    theta_deg = np.rad2deg(np.arccos((delta_point [:,1]) / (np.sqrt(np.nansum(np.square(delta_point), 1))* np.linalg.norm([1,0])) ))
    return theta_deg


def plot_polar(theta):
    theta = np.array(theta)[~np.isnan(theta)]
    N = 10
    bottom = 8
    bar_heights = np.histogram(theta, bins=N, normed=True)
    width = (2 * np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(np.linspace(0,np.pi*2, N),bar_heights[0], width=36)
    plt.show()


def get_cricket_bouts(data_x, data_y, distance_threshold=0.25, inter_bout_interval=50):
    distances = get_distance_between_cricket_and_mouse_centre(data_x, data_y)
    mouse_within_range_of_cricket = np.array(distances) < distance_threshold
    encounters = np.where(np.diff(mouse_within_range_of_cricket))[0]
    y=0
    encounter_starts = []
    encounter_ends = []
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
    split_encounters = get_consecutive_points(list(encounters))

    closest_pos = []

    for bout in split_encounters:
        if len(bout) > minimum_bout_length:
            bout_distances = distances[bout[0]:bout[-1]]
            minimum_pos = np.argmin(bout_distances)
            closest_pos.append(bout[minimum_pos])
    y=0
    closest_pos_removed_interval = []
    for x in closest_pos:
        if x - y > inter_bout_interval:
            closest_pos_removed_interval.append(x)
        y=x
    return closest_pos_removed_interval

def get_consecutive_points(data):
    return [list(group) for group in mit.consecutive_groups(list(data))]

def get_distance_between_cricket_and_mouse_centre(data_x, data_y):
    body = np.array([data_x['body'], data_y['body']]).T
    cricket = np.array([data_x['cricket'], data_y['cricket']]).T
    distances = []
    for pt1, pt2 in zip(body, cricket):
        euc_dist = np.linalg.norm(pt1 - pt2)
        distances.extend([euc_dist])
    return distances


#theta_deg = acos((delta_point * [1; 0])./(sqrt(sum(delta_point.^2, 2)) * norm([1, 0])));

#(sqrt(sum(delta_point.^2, 2)) * norm([1, 0])));

ax2 = plt.subplot(111)
fig,axes = plt.subplots(2,3)

speeds_all = []
for i, (paths, c) in enumerate(zip([naive_fpaths, LSIE_fpaths], ['k', 'b'])):
    eventlist, events_pooled, speeds, x_tracks_all = plot_crickets(paths, max_n_samples=30000)
    speeds_all.append(speeds)
    plt.sca(axes[i][0])
    plt.eventplot(eventlist, color=c)
    plt.xlabel('sample number (50hz), first 60 min')
    plt.ylabel('mouse id')

    plt.sca(axes[i][1])
    plt.hist(events_pooled, histtype='step', bins=30, color=c)
    plt.ylim([0,60])
    plt.xlabel('sample number (50hz), first 60 min')
    plt.ylabel('count')

    plt.sca(axes[i][2])
    plt.hist(speeds, histtype='step', bins=30, color=c,normed=True)
    plt.ylim([0, 300])
    plt.xlim([0, 0.06])
    plt.xlabel('speed a.u.')
    plt.ylabel('count')

    plt.sca(ax2)
    for x in x_tracks_all:
        plt.plot(np.arange(len(x)),x, color=c, alpha=0.4)

from scipy import stats
print(stats.ttest_ind(speeds_all[0], speeds_all[1]))

plt.show()
print('done')
