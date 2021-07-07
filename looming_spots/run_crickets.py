import random

import pandas as pd
import numpy as np
from looming_spots.crickets import plot_tracks_all_mice, get_x_and_y_tracks, get_peak_speed, get_all_angles, plot_stats, \
    generate_summary_df
import scipy
import math
import scipy.stats
import matplotlib.pyplot as plt

PIXEL_DISTANCE = 0.0833

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
                #'1114306/20210427_14_22_57/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                '1114303/20210429_16_04_42/cameraDLC_resnet50_crickets_small_videosJun2shuffle1_1030000.h5',
                ]

# new_paths = ['/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114309/20210423_14_28_47/camera.avi',
#              '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114306/20210427_14_22_57/camera.avi',
#              '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114302/20210429_10_41_59/camera.avi',
#              '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114303/20210429_16_04_42/camera.avi',
#              ]
#
# vpaths= ['/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114178/20210417_14_41_09/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114170/20210408_15_49_13/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114177/20210421_11_48_27/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114179/20210415_13_08_19/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114171/20210410_12_52_20/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114174/20210408_11_25_40/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114176/20210416_13_18_24/camera.avi',
# '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/1114175/20210421_15_34_16/camera.avi',]

from scipy.ndimage import gaussian_filter1d


def plot_crickets(pathlist, max_n_samples, group_label, house_start=0.3,n_examples=5, subset='to_shelter'):
    n_frames_to_show = 100
    eventlist = []
    events_pooled = []
    speeds = []
    n_bouts = -1
    mouse_bouts_x = []
    mouse_bouts_y = []

    cricket_bouts_x = []
    cricket_bouts_y = []

    all_trials = []

    df_all = pd.DataFrame()
    fig, axes = plt.subplots(6, 1,figsize=(4, 10))
    plt.title(f'{group_label}')
    approach_distances_group=[]
    for i, fname in enumerate(pathlist):
        example_track_dict = {}
        df_dict = {}
        mid = fname.split('/')[0]
        speeds = []
        x_tracks_unspecified, y_tracks_unspecified, bout_idx_unspecified, distances, \
        cricket_x_tracks_unspecified, cricket_y_tracks_unspecified, approach_distances = get_x_and_y_tracks(fname, n_frames_to_show, max_n_samples)

        x_tracks=[] # to shelter
        y_tracks=[]
        bout_idx=[]
        cricket_x_tracks = []
        cricket_y_tracks = []
        approach_distances_group.extend(approach_distances)

        for x, y, idx, crick_x, crick_y in zip(x_tracks_unspecified, y_tracks_unspecified, bout_idx_unspecified, cricket_x_tracks_unspecified, cricket_y_tracks_unspecified):
            if subset == 'all':
                x_tracks.append(x)
                y_tracks.append(y)
                bout_idx.append(idx)
                cricket_x_tracks.append(crick_x)
                cricket_y_tracks.append(crick_y)
            elif subset=='to_shelter':
                if any(x < house_start):
                    x_tracks.append(x)
                    y_tracks.append(y)
                    bout_idx.append(idx)
                    cricket_x_tracks.append(crick_x)
                    cricket_y_tracks.append(crick_y)
            elif subset == 'not_to_shelter':
                if not any(x < house_start):
                    x_tracks.append(x)
                    y_tracks.append(y)
                    bout_idx.append(idx)
                    cricket_x_tracks.append(crick_x)
                    cricket_y_tracks.append(crick_y)

        number_of_bouts = len(bout_idx)
        bout_numbers = np.arange(number_of_bouts)
        groups = [group_label] * number_of_bouts
        mouse_ids = [mid]*number_of_bouts

        plot_tracks_all_mice(axes, cricket_x_tracks, cricket_y_tracks, i, n_bouts, x_tracks, y_tracks)

        frame_rate = 50
        events = np.array(bout_idx)/ frame_rate / 60
        eventlist.append(events)
        events_pooled.extend(events)

        print(fname, len(x_tracks))
        mouse_bouts_x.extend(x_tracks)
        mouse_bouts_y.extend(y_tracks)
        cricket_bouts_x.extend(cricket_x_tracks)
        cricket_bouts_y.extend(cricket_y_tracks)

        for x, y in zip(x_tracks,y_tracks):
            speeds.extend([get_peak_speed(x, y)])

        df_dict.setdefault('speeds', speeds)
        df_dict.setdefault('bout_idx', bout_numbers)
        df_dict.setdefault('group', groups)
        df_dict.setdefault('mouse_id', mouse_ids)

        mouse_df = pd.DataFrame.from_dict(df_dict)

        df_all = df_all.append(mouse_df, ignore_index=True)

        n_example_trials=(min(len(x_tracks), n_examples))

        if len(x_tracks) > 1:
            for j in range(n_example_trials):
                example_track_dict.setdefault(f't{j}', [x_tracks[j], y_tracks[j],cricket_x_tracks[j],cricket_y_tracks[j]])
            all_trials.append(example_track_dict)

    return eventlist, events_pooled, speeds, mouse_bouts_x, df_all, \
           approach_distances_group, mouse_bouts_y, all_trials, cricket_bouts_x, cricket_bouts_y


def check_x_distribution(x_tracks_grp1, x_tracks_grp2):
    grp1_pos = [bout.values[0] for bout in x_tracks_grp1]
    grp2_pos = [bout.values[0] for bout in x_tracks_grp2]
    plt.figure()
    plt.hist(grp1_pos, bins=80, histtype='step')
    plt.hist(grp2_pos, bins=80, histtype='step')
    print(f'comparison of starting pos: {scipy.stats.ks_2samp(grp1_pos, grp2_pos)}')
    return grp1_pos, grp2_pos


def get_from_all_trials_dict(all_trials):
    example_tracks_x = []
    example_tracks_y = []
    example_tracks_cricket_x = []
    example_tracks_cricket_y = []
    for group in all_trials:
        group_examples_x =[]
        group_examples_y =[]
        group_examples_cricket_x =[]
        group_examples_cricket_y =[]
        for mouse in group:
            for trial in mouse.values():
                group_examples_x.append(trial[0])
                group_examples_y.append(trial[1])
                group_examples_cricket_x.append(trial[2])
                group_examples_cricket_y.append(trial[3])
        example_tracks_x.append(group_examples_x)
        example_tracks_y.append(group_examples_y)
        example_tracks_cricket_x.append(group_examples_cricket_x)
        example_tracks_cricket_y.append(group_examples_cricket_y)
    return example_tracks_x, example_tracks_y, example_tracks_cricket_x, example_tracks_cricket_y


def plot_example_tracks(examples_x, examples_y, example_tracks_cricket_x, example_tracks_cricket_y, axes=None):
    if axes is None:
        fig_tracks, axes = plt.subplots(2, 1)
    plt.sca(axes[0])
    #circle1 = plt.Circle((0.8, 0.2), 0.1, color='k', alpha=0.2, zorder=1000, linewidth=0)
    #circle2 = plt.Circle((0.8, 0.2), 0.1, color='k', alpha=0.2, zorder=1000, linewidth=0)
    shelter1 = plt.Circle((0.2, 0.2), 0.08, fill=False, color='k', zorder=1000, linewidth=2, edgecolor='k',
                          linestyle='--')
    shelter2 = plt.Circle((0.2, 0.2), 0.08, fill=False, zorder=1000, linewidth=2, edgecolor='k', linestyle='--')
    sigma = 7
    plt.sca(axes[0])
    ax5 = plt.gca()
    for x, y, c_x, c_y in zip(examples_x[0], examples_y[0], example_tracks_cricket_x[0], example_tracks_cricket_y[0]):
        plt.plot(gaussian_filter1d(x, sigma), gaussian_filter1d(y, sigma), color='r')
        plt.plot(c_x.values[0], c_y.values[0], 'o',color='darkgrey')
    #ax5.add_patch(circle1)
    ax5.add_patch(shelter1)
    plt.text(0.2, 0.2, 'S')
    plt.text(0.8, 0.2, 'C')
    plt.xlim([0, 1])
    plt.ylim([-0.25, 0.65])
    plt.axis('off')

    plt.sca(axes[1])
    ax6 = plt.gca()
    for x, y, c_x, c_y in zip(examples_x[1], examples_y[1],example_tracks_cricket_x[1], example_tracks_cricket_y[1]):
        plt.plot(gaussian_filter1d(x, sigma), gaussian_filter1d(y, sigma), color='r')
        plt.plot(c_x.values[0], c_y.values[0], 'o', color='darkgrey')

    plt.xlim([0, 1])
    plt.ylim([-0.25, 0.65])

    #ax6.add_patch(circle2)
    ax6.add_patch(shelter2)
    plt.text(0.2, 0.2, 'S')
    plt.text(0.8, 0.2, 'C')
    plt.axis('off')



fig,axes = plt.subplots(3,3,figsize=(10,10))

speeds_all = []
counts_all = []
approach_distances_all = []
df_main = pd.DataFrame()
time_shown = 60

all_example_tracks_x = []
all_example_tracks_y = []

all_example_cricket_tracks_x = []
all_example_cricket_tracks_y = []
frame_rate = 50
all_trials = []
percents=[]

all_x_bouts = []

polar_fig, pol_ax = plt.subplots(subplot_kw={'projection': 'polar'});


def plot_percentage_to_shelter(eventlist_to_shelter, eventlist_all_interactions):
    plt.sca(axes[3][2])
    all_mouse_percents = []
    for mouse_to_shelter, mouse_all_interactions in zip(eventlist_to_shelter, eventlist_all_interactions):
        mouse_count, _ = np.histogram(mouse_to_shelter,bins=int(time_shown / 4), range=(0, 60))[0]
        mouse_interaction_count, bin_edges = np.histogram(mouse_all_interactions,bins=int(time_shown / 4), range=(0,60))
        percent_to_shelter = mouse_count / mouse_interaction_count
        all_mouse_percents.append(percent_to_shelter)
        plt.plot(np.arange(len(percent_to_shelter)), percent_to_shelter * 100, color=c, linewidth=1, alpha=0.2)

    #plt.plot(np.arange(len(percent_to_shelter)), np.nanmean(all_mouse_percents, axis=0) * 100, color=c, linewidth=2)
    plt.ylabel('% to shelter')
    plt.ylim([0, 100])
    ax=plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


for i, (paths, c, group_label) in enumerate(zip([naive_fpaths, LSIE_fpaths], ['k', 'b'], ['naive', 'LSIE'])):
    eventlist, events_pooled, speeds, mouse_bouts_x, df_group, approach_distances, mouse_bouts_y, trials, \
    cricket_bouts_x, cricket_bouts_y = plot_crickets(paths, max_n_samples=30000, group_label=group_label, n_examples=int(30/len(paths)), subset='to_shelter')

    eventlist_all_bouts, events_pooled_all_bouts, _, _, _, _, _, _, \
    _, _ = plot_crickets(paths, max_n_samples=30000, group_label=group_label,
                                                     n_examples=int(30 / len(paths)), subset='all')

    angles, distances = get_all_angles(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y)
    #verify_angle_calculations(mouse_bouts_x, mouse_bouts_y, cricket_bouts_x, cricket_bouts_y)

    for ang, distance in zip(angles, distances):
        pol_ax.plot(ang, distance*0.5, 'o', color=c, markersize=5,zorder=random.randrange(0,1000))

    all_trials.append(trials)
    all_x_bouts.append(mouse_bouts_x)
    df_main = df_main.append(df_group, ignore_index=True)
    speeds_all.append(speeds)
    approach_distances_all.append(approach_distances)

    counts_all.append(events_pooled)

    plt.sca(axes[1][i])
    plt.eventplot(eventlist, color=c)
    ax=plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'time (min)')
    plt.ylabel('mouse id')

    plt.sca(axes[2][i])
    count = plt.hist(events_pooled, histtype='step', bins=int(time_shown/4), range=(0,60), color=c)[0]
    interaction_count, bin_edges, _ = plt.hist(events_pooled_all_bouts, histtype='step', bins=int(time_shown/4), color=c, linestyle='--')
    percent_to_shelter = count/interaction_count
    percents.append(percent_to_shelter)


    plt.ylim([0, 100])
    plt.xlabel(f'time (min)')
    plt.ylabel('count')
    ax=plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.sca(axes[2][2])
    ax=plt.gca()
    plt.plot(np.arange(len(percent_to_shelter)), percent_to_shelter*100, color=c, linewidth=2)
    #[ax.bar(i,percent_to_shelter[i]*100,width=len(percent_to_shelter)//50) for i in range(len(percent_to_shelter))]
    #plt.hist(range(0,60,4), weights=percents[0] * 100, bins=range(0,60,4), color=c, histtype='step')
    plt.ylabel('% to shelter')
    #plt.ylim([0,100])
    plt.xlabel(f'time (min)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



check_x_distribution(all_x_bouts[0], all_x_bouts[1])


print(f'ks test 2 sample speeds: {scipy.stats.ks_2samp(speeds_all[0], speeds_all[1])}')
print(f'ks test 2 sample counts: {scipy.stats.ks_2samp(counts_all[0], counts_all[1])}')
print(f'ks test 2 % to shelter: {scipy.stats.ks_2samp(percents[0], percents[1])}')

examples_x, examples_y, example_tracks_cricket_x, example_tracks_cricket_y = get_from_all_trials_dict(all_trials)
plot_example_tracks(examples_x, examples_y, example_tracks_cricket_x, example_tracks_cricket_y, axes=axes[0][0:2])


summary_df = generate_summary_df(df_main)
plot_stats(summary_df, axes)
plt.show()
print('done')
