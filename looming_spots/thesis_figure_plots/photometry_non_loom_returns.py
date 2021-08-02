import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
from looming_spots.db import trial_group, experimental_log
from looming_spots.thesis_figure_plots import photometry_example_traces
group='photometry_habituation_tre-GCaMP-contrasts'
mtgs = experimental_log.get_mtgs_in_experiment(group)

mtgs = [m for m in mtgs if m.mouse_id not in ["074744"]]


def entry_bools(normalised_x_track):
    shelter_boundary = 0.2  # normalised_shelter_front(context)
    tz_boundary = 0.6
    positions = {
        "shelter": normalised_x_track < shelter_boundary,
        "middle": np.logical_and(
            (tz_boundary > normalised_x_track),
            (normalised_x_track > shelter_boundary),
        ),
        "tz": normalised_x_track > tz_boundary,
    }
    return positions


def get_all_entries(normalised_x_track):
    entries = entry_bools(normalised_x_track)
    shelter_entries = np.where(np.diff(entries['shelter'].astype(int)) == 1)[0]
    tz_entries = np.where(np.diff(entries['tz'].astype(int)) == 1)[0]
    middle_entries = np.where(np.diff(entries['middle'].astype(int)) == 1)[0]
    track_starts = []

    for i, tz_entry in enumerate(tz_entries):
        try:
            first_next_middle = min(middle_entries[middle_entries > tz_entry], key=lambda x: abs(x - tz_entry))

            first_next_shelter = min(shelter_entries[shelter_entries > first_next_middle],
                                     key=lambda x: abs(x - first_next_middle))
            first_next_tz = min(tz_entries[tz_entries > first_next_middle], key=lambda x: abs(x - first_next_middle))
            if first_next_tz < first_next_shelter:
                continue
            else:
                track_starts.append(first_next_middle)
        except Exception as e:
            return track_starts

    return track_starts


def get_first_7min_track_and_signal(mtg):
    s=90
    e = 30*7*60
    for session in mtg.sessions:
        if len(np.unique(session.track()[0])) > 10000:
            return 1 - (session.track()[0][s:e]/600), session.data['delta_f'][s:e]

mtg=mtgs[-1]
end = 30*7*60
track, signal=get_first_7min_track_and_signal(mtg)

plt.figure()
avg_df =[]
avg_df_explore =[]

for start in get_all_entries(track[:-2000]):
    delta_f=signal[start - 200:start + 400]-np.median(signal[start-25:start])
    plt.plot(delta_f, color='k')
    avg_df_explore.append(delta_f)
plt.plot(np.mean(avg_df_explore, axis=0), linewidth=3)

for t in mtg.loom_trials():
    if t.contrast ==0:
        delta_f=t.delta_f()[:600]
        plt.plot(delta_f, color='r')
        avg_df.append(delta_f)
plt.plot(np.mean(avg_df, axis=0), linewidth=3)


plt.figure()
avg_track_before=[]
for start in get_all_entries(track[:-2000]):
    track_before= track[start - 200:start + 400]
    plt.plot(track_before, color='k')
    avg_track_before.append(track_before)
plt.plot(np.mean(avg_track_before,axis=0), linewidth=3)
avg_track_loom=[]
for t in mtg.loom_trials():
    if t.contrast ==0:
        plt.plot(t.normalised_x_track[:600], color='r')
        avg_track_loom.append(t.normalised_x_track[:600])
plt.plot(np.mean(avg_track_loom, axis=0), linewidth=3)