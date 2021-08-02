from looming_spots.constants import LOOMING_STIMULUS_ONSET, LOOM_ONSETS
import numpy as np


def track_overlay(self, duration_in_samples=200, track_heatmap=None):
    if track_heatmap is None:
        track_heatmap = np.zeros((240, 600))  # TODO: get shape from raw

    x, y = (
        np.array(
            self.track_in_standard_space[0][
            LOOMING_STIMULUS_ONSET: LOOMING_STIMULUS_ONSET
                                    + duration_in_samples
            ]
        ),
        np.array(
            self.track_in_standard_space[1][
            LOOMING_STIMULUS_ONSET: LOOMING_STIMULUS_ONSET
                                    + duration_in_samples
            ]
        ),
    )
    for coordinate in zip(x, y):
        if not np.isnan(coordinate).any():
            track_heatmap[int(coordinate[1]), int(coordinate[0])] += 1
    return track_heatmap


def get_mean_speed_in_range(normalised_x_speed, s, e):
    return np.mean(normalised_x_speed[s:e])


def movement_loom_on_vs_loom_off(normalised_x_speed):
    loom_on_speeds = [
        get_mean_speed_in_range(
            normalised_x_speed, loom_onset, loom_onset + 14
        )
        for loom_onset in LOOM_ONSETS
    ]
    loom_off_speeds = [
        get_mean_speed_in_range(
            normalised_x_speed, loom_onset + 14, loom_onset + 28
        )
        for loom_onset in LOOM_ONSETS
    ]
    return min(loom_on_speeds) - min(loom_off_speeds)


def loom_evoked_speed_change(
    normalised_x_speed, loom_onset, window_before=25, window_after=150
):
    return (
        np.mean(normalised_x_speed[(loom_onset - window_before) : loom_onset]),
        np.mean(
            normalised_x_speed[loom_onset + 5 : (loom_onset + window_after)]
        ),
    )
