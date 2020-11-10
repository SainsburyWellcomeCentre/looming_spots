import numpy as np
from scipy.ndimage import gaussian_filter

from looming_spots.preprocess.normalisation import (
    load_normalised_track,
    normalised_shelter_front,
)
from looming_spots.db.constants import (
    CLASSIFICATION_WINDOW_END,
    CLASSIFICATION_WINDOW_START,
    CLASSIFICATION_LATENCY,
    FRAME_RATE,
    SPEED_THRESHOLD,
    LOOM_ONSETS,
)


def leaves_house(smoothed_track, context):
    if any(
        smoothed_track[
            CLASSIFICATION_WINDOW_END : CLASSIFICATION_WINDOW_END + 150
        ]
        > normalised_shelter_front(context)
    ):
        return False
    else:
        return True


def fast_enough(speed):
    return any(
        [
            x < CLASSIFICATION_SPEED
            for x in speed[
                CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END
            ]
        ]
    )


def reaches_home(track, context):
    house_front = normalised_shelter_front(context)
    return any(
        [
            x < house_front
            for x in track[
                CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END
            ]
        ]
    )


def retreats_rapidly_at_onset(track):
    latency, _ = estimate_latency(track)
    if latency < CLASSIFICATION_LATENCY:
        return True


def classify_flee(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    speed = np.diff(track)

    if (
        fast_enough(speed)
        and reaches_home(track, context)
        and leaves_house(loom_folder, context)
    ):
        print(f"leaves: {leaves_house(loom_folder, context)}")
        return True

    print(
        f"fast enough: {fast_enough(speed)}, reaches home: {reaches_home(track, context)}"
    )
    return False


def estimate_latency(
    track,
    start=CLASSIFICATION_WINDOW_START,
    end=CLASSIFICATION_WINDOW_END,
    threshold=SPEED_THRESHOLD,
):
    speeds = np.diff(track)
    for i, speed in enumerate(speeds[start:end]):
        if speed < threshold:
            return start + i, track[start + i]
    return np.nan


def get_flee_duration(loom_folder, context):
    track = load_normalised_track(loom_folder, context)
    house_front = normalised_shelter_front(context)

    for i, x in enumerate(track[LOOM_ONSETS[0] :]):
        if x < house_front:
            return i
    return np.nan


def time_to_reach_home(track, context):
    house_front = normalised_shelter_front(context)
    in_home_idx = np.where(
        [x < house_front for x in track[CLASSIFICATION_WINDOW_START:]]
    )[0]
    if len(in_home_idx) == 0:
        return np.nan
    return in_home_idx[0] / FRAME_RATE


def samples_to_shelter(normalised_x_track):
    house_front = 0.2
    in_home_idx = np.where(
        [x < house_front for x in normalised_x_track[CLASSIFICATION_WINDOW_START:]]
    )[0]
    if len(in_home_idx) == 0:
        return np.nan
    return in_home_idx[0]


def get_peak_speed_and_latency(normalised_track):
    """
    :return peak_speed:
    :return arg_peak: the frame number of the peak speed
    """
    filtered_track = gaussian_filter(normalised_track, 3)
    distances = np.diff(filtered_track)
    peak_speed = np.nanmin(
        distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]
    )
    arg_peak = np.argmin(
        distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]
    )
    return -peak_speed, arg_peak + CLASSIFICATION_WINDOW_START


def loom_evoked_speed_change(
    normalised_x_speed, loom_onset, window_before=25, window_after=150
):
    return (
        np.mean(normalised_x_speed[(loom_onset - window_before) : loom_onset]),
        np.mean(
            normalised_x_speed[loom_onset + 5 : (loom_onset + window_after)]
        ),
    )


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
