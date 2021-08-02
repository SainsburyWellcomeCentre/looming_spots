import numpy as np
from looming_spots.analyse.tracks import get_peak_speed_and_latency
from scipy.ndimage import gaussian_filter

from looming_spots.preprocess.normalisation import (
    load_normalised_track,
    normalised_shelter_front,
)
from looming_spots.constants import (
    CLASSIFICATION_WINDOW_END,
    CLASSIFICATION_WINDOW_START,
    FRAME_RATE,
    SPEED_THRESHOLD,
    LOOM_ONSETS,
    ARENA_SIZE_CM, LOOMING_STIMULUS_ONSET)

from looming_spots.tracks import latency_peak_detect, n_samples_to_reach_shelter


def get_starts_and_ends(above_threshold, min_event_size=3):
    diff = np.diff(above_threshold.astype(int))
    unfiltered_starts = np.where(diff > 0)[0]
    unfiltered_ends = np.where(diff < 0)[0]

    if unfiltered_ends[0] < unfiltered_starts[0]:
        unfiltered_ends = unfiltered_ends[1:]
    if unfiltered_starts[-1] > unfiltered_ends[-1]:
        unfiltered_starts = unfiltered_starts[:-1]

    starts = [
        s
        for (s, e) in zip(unfiltered_starts, unfiltered_ends)
        if e - s > min_event_size
    ]
    ends = [
        e
        for (s, e) in zip(unfiltered_starts, unfiltered_ends)
        if e - s > min_event_size
    ]

    return starts, ends


def estimate_latency(normalised_x_track, smooth=False, limit=600):
    home_front = 0.2

    inside_house = normalised_x_track[:limit] < home_front
    smoothed_x_track = gaussian_filter(normalised_x_track, 2)
    smoothed_x_speed = np.diff(smoothed_x_track)
    normalised_x_speed = np.concatenate(
        [[np.nan], np.diff(normalised_x_track)])
    if smooth:
        speed = smoothed_x_speed[:limit]
    else:
        speed = normalised_x_speed[:limit]

    towards_house = speed < -0.0001

    starts, ends = get_starts_and_ends(towards_house, 7)

    for s, e in zip(starts, ends):
        if s > LOOMING_STIMULUS_ONSET:
            if s < LOOMING_STIMULUS_ONSET:
                continue
            elif any(inside_house[s:e]):
                return s
    print("did not find any starts... attempting with smoothed track")

    if not smooth:
        try:
            return estimate_latency(smooth=True) + 5
        except Exception as e:
            print(e)
            return np.nan


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
            x < SPEED_THRESHOLD
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


def peak_speed(normalised_x_track, return_loc=False):
    peak_speed, arg_peak_speed = get_peak_speed_and_latency(
        normalised_x_track
    )
    peak_speed = peak_speed * FRAME_RATE * ARENA_SIZE_CM
    if return_loc:
        return peak_speed, arg_peak_speed
    return peak_speed


def classify_escape(normalised_x_track, speed_thresh=-SPEED_THRESHOLD):
    peak_speed, arg_peak_speed = peak_speed(normalised_x_track, return_loc=True)
    latency = latency_peak_detect()
    time_to_shelter = n_samples_to_reach_shelter()

    print(f'speed: {peak_speed}, '
          f'threshold: {speed_thresh}, '
          f'latency: {latency} '
          f'limit: {CLASSIFICATION_WINDOW_END-20}, '
          f'time to shelter: {time_to_shelter}')

    if time_to_shelter is None or latency is None:
        return False

    return (peak_speed > speed_thresh) and (time_to_shelter < CLASSIFICATION_WINDOW_END)


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
