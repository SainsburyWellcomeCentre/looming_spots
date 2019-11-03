import numpy as np
from scipy.ndimage import gaussian_filter

from looming_spots.preprocess.normalisation import (
    load_normalised_track,
    normalised_shelter_front,
)
from looming_spots.db.constants import (
    CLASSIFICATION_WINDOW_END,
    CLASSIFICATION_SPEED,
    CLASSIFICATION_WINDOW_START,
    CLASSIFICATION_LATENCY,
    FRAME_RATE,
    SPEED_THRESHOLD,
    STIMULUS_ONSETS,
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


def time_spent_hiding_deprecated(loom_folder, context):
    track = gaussian_filter(load_normalised_track(loom_folder, context), 3)
    stimulus_relevant_track = track[CLASSIFICATION_WINDOW_START:]

    home_front = normalised_shelter_front(context)
    safety_zone_border_crossings = np.where(
        np.diff(stimulus_relevant_track < home_front)
    )

    if len(safety_zone_border_crossings[0]) == 0:  # never runs away
        return 0
    elif len(safety_zone_border_crossings[0]) == 1:  # never comes back out
        print(f"this mouse never leaves {loom_folder}")
        print(safety_zone_border_crossings)
        return (
            int(
                len(stimulus_relevant_track)
                - int(safety_zone_border_crossings[0])
            )
            / FRAME_RATE
        )
    else:
        return int(safety_zone_border_crossings[0][1]) / FRAME_RATE


def time_spent_hiding(normalised_track, context):
    track = gaussian_filter(normalised_track, 3)
    stimulus_relevant_track = track[CLASSIFICATION_WINDOW_START:]

    home_front = normalised_shelter_front(context)
    safety_zone_border_crossings = np.where(
        np.diff(stimulus_relevant_track < home_front)
    )

    if len(safety_zone_border_crossings[0]) == 0:  # never runs away
        return 0
    elif len(safety_zone_border_crossings[0]) == 1:  # never comes back out
        print("this mouse never leaves")
        print(safety_zone_border_crossings)
        return (
            int(
                len(stimulus_relevant_track)
                - int(safety_zone_border_crossings[0])
            )
            / FRAME_RATE
        )
    else:
        return int(safety_zone_border_crossings[0][1]) / FRAME_RATE


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

    for i, x in enumerate(track[STIMULUS_ONSETS[0] :]):
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