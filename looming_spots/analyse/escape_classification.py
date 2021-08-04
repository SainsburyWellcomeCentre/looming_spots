import numpy as np

from looming_spots.constants import (
    CLASSIFICATION_WINDOW_END,
    SPEED_THRESHOLD,
    FRAME_RATE,
    LOOMING_STIMULUS_ONSET, FREEZE_BUFFER_FRAMES)

from looming_spots.analyse.tracks import (
    n_samples_to_reach_shelter,
    get_peak_speed,
)

"""
Contains all functions used for classification of behavioural responses as escape or freezing
"""


def classify_escape(normalised_x_track, speed_thresh=-SPEED_THRESHOLD):

    f"""
    Escape is classified as returning to shelter with a peak speed of at least: {-SPEED_THRESHOLD}
    within {CLASSIFICATION_WINDOW_END/FRAME_RATE}s of stimulus onset

    :param normalised_x_track:
    :param speed_thresh:
    :return:
    """

    peak_speed, arg_peak_speed = get_peak_speed(
        normalised_x_track, return_loc=True
    )
    time_to_shelter = n_samples_to_reach_shelter(normalised_x_track)

    print(
        f"speed: {peak_speed}, "
        f"threshold: {speed_thresh}, "
        f"limit: {CLASSIFICATION_WINDOW_END}, "
        f"time to shelter: {time_to_shelter}"
    )

    if time_to_shelter is None:
        print("never returns to shelter")
        return False

    is_escape = (peak_speed > speed_thresh) and (
        time_to_shelter < CLASSIFICATION_WINDOW_END
    )
    print(f"classified as escape: {is_escape}")
    return is_escape


def is_track_a_freeze(unsmoothed_speed):

    upper_percentile = 97.5
    lower_percentile = 2.5
    freeze_metric_threshold = 2.5

    onset = LOOMING_STIMULUS_ONSET + FREEZE_BUFFER_FRAMES

    freeze_metric = \
        np.percentile(unsmoothed_speed[onset:CLASSIFICATION_WINDOW_END], upper_percentile) - \
        np.percentile(unsmoothed_speed[onset:CLASSIFICATION_WINDOW_END], lower_percentile)

    is_freeze = freeze_metric < freeze_metric_threshold

    return is_freeze


def classify_response():
    pass
