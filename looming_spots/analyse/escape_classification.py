import numpy as np

from looming_spots.constants import (
    CLASSIFICATION_WINDOW_END,
    CLASSIFICATION_WINDOW_START,
    FRAME_RATE,
    SPEED_THRESHOLD,
    SHELTER_FRONT)

from looming_spots.analyse.tracks import n_samples_to_reach_shelter, get_peak_speed


def leaves_house(smoothed_track):
    if any(
        smoothed_track[
            CLASSIFICATION_WINDOW_END : CLASSIFICATION_WINDOW_END + 150
        ]
        > SHELTER_FRONT
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


def reaches_home(track):

    return any(
        [
            x < SHELTER_FRONT
            for x in track[
                CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END
            ]
        ]
    )


def classify_escape(normalised_x_track, speed_thresh=-SPEED_THRESHOLD):
    peak_speed, arg_peak_speed = get_peak_speed(normalised_x_track, return_loc=True)
    time_to_shelter = n_samples_to_reach_shelter(normalised_x_track)

    print(f'speed: {peak_speed}, '
          f'threshold: {speed_thresh}, '
          f'limit: {CLASSIFICATION_WINDOW_END-20}, '
          f'time to shelter: {time_to_shelter}')

    if time_to_shelter is None:
        print('never returns to shelter')
        return False

    is_escape = (peak_speed > speed_thresh) and (time_to_shelter < CLASSIFICATION_WINDOW_END)
    print(f'classified as escape: {is_escape}')
    return is_escape


def time_to_reach_home(track):

    in_home_idx = np.where(
        [x < SHELTER_FRONT for x in track[CLASSIFICATION_WINDOW_START:]]
    )[0]
    if len(in_home_idx) == 0:
        return np.nan
    return in_home_idx[0] / FRAME_RATE


def samples_to_shelter(normalised_x_track):
    in_home_idx = np.where(
        [x < SHELTER_FRONT for x in normalised_x_track[CLASSIFICATION_WINDOW_START:]]
    )[0]
    if len(in_home_idx) == 0:
        return np.nan
    return in_home_idx[0]
