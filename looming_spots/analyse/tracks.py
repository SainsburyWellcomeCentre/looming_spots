import numpy as np
from looming_spots.db.constants import FRAME_RATE, N_SAMPLES_BEFORE, CLASSIFICATION_WINDOW_START, \
    CLASSIFICATION_WINDOW_END, ARENA_SIZE_CM, BOX_CORNER_COORDINATES
from looming_spots.util.transformations import get_inverse_projective_transform
from scipy.ndimage import gaussian_filter


def downsample_track(normalised_track, frame_rate, n_samples_before):
    n_points_ori = len(normalised_track)
    n_points_new = int(n_points_ori * (FRAME_RATE / frame_rate))
    track_timebase = (np.arange(len(normalised_track)) - n_samples_before) / frame_rate
    new_timebase = (np.arange(n_points_new) - N_SAMPLES_BEFORE) / FRAME_RATE
    normalised_track = np.interp(new_timebase, track_timebase, normalised_track)
    return normalised_track


def normalise_speed(normalised_track):
    normalised_speed = np.concatenate([[np.nan], np.diff(normalised_track)])
    return normalised_speed


def smooth_track(normalised_track):
    smoothed_track = gaussian_filter(normalised_track, 2)
    return smoothed_track


def smooth_speed(normalised_track):
    smoothed_track = smooth_track(normalised_track)
    return np.diff(smoothed_track)


def smooth_acceleration(normalised_track):
    smoothed_speed = smooth_speed(normalised_track)
    return np.roll(np.diff(smoothed_speed), 2)


def peak_speed(normalised_x_track, return_loc=False):
    peak_speed, arg_peak_speed = get_peak_speed_and_latency(
        normalised_x_track
    )
    peak_speed = peak_speed * FRAME_RATE * ARENA_SIZE_CM

    if return_loc:
        return peak_speed, arg_peak_speed

    return peak_speed


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


def projective_transform_tracks(Xin, Yin,
                                observed_box_corner_coordinates,
                                target_box_corner_coordinates=BOX_CORNER_COORDINATES):
    p = get_inverse_projective_transform(dest=observed_box_corner_coordinates,
                                         src=np.array(target_box_corner_coordinates),
                                         )
    new_track_x = []
    new_track_y = []
    for x, y in zip(Xin, Yin):
        inverse_mapped = p.inverse([x, y])[0]
        new_track_x.append(inverse_mapped[0])
        new_track_y.append(inverse_mapped[1])
    return new_track_x, new_track_y
