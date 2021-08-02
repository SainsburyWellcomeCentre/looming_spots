import numpy as np
from looming_spots.db.constants import FRAME_RATE, N_SAMPLES_BEFORE
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
