import os

import numpy as np
import pims
import scipy.signal

import looming_spots.util.video_processing
from looming_spots.db.constants import FRAME_RATE

from looming_spots.preprocess.io import (
    load_pd_on_clock_ups,
    load_auditory_on_clock_ups,
    load_pd_and_clock_raw,
)
from looming_spots.exceptions import PdTooShortError


def get_loom_idx_from_raw(directory, save=True):  # TODO: save npy file instead
    try:
        # convert_videos.compare_pd_and_video(directory)
        ai = load_pd_on_clock_ups(directory)
        print(len(ai))
        aud = load_auditory_on_clock_ups(directory)
        loom_starts, loom_ends = find_pd_threshold_crossings(ai)
    except looming_spots.util.video_processing.NoPdError as e:
        loom_starts = []
        loom_ends = []

    except PdTooShortError as e:
        loom_starts = []
        loom_ends = []

    if len(loom_starts) % 5 != 0 and (aud < 1).all():
        print(directory, len(loom_starts))
        # auto_fix_ai(directory)
        # raise LoomNumberError(Exception)

    dest = os.path.join(directory, "loom_starts.npy")
    if save:
        np.save(dest, loom_starts)
    return loom_starts, loom_ends


def get_test_loom_idx(
    loom_idx, n_looms_per_stimulus=5
):  # WARNING: THIS DOES NOT DO WHAT THE USER EXPECTS
    if contains_habituation(loom_idx):
        loom_burst_onsets = np.diff(loom_idx[::n_looms_per_stimulus])
        min_ili = min(loom_burst_onsets)
        print("min_ili: {min_ili}")
        test_loom_idx = np.where(loom_burst_onsets > min_ili + 200)[0] + 1
        return test_loom_idx * n_looms_per_stimulus


def get_habituation_loom_idx(loom_idx, n_looms_per_stimulus=5):
    if contains_habituation(loom_idx):
        loom_burst_onsets = np.diff(loom_idx[::n_looms_per_stimulus])
        min_ili = min(loom_burst_onsets)
        habituation_loom_idx = np.where(loom_burst_onsets < min_ili + 25)[
            0
        ]  # FIXME: this value is chosen for.. reasons
        habituation_loom_idx = np.concatenate(
            [habituation_loom_idx, [max(habituation_loom_idx) + 1]]
        )  # adds last loom as ILI will always be bigger
        return loom_idx[habituation_loom_idx * n_looms_per_stimulus]


def get_habituation_idx(idx, n_looms_per_stimulus=5):
    if contains_habituation(idx, n_looms_per_stimulus):
        onsets_diff = np.diff(idx[::n_looms_per_stimulus])
        min_ili = min(onsets_diff)
        habituation_loom_idx = np.where(onsets_diff < min_ili + 25)[
            0
        ]  # FIXME: this value is chosen for.. reasons
        habituation_loom_idx = np.concatenate(
            [habituation_loom_idx, [max(habituation_loom_idx) + 1]]
        )  # adds last loom as ILI will always be bigger
        return idx[habituation_loom_idx * n_looms_per_stimulus]


def get_habituation_start(loom_idx, n_looms_per_stimulus=5):
    return get_habituation_loom_idx(loom_idx, n_looms_per_stimulus)[0]


def contains_habituation(loom_idx, n_looms_per_stimulus=5):
    if not loom_idx.shape:
        return False
    ili = np.diff(np.diff(loom_idx[::n_looms_per_stimulus]))
    if np.count_nonzero([np.abs(x) < 5 for x in ili]) >= 3:
        return True
    return False


def get_nearest_clock_up(raw_pd_value, clock_ups_pd):
    from bisect import bisect_left

    insertion_point = bisect_left(clock_ups_pd, raw_pd_value)
    difference_left = raw_pd_value - clock_ups_pd[insertion_point - 1]
    difference_right = raw_pd_value - clock_ups_pd[insertion_point]

    increment = 0 if difference_right < difference_left else -1
    nearest_clock_up_idx = insertion_point + increment
    distance_from_clock_up = (
        difference_left
        if abs(difference_left) < abs(difference_right)
        else difference_right
    )

    return nearest_clock_up_idx, distance_from_clock_up


def find_pd_threshold_crossings(ai, threshold=0.4):

    filtered_pd = filter_pd(ai)

    if not (filtered_pd > threshold).any():
        return [], []

    threshold = np.median(filtered_pd) + np.nanstd(filtered_pd) * 3  # 3
    print(f"threshold: {threshold}")
    loom_on = (filtered_pd > threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    loom_downs = np.diff(loom_on) == -1
    loom_ends = np.where(loom_downs)[0]
    return loom_starts, loom_ends


def filter_pd(pd_trace, fs=10000):  # 10000
    b1, a1 = scipy.signal.butter(3, 1000.0 / fs * 2.0, "low")
    pd_trace = scipy.signal.filtfilt(b1, a1, pd_trace)
    return pd_trace


def get_pd_from_video(directory, start, end, video_name="camera.mp4"):
    path = os.path.join(directory, video_name)
    video = pims.Video(path)
    video = video[start:end]
    return np.mean(video, axis=(1, 2, 3))


def get_inter_loom_interval(loom_idx):
    return (int(loom_idx[5]) - int(loom_idx[4])) / FRAME_RATE


def get_auditory_onsets_from_analog_input(directory, save=True):
    aud = load_auditory_on_clock_ups(directory)
    aud -= np.mean(aud)

    if not (aud > 0.7).any():
        auditory_onsets = []
    else:
        aud_on = aud < -(2 * np.std(aud[:200]))
        aud_diff = np.diff(np.where(aud_on)[0])
        auditory_onsets = np.concatenate(
            [
                [np.where(aud_on)[0][0]],
                np.array(np.where(aud_on)[0])[1:][aud_diff > 1000],
            ]
        )
    dest = os.path.join(directory, "auditory_starts.npy")

    if save:
        np.save(dest, auditory_onsets)
    return auditory_onsets


def get_visual_onsets_from_analog_input(directory):
    ai = load_pd_on_clock_ups(directory)
    loom_starts, loom_ends = find_pd_threshold_crossings(ai)
    return loom_starts


def get_manual_looms(loom_idx, n_looms_per_stimulus=5):
    if not contains_habituation(loom_idx, n_looms_per_stimulus):
        return loom_idx[::n_looms_per_stimulus]
    else:
        test_loom_idx = get_test_loom_idx(loom_idx, n_looms_per_stimulus)
        return loom_idx[test_loom_idx]


def get_manual_looms_raw(directory):
    loom_idx, _ = get_loom_idx_from_raw(directory)
    return get_manual_looms(loom_idx)


def find_nearest_pd_up_from_frame_number(
    directory, frame_number, sampling_rate=10000
):
    pd, _, _ = load_pd_and_clock_raw(directory)
    raw_pd_ups, raw_pd_downs = find_pd_threshold_crossings(pd)
    start_p = frame_number * sampling_rate / FRAME_RATE
    return raw_pd_ups[np.argmin(abs(raw_pd_ups - start_p))]


