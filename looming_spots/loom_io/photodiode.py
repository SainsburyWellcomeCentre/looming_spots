import os

import numpy as np
import scipy.signal

from looming_spots.loom_io.load import load_all_channels_on_clock_ups
from looming_spots.exceptions import PdTooShortError


def get_test_looms_from_loom_idx(loom_idx, n_looms_per_stimulus=5):
    if not contains_lse(loom_idx, n_looms_per_stimulus):
        return loom_idx[::n_looms_per_stimulus]
    else:
        test_loom_idx = get_test_loom_idx(loom_idx, n_looms_per_stimulus)
        return loom_idx[test_loom_idx]


def get_test_loom_idx(loom_idx, n_looms_per_stimulus=5):
    if contains_lse(loom_idx):
        loom_burst_onsets = np.diff(loom_idx[::n_looms_per_stimulus])
        min_ili = min(loom_burst_onsets)
        print("min_ili: {min_ili}")
        test_loom_idx = np.where(loom_burst_onsets > min_ili + 200)[0] + 1
        return test_loom_idx * n_looms_per_stimulus


def get_loom_idx_from_photodiode_trace(directory, save=True):
    try:
        data = load_all_channels_on_clock_ups(directory)
        photodiode_trace = data["photodiode"]
        print(len(photodiode_trace))

        loom_starts, loom_ends = find_pd_threshold_crossings(photodiode_trace)

    except NoPdError as e:
        print(e)
        loom_starts = []
        loom_ends = []

    except PdTooShortError as e:
        print(e)
        loom_starts = []
        loom_ends = []

    dest = os.path.join(directory, "loom_starts.npy")
    if save:
        np.save(dest, loom_starts)
    return loom_starts, loom_ends


def get_lse_loom_idx(idx, n_looms_per_stimulus=5):
    if contains_lse(idx, n_looms_per_stimulus):
        onsets_diff = np.diff(idx[::n_looms_per_stimulus])
        min_ili = min(onsets_diff)
        loom_idx_lsie = np.where(onsets_diff < min_ili + 150)[0]
        loom_idx_lsie = np.concatenate(
            [loom_idx_lsie, [max(loom_idx_lsie) + 1]]
        )  # adds last loom as ILI will always be bigger
        return idx[loom_idx_lsie * n_looms_per_stimulus]


def get_lse_start(loom_idx, n_looms_per_stimulus=5):
    return get_lse_loom_idx(loom_idx, n_looms_per_stimulus)[0]


def contains_lse(loom_idx, n_looms_per_stimulus=5):
    if not loom_idx.shape:
        return False
    ili = np.diff(np.diff(loom_idx[::n_looms_per_stimulus]))
    if np.count_nonzero([np.abs(x) < 5 for x in ili]) >= 3:
        return True
    return False


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


def get_auditory_onsets_from_auditory_trace(directory, save=True):
    aud = load_all_channels_on_clock_ups(directory)["auditory_stimulus"]
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


class NoPdError(Exception):
    pass
