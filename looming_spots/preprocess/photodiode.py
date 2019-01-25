import os

import numpy as np
import pims
import scipy.signal

from looming_spots import exceptions
from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import convert_videos

from looming_spots.db.constants import FRAME_RATE


def get_manual_looms_raw(directory):
    loom_idx, _ = get_loom_idx_from_raw(directory)
    return get_manual_looms(loom_idx)


def get_manual_looms_from_metadata(directory):
    loom_idx = experiment_metadata.get_loom_idx(directory)
    return get_manual_looms(loom_idx)


def get_test_loom_idx(loom_idx,  n_looms_per_stimulus=5):  #WARNING: THIS DOES NOT DO WHAT THE USER EXPECTS
    if contains_habituation(loom_idx):
        loom_burst_onsets = np.diff(loom_idx[:: n_looms_per_stimulus])
        min_ili = min(loom_burst_onsets)
        print('min_ili: {}'.format(min_ili))
        test_loom_idx = np.where(loom_burst_onsets > min_ili + 200)[0] + 1
        return test_loom_idx*n_looms_per_stimulus


def get_habituation_loom_idx(loom_idx,  n_looms_per_stimulus=5):
    if contains_habituation(loom_idx):
        loom_burst_onsets = np.diff(loom_idx[::n_looms_per_stimulus])
        min_ili = min(loom_burst_onsets)
        habituation_loom_idx = np.where(loom_burst_onsets < min_ili + 25)[0]  # FIXME: this value is chosen for.. reasons
        habituation_loom_idx = np.concatenate([habituation_loom_idx, [max(habituation_loom_idx)+1]])  # adds last loom as ILI will always be bigger
        return loom_idx[habituation_loom_idx*n_looms_per_stimulus]


def get_habituation_start(loom_idx,  n_looms_per_stimulus=5):
    return get_habituation_loom_idx(loom_idx, n_looms_per_stimulus)[0]


def is_pre_test(loom_idx):
    test_loom_idx = get_test_loom_idx(loom_idx)
    if 0 in test_loom_idx:
        return True
    return False


def is_post_test(loom_idx):
    test_loom_idx = get_test_loom_idx(loom_idx)
    if len(test_loom_idx) in test_loom_idx:
        return True
    return False


def contains_habituation(loom_idx, n_looms_per_stimulus=5):
    ili = np.diff(np.diff(loom_idx[::n_looms_per_stimulus]))
    if np.count_nonzero([np.abs(x) < 5 for x in ili]) >= 3:
        return True
    return False


def get_manual_looms(loom_idx, n_looms_per_stimulus=5):
    if not contains_habituation(loom_idx, n_looms_per_stimulus):
        return loom_idx[::n_looms_per_stimulus]
    else:
        test_loom_idx = get_test_loom_idx(loom_idx,  n_looms_per_stimulus)
        return loom_idx[test_loom_idx]   #loom_idx[test_loom_idx[1:]]


def check_n_manual_looms(n_manual_looms, first_loom_idx, n_looms_per_stimulus):
    remainder = n_manual_looms % 5
    if remainder != 0:
        print("expected looms to be in multiple of: {}, got remainder: {}, skipping".format(n_looms_per_stimulus,
                                                                                            remainder))
        return
    manual_looms = np.arange(first_loom_idx, first_loom_idx+n_manual_looms, 5)

    if len(manual_looms) > 20:
        print('way too many stimuli to be correct: {}, skipping'.format(len(manual_looms)))
        return
    return manual_looms


def downsample_ai_bin(directory):
    pd_trace = load_pd_on_clock_ups(directory)
    savepath = os.path.join(directory, 'test.bin')
    np.save(savepath, pd_trace[::3])


def load_pd_on_clock_ups(directory, pd_threshold=2.5):
    if 'AI_corrected.npy' in os.listdir(directory):
        print('loading corrected/downsampled ai')
        path = os.path.join(directory, 'AI_corrected.npy')
        downsampled_processed_ai = np.load(path)
        return downsampled_processed_ai
    else:
        pd, clock, auditory = load_pd_and_clock_raw(directory)
        clock_ups = get_pd_clock_ups(clock, pd_threshold)
        print('number of clock ups found: {}'.format(len(clock_ups)))
        return pd[clock_ups]


def load_auditory_on_clock_ups(directory, pd_threshold=2.5):
    pd, clock, auditory = load_pd_and_clock_raw(directory)
    clock_ups = get_pd_clock_ups(clock, pd_threshold)
    print('number of clock ups found: {}'.format(len(clock_ups)))
    return auditory[clock_ups]


def manually_correct_ai(directory, start, end):
    ai = load_pd_on_clock_ups(directory)
    ai[start:end] = np.median(ai)
    save_path = os.path.join(directory, 'AI_corrected')
    np.save(save_path, ai)


def auto_fix_ai(directory, n_samples_to_replace=500):
    ai = load_pd_on_clock_ups(directory)
    screen_off_locs = np.where(ai < 0.02)[0]  # TODO: remove hard var

    if len(screen_off_locs) == 0:
        return

    start = screen_off_locs[0]
    end = start + n_samples_to_replace
    ai[start:end] = np.median(ai)
    save_path = os.path.join(directory, 'AI_corrected')
    np.save(save_path, ai)
    auto_fix_ai(directory, n_samples_to_replace=n_samples_to_replace)


def get_nearest_clock_up(raw_pd_value, clock_ups_pd):
    from bisect import bisect_left
    insertion_point = bisect_left(clock_ups_pd, raw_pd_value)
    difference_left = raw_pd_value - clock_ups_pd[insertion_point - 1]
    difference_right = raw_pd_value - clock_ups_pd[insertion_point]

    increment = 0 if difference_right < difference_left else -1
    nearest_clock_up_idx = insertion_point + increment
    distance_from_clock_up = difference_left if abs(difference_left) < abs(difference_right) else difference_right

    return nearest_clock_up_idx, distance_from_clock_up


def get_loom_idx_on_probe(directory, path_to_traces, upsample_factor=3):
    pd, clock = load_pd_and_clock_raw(directory)
    raw_pd_ups, raw_pd_downs = find_pd_threshold_crossings(pd)
    clock_ups_pd = get_pd_clock_ups(clock, 2.5)
    clock_ups_probe = get_probe_clock_ups(path_to_traces)

    stimulus_onsets_on_probe = []
    for pd_crossing in raw_pd_ups:
        nearest_clock_up_idx, distance_from_clock_up = get_nearest_clock_up(pd_crossing, clock_ups_pd)
        probe_nth_clock_up_sample = clock_ups_probe[nearest_clock_up_idx]
        new_stimulus_idx = probe_nth_clock_up_sample + distance_from_clock_up * upsample_factor
        print(probe_nth_clock_up_sample, distance_from_clock_up*upsample_factor)
        stimulus_onsets_on_probe.append(new_stimulus_idx - 5*30000)  # FIXME: shifts region to include baseline

    return stimulus_onsets_on_probe


def get_probe_clock_ups(path_to_traces, n_chan=385):
    print('getting clock ups from probe trace... this can take a while')
    clock_input = get_clock_from_traces(path_to_traces, n_chan=n_chan)
    clock_ups = np.where(np.diff(clock_input) == 1)[0]
    return clock_ups


def get_clock_from_traces(path_to_traces, n_chan=385):
    data = np.memmap(path_to_traces, dtype=np.int16)
    if data.shape[0] % n_chan != 0:
        raise 'data of length {} cannot be reshaped into {} channels'.format(data.shape[0], n_chan)
    shape = (int(data.shape[0] / n_chan), n_chan)
    shaped_data = np.memmap(path_to_traces, shape=shape, dtype=np.int16)
    return shaped_data[:, -1]


def get_pd_clock_ups(clock, pd_threshold=2.5):
    clock_on = (clock > pd_threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    return clock_ups


def load_pd_and_clock_raw(directory, auditory_present=True):
    path = os.path.join(directory, 'AI.bin')
    raw_ai = np.fromfile(path, dtype='double')

    if auditory_present:
        raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 3), 3)
        pd = raw_ai[:, 0]
        clock = raw_ai[:, 1]
        auditory = raw_ai[:, 2]
        return pd, clock, auditory

    raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 2), 2)
    pd = raw_ai[:, 0]
    clock = raw_ai[:, 1]
    return pd, clock, []  #FIXME: hack


def get_loom_idx_from_raw(directory, save=True):  # TODO: save npy file instead
    convert_videos.compare_pd_and_video(directory)
    ai = load_pd_on_clock_ups(directory)
    loom_starts, loom_ends = find_pd_threshold_crossings(ai)
    if len(loom_starts) % 5 != 0:
        print(directory, len(loom_starts))
        #auto_fix_ai(directory)
        raise LoomNumberError(Exception)
    dest = os.path.join(directory, 'loom_starts.npy')
    if save:
        np.save(dest, loom_starts)
    return loom_starts, loom_ends


def find_pd_threshold_crossings(ai):

    filtered_pd = filter_pd(ai)

    if not (filtered_pd > 0.4).any():
        return [], []

    threshold = np.median(filtered_pd) + np.nanstd(filtered_pd)*3
    print('threshold: {}'.format(threshold))
    loom_on = (filtered_pd > threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    loom_downs = np.diff(loom_on) == -1
    loom_ends = np.where(loom_downs)[0]
    return loom_starts, loom_ends


def filter_pd(pd_trace, fs=10000):
    b1, a1 = scipy.signal.butter(3, 1000./fs*2., 'low',)
    pd_trace = scipy.signal.filtfilt(b1, a1, pd_trace)
    return pd_trace


def get_fpath(directory, extension):
    for item in os.listdir(directory):
        if extension in item:
            return os.path.join(directory, item)

    raise exceptions.FileNotPresentError('there is no file with extension: {}'
                                         ' in directory {}'.format(extension, directory))


def get_pd_from_video(directory, start, end, video_name='camera.mp4'):
    path = os.path.join(directory, video_name)
    video = pims.Video(path)
    video = video[start:end]
    return np.mean(video, axis=(1, 2, 3))


def get_inter_loom_interval(loom_idx):
    return (int(loom_idx[5])-int(loom_idx[4]))/FRAME_RATE


class LoomNumberError(Exception):
    pass
