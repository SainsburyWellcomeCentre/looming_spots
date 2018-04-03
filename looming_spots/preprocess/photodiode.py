import os

import numpy as np
import scipy.signal

from looming_spots import exceptions
from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import convert_videos

N_HABITUATION_LOOMS = 120
# PATH_TO_TRACES = '/home/slenzi/spine_shares/loomer/20180311_20_48_18/probe_data_g0_t0.imec.ap.bin'
FRAME_RATE = 30


def get_manual_looms_raw(directory):
    loom_idx = get_loom_idx_from_raw(directory)
    return get_manual_looms(loom_idx)


def get_manual_looms_from_metadata(directory):
    loom_idx = experiment_metadata.get_loom_idx(directory)
    return get_manual_looms(loom_idx)


def get_manual_looms(loom_idx, n_looms_per_stimulus=5):

    if isinstance(loom_idx, list):
        loom_idx = np.array(loom_idx).astype(int)

    ilis = np.roll(loom_idx, -1) - loom_idx

    if len(ilis) > N_HABITUATION_LOOMS:
        first_loom_idx = N_HABITUATION_LOOMS
        n_manual_looms = len(ilis) - N_HABITUATION_LOOMS

    elif len(ilis) == N_HABITUATION_LOOMS:
        print('HABITUATION ONLY, {} looms detected'.format(len(ilis)))
        return

    else:
        first_loom_idx = 0
        n_manual_looms = len(ilis)

    manual_looms = check_n_manual_looms(n_manual_looms, first_loom_idx,n_looms_per_stimulus)
    if manual_looms is not None:
        return loom_idx[manual_looms]


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
        pd, clock = load_pd_and_clock_raw(directory)
        clock_ups = get_pd_clock_ups(clock, pd_threshold)
        print('number of clock ups found: {}'.format(len(clock_ups)))
        return pd[clock_ups]


def manually_correct_ai(directory, start=0, end=None):
    ai = load_pd_on_clock_ups(directory)
    ai[:start] = np.median(ai)
    if end is None:
        end = len(ai)
    ai[end:] = np.median(ai)
    save_path = os.path.join(directory, 'AI_corrected')
    np.save(save_path, ai)


def get_nearest_clock_up(raw_pd_value, clock_ups_pd):
    from bisect import bisect_left
    insertion_point = bisect_left(clock_ups_pd, raw_pd_value)
    difference_left = raw_pd_value - clock_ups_pd[insertion_point - 1]
    difference_right = raw_pd_value - clock_ups_pd[insertion_point]

    increment = 0 if difference_right < difference_left else -1
    nearest_clock_up_idx = insertion_point + increment
    distance_from_clock_up = difference_left if abs(difference_left) < abs(difference_right) else difference_right

    return nearest_clock_up_idx, distance_from_clock_up


def get_loom_idx_on_probe(directory, upsample_factor=3):
    clock, pd = load_pd_and_clock_raw(directory)
    raw_pd_crossings = find_pd_threshold_crossings(pd)
    clock_ups_pd = get_pd_clock_ups(clock, 2.5)
    clock_ups_probe = get_probe_clock_ups()

    stimulus_onsets_on_probe = []
    for pd_crossing in raw_pd_crossings:
        nearest_clock_up_idx, distance_from_clock_up = get_nearest_clock_up(pd_crossing, clock_ups_pd)
        probe_nth_clock_up_sample = clock_ups_probe[nearest_clock_up_idx]
        new_stimulus_idx = probe_nth_clock_up_sample + distance_from_clock_up * upsample_factor
        print(probe_nth_clock_up_sample, distance_from_clock_up*upsample_factor)
        stimulus_onsets_on_probe.append(new_stimulus_idx - 5*30000)  # FIXME: shifts region to include baseline

    return stimulus_onsets_on_probe


def get_probe_clock_ups(path_to_traces, n_chan=385):
    clock_input = get_clock_from_traces(path_to_traces, n_chan=n_chan)
    clock_ups = np.where(np.diff(clock_input) == 1)[0]
    print('getting clock ups from probe trace... this can take a while')
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


def load_pd_and_clock_raw(directory):
    path = os.path.join(directory, 'AI.bin')
    raw_ai = np.fromfile(path, dtype='double')
    raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 2), 2)
    pd = raw_ai[:, 0]
    clock = raw_ai[:, 1]
    return pd, clock


def get_loom_idx_from_raw(directory):  # TODO: save npy file instead
    convert_videos.compare_pd_and_video(directory)
    ai = load_pd_on_clock_ups(directory)
    loom_starts = find_pd_threshold_crossings(ai)
    if len(loom_starts) % 5 != 0:
        raise LoomNumberError(Exception)
    return loom_starts


def find_pd_threshold_crossings(ai):
    filtered_pd = filter_pd(ai)
    threshold = np.nanstd(filtered_pd)*3
    loom_on = (filtered_pd > threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    return loom_starts


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


def get_ILI(loom_idx):
    return (int(loom_idx[5])-int(loom_idx[4]))/FRAME_RATE


class LoomNumberError(Exception):
    pass