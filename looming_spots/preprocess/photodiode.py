import os

import numpy as np
import scipy.signal

from looming_spots import exceptions

N_HABITUATION_LOOMS = 120


def get_manual_looms(loom_idx, n_looms_per_stimulus=5):
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

    if len(manual_looms) > 5:
        print('way too many stimuli to be correct: {}, skipping'.format(len(manual_looms)))
        return
    return manual_looms


def load_pd_on_clock_ups(directory, pd_threshold=2.5):
    path = get_fpath(directory, '.bin')
    raw_ai = np.fromfile(path, dtype='double')
    raw_ai = raw_ai.reshape(int(raw_ai.shape[0]/2), 2)
    pd = raw_ai[:, 0]
    clock = raw_ai[:, 1]
    clock_on = (clock > pd_threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    print('number of clock ups found: {}'.format(len(clock_ups)))
    return pd[clock_ups]


def get_loom_idx_from_raw(directory):
    ai = load_pd_on_clock_ups(directory)
    filtered_pd = filter_pd(ai)
    loom_on = (filtered_pd > 1).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    return loom_starts


def get_loom_idx_from_raw_filtfilt(directory):
    ai = load_pd_on_clock_ups(directory)
    filtered_pd = filter_pd_twoway(ai)
    loom_on = (filtered_pd > 1).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    return loom_starts


def filter_pd(pd_trace, fs=10000):
    b1, a1 = scipy.signal.butter(3, 1000./fs*2., 'low',)
    pd_trace = scipy.signal.lfilter(b1, a1, pd_trace)
    return pd_trace


def filter_pd_twoway(pd_trace, fs=10000):
    b1, a1 = scipy.signal.butter(3, 1000./fs*2., 'low',)
    pd_trace = scipy.signal.filtfilt(b1, a1, pd_trace)
    return pd_trace


def get_fpath(directory, extension):
    for item in os.listdir(directory):
        if extension in item:
            return os.path.join(directory, item)

    raise exceptions.FileNotPresentError('there is no file with extension: {}'
                                              ' in directory {}'.format(extension, directory))
