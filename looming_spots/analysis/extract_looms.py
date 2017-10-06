import os
import numpy as np
import skvideo
from datetime import datetime
import scipy.signal
import skvideo.io


def is_datetime(folder_name):
    try:
        date_time = datetime.strptime(folder_name, '%Y%m%d_%H_%M_%S')
        print('string is in date_time format: {}'.format(date_time))
        return True
    except Exception as e:
        return False


def get_fpath(directory, extension):
    for item in os.listdir(directory):
        if extension in item:
            return os.path.join(directory, item)


def load_ai(directory, pd_threshold=2.5):
    path = get_fpath(directory, '.bin')
    raw = np.fromfile(path, dtype='double')
    raw_reshaped = raw.reshape(int(raw.shape[0]/2), 2)
    raw_ai = raw_reshaped[:, 0]
    raw_clock = raw_reshaped[:, 1]
    clock_on = (raw_clock > pd_threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    print('number of clock ups found: {}'.format(len(clock_ups)))
    return raw_ai[clock_ups]


def filter_raw_pd_trace(pd_trace):
    fs = 10000
    b1, a1 = scipy.signal.butter(3, 1000/fs*2, 'low')
    pd_trace = scipy.signal.lfilter(b1, a1, pd_trace)
    return pd_trace


def get_loom_idx(filtered_ai):
    loom_on = (filtered_ai > 1).astype(int)
    loom_ups = np.diff(loom_on) == 1
    # loom_downs = np.diff(loom_on) == -1
    loom_starts = np.where(loom_ups)[0]
    return loom_starts


def get_manual_looms(loom_idx, n_looms_per_stimulus=5, n_auto_looms = 120, ILI_ignore_n_samples=1300):
    ilis = np.roll(loom_idx, -1) - loom_idx
    if len(ilis) > n_auto_looms:
        first_loom_idx = n_auto_looms
        n_manual_looms = len(ilis)-n_auto_looms
    elif len(ilis) == n_auto_looms:
        print('HABITUATION ONLY, {} looms detected'.format(len(ilis)))
        return []
    else:
        first_loom_idx = 0
        n_manual_looms = len(ilis)

    remainder = n_manual_looms % 5
    if remainder != 0:
        print("expected looms to be in multiple of: {}, got remainder: {}, skipping".format(n_looms_per_stimulus,
                                                                                            remainder))
        return []
    manual_looms = np.arange(first_loom_idx, first_loom_idx+n_manual_looms, 5)
    if len(manual_looms) > 5:
        print('way too many stimuli to be correct: {}, skipping'.format(len(manual_looms)))
        return []
    return loom_idx[manual_looms]


def extract_loom_videos(directory, manual_loom_indices):
    for loom_number, loom_idx in enumerate(manual_loom_indices):
        extract_loom_video(directory, loom_idx, loom_number)


def extract_loom_video(directory, loom_start, loom_number, n_samples_before=200, n_samples_after=200, shape=(480, 640)):
    loom_video_path = os.path.join(directory, 'loom{}.h264'.format(loom_number))
    if os.path.isfile(loom_video_path):
        return
    video_path = get_fpath(directory, '.mp4')
    rdr = skvideo.io.vreader(video_path)
    loom_video = np.zeros((n_samples_before+n_samples_after+1, shape[0], shape[1], 3))
    a = 0
    for i, frame in enumerate(rdr):
        if (loom_start-n_samples_before) < i < (loom_start + n_samples_after):
            loom_video[a, :, :, :] = frame
            a += 1
    skvideo.io.vwrite(loom_video_path, loom_video)


def auto_extract_all(directory):
    ai = load_ai(directory)
    ai_filtered = filter_raw_pd_trace(ai)
    all_loom_idx = get_loom_idx(ai_filtered)
    manual_loom_indices = get_manual_looms(all_loom_idx)
    extract_loom_videos(directory, manual_loom_indices)
