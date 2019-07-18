import os

import numpy as np
import pims
import scipy.signal

from datetime import datetime

from looming_spots import exceptions
from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import convert_videos

from looming_spots.db.constants import FRAME_RATE, AUDITORY_STIMULUS_CHANNEL_ADDED_DATE
from nptdms import TdmsFile


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


def get_habituation_idx(idx, n_looms_per_stimulus=5):
    if contains_habituation(idx, n_looms_per_stimulus):
        onsets_diff = np.diff(idx[::n_looms_per_stimulus])
        min_ili = min(onsets_diff)
        habituation_loom_idx = np.where(onsets_diff < min_ili + 25)[0]  # FIXME: this value is chosen for.. reasons
        habituation_loom_idx = np.concatenate([habituation_loom_idx, [max(habituation_loom_idx)+1]])  # adds last loom as ILI will always be bigger
        return idx[habituation_loom_idx*n_looms_per_stimulus]


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
    if not loom_idx.shape:
        return False
    ili = np.diff(np.diff(loom_idx[::n_looms_per_stimulus]))
    if np.count_nonzero([np.abs(x) < 5 for x in ili]) >= 3:
        return True
    return False


def get_manual_looms(loom_idx, n_looms_per_stimulus=5):
    if not contains_habituation(loom_idx, n_looms_per_stimulus):
        return loom_idx[::n_looms_per_stimulus]
    else:
        test_loom_idx = get_test_loom_idx(loom_idx,  n_looms_per_stimulus)
        return loom_idx[test_loom_idx]


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
        clock_ups = get_clock_ups(clock, pd_threshold)
        print('number of clock ups found: {}'.format(len(clock_ups)))
        return pd[clock_ups]


def load_auditory_on_clock_ups(directory, pd_threshold=2.5):
    pd, clock, auditory = load_pd_and_clock_raw(directory)
    clock_ups = get_clock_ups(clock, pd_threshold)
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
    clock_ups_pd = get_clock_ups(clock, 2.5)
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


def get_clock_ups(clock, threshold=2.5):
    clock_on = (clock > threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    return clock_ups


def load_all_channels_raw(directory):
    if 'AI.tdms' in os.listdir(directory):
        path = os.path.join(directory, 'AI.tdms')
        tdms_file = TdmsFile(path)
        all_channels = tdms_file.group_channels('acq_task')
        pd, clock, auditory, pmt, led211, led531 = (c.data for c in all_channels)
        return pd, clock, auditory, pmt, led211, led531


def load_pd_and_clock_raw(directory):
    if 'AI.tdms' in os.listdir(directory):
        path = os.path.join(directory, 'AI.tdms')
        tdms_file = TdmsFile(path)
        all_channels = tdms_file.group_channels('acq_task')
        pd, clock, auditory, pmt, led211, led531 = (c.data for c in all_channels)
        return pd, clock, auditory

    else:
        path = os.path.join(directory, 'AI.bin')
        raw_ai = np.fromfile(path, dtype='double')

    recording_date = datetime.strptime(os.path.split(directory)[-1], '%Y%m%d_%H_%M_%S')

    if recording_date > AUDITORY_STIMULUS_CHANNEL_ADDED_DATE:
        raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 3), 3)
        pd = raw_ai[:, 0]
        clock = raw_ai[:, 1]
        auditory = raw_ai[:, 2]
        return pd, clock, auditory

    raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 2), 2)
    pd = raw_ai[:, 0]
    clock = raw_ai[:, 1]
    return pd, clock, []  # FIXME: hack


def get_loom_idx_from_raw(directory, save=True):  # TODO: save npy file instead
    try:
        #convert_videos.compare_pd_and_video(directory)
        ai = load_pd_on_clock_ups(directory)
        aud = load_auditory_on_clock_ups(directory)
        loom_starts, loom_ends = find_pd_threshold_crossings(ai)
    except convert_videos.NoPdError as e:
        loom_starts = []
        loom_ends = []

    if len(loom_starts) % 5 != 0 and (aud < 1).all():
        print(directory, len(loom_starts))
        #auto_fix_ai(directory)
        raise LoomNumberError(Exception)

    dest = os.path.join(directory, 'loom_starts.npy')
    if save:
        np.save(dest, loom_starts)
    return loom_starts, loom_ends


def find_pd_threshold_crossings(ai, threshold=0.4):

    filtered_pd = filter_pd(ai)

    if not (filtered_pd > threshold).any():
        return [], []

    threshold = np.median(filtered_pd) + np.nanstd(filtered_pd)*3  #3
    print('threshold: {}'.format(threshold))
    loom_on = (filtered_pd > threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    loom_downs = np.diff(loom_on) == -1
    loom_ends = np.where(loom_downs)[0]
    return loom_starts, loom_ends


def filter_pd(pd_trace, fs=10000):  # 10000
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


def get_auditory_onsets_from_analog_input(directory, save=True):
    aud = load_auditory_on_clock_ups(directory)
    aud -= np.mean(aud)

    if not (aud > 0.7).any():
        auditory_onsets = []
    else:
        aud_on = aud < -(2 * np.std(aud[:200]))
        aud_diff = np.diff(np.where(aud_on)[0])
        auditory_onsets = np.concatenate([[np.where(aud_on)[0][0]], np.array(np.where(aud_on)[0])[1:][aud_diff > 1000]])

        # auditory_on = np.where(aud < -1.05)[0]
        # onsets = list(auditory_on[np.where(np.diff(auditory_on) > 1000)[0] + 1])
        # auditory_onsets = [auditory_on[0]] + onsets

    # aud_diff = np.diff(gaussian_filter(abs(np.diff(aud)), 2) )
    # if np.count_nonzero(aud_diff) == 0:
    #     auditory_onsets = []
    # else:
    #     auditory_onsets = np.where(aud_diff)[0][::2]  # > 0.15

    dest = os.path.join(directory, 'auditory_starts.npy')

    if save:
        np.save(dest, auditory_onsets)
    return auditory_onsets


def get_visual_onsets_from_analog_input(directory):
    ai = load_pd_on_clock_ups(directory)
    loom_starts, loom_ends = find_pd_threshold_crossings(ai)
    return loom_starts


def find_nearest_pd_up_from_frame_number(directory, frame_number, sampling_rate=10000):
    pd, _, _ = load_pd_and_clock_raw(directory)
    raw_pd_ups, raw_pd_downs = find_pd_threshold_crossings(pd)
    start_p = frame_number * sampling_rate/FRAME_RATE
    return raw_pd_ups[np.argmin(abs(raw_pd_ups-start_p))]


class LoomNumberError(Exception):
    pass


def get_calibration_starts_ends(directory):
    pd = load_pd_on_clock_ups(directory)
    starts, ends = find_pd_calibration_crossings(pd, 0.9)
    return starts, ends


def find_pd_calibration_crossings(ai, threshold=0.4):

    filtered_pd = filter_pd(ai)

    print('threshold: {}'.format(threshold))
    loom_on = (filtered_pd < threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    loom_downs = np.diff(loom_on) == -1
    loom_ends = np.where(loom_downs)[0]
    return loom_starts, loom_ends