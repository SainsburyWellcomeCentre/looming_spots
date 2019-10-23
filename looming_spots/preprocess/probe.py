import numpy as np

from looming_spots.preprocess.photodiode import find_pd_threshold_crossings, get_nearest_clock_up
from looming_spots.preprocess.io import load_pd_and_clock_raw, get_clock_ups


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