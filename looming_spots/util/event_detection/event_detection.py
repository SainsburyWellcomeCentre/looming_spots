import numpy as np
from scipy import signal

from margrie_libs.signal_processing.filters import box_smooth, diff
from margrie_libs.signal_processing.measures import get_sd
from margrie_libs.signal_processing.thresholding import (
    find_levels,
    find_level_increase,
)
from margrie_libs.utils.print_utils import dprint


def _detect_events(
    trace,
    n_pnts_bsl,
    n_pnts_peak,
    n_pnts_rise_t,
    threshold,
    min_n_pnts_from_start=10,
):
    peak_wave = box_smooth(trace, n_pnts_peak)
    bsl_wave = box_smooth(trace, n_pnts_bsl)
    delta_wave = diff(peak_wave, bsl_wave, n_pnts_rise_t)
    events = find_levels(delta_wave, -threshold)  # TODO: check sign

    if len(events) > 0:
        if (
            events[0] < min_n_pnts_from_start
        ):  # Remove first one if too close to start
            events = events[1:]
    else:
        raise StopIteration(
            "No events found with parameters: {}, {}, {}, {}".format(
                n_pnts_bsl, n_pnts_peak, n_pnts_rise_t, threshold
            )
        )
    return events


def find_events_peaks(trace, events_pos, time_range):
    peaks_pos = events_pos.copy()
    for i, event_pos in enumerate(events_pos):
        event_trace = trace[event_pos : (event_pos + time_range)]
        max_locs = np.where(event_trace == event_trace.max())[0]
        if max_locs.size:
            peak_loc = max_locs[0]  # TODO: do for positive and negative
            peaks_pos[i] = event_pos + peak_loc
        else:
            peaks_pos[i] = event_pos  # FIXME: No event max found use risetime
    return peaks_pos


def find_events_half_rises(trace, events_pos, peaks_pos):
    # FIXME: seems to be imprecise
    half_rises = events_pos.copy()
    for i, bounds in enumerate(zip(events_pos, peaks_pos)):
        try:  # FIXME: better error handling
            event_start, event_end = bounds
            event_rise = trace[event_start:event_end]
            event_ampl = event_rise[-1] - event_rise[0]
            half_rise_loc = find_level_increase(
                event_rise, event_rise[0] + (event_ampl / 2.0)
            )
        except IndexError:
            half_rise_loc = 0  # FIXME: better error handling
        half_rises[i] = event_start + half_rise_loc
    return half_rises


def find_events_amplitudes(trace, events_pos, peaks_pos):
    ampls = np.empty(len(events_pos), dtype=np.float64)
    for i, bounds in enumerate(zip(events_pos, peaks_pos)):
        event_start, event_end = bounds
        ampls[i] = (
            trace[event_end] - trace[event_start]
        )  # TODO: do for both pos and neg
    return ampls


def filter_with_sd(peak_ampls, sd, n_sds):
    threshold = n_sds * sd
    events = np.where(peak_ampls > threshold)[0]
    return np.array(events, dtype=np.int64)


def remove_refractory_period(
    trace,
    delta_wave,
    events_pos,
    half_rises,
    peaks_pos,
    n_pnts_refractory_period,
):
    trace = trace[n_pnts_refractory_period:]
    delta_wave = delta_wave[n_pnts_refractory_period:]
    events_pos -= n_pnts_refractory_period
    peaks_pos -= n_pnts_refractory_period
    half_rises -= n_pnts_refractory_period
    # Remove events that where during refractory period
    old_events_pos = events_pos.copy()
    events_pos = events_pos[old_events_pos >= 0]
    peaks_pos = peaks_pos[old_events_pos >= 0]
    half_rises = half_rises[old_events_pos >= 0]
    return trace, delta_wave, events_pos, half_rises, peaks_pos


def add_refractory_period(n_pnts_bsl, n_pnts_peak, n_pnts_rise_t, trace):
    n_pnts_refractory_period = n_pnts_bsl + n_pnts_peak + n_pnts_rise_t
    bsl = np.repeat(trace[0], n_pnts_refractory_period)
    trace = np.hstack((bsl, trace))
    return trace, n_pnts_refractory_period


def filter_duplicates(events_pos, minimum_time_delta):
    """
    When several events have similar peak_pos (within minimum_time_delta) keep
    only the first one
    """
    dprint(
        "{} events before remove duplicates, {}".format(
            len(events_pos), events_pos
        )
    )
    good_indices = []
    try:
        reversed_idx = reversed(list(range(len(events_pos))))
    except TypeError:
        print(type(events_pos), events_pos)
        raise
    for i in reversed_idx:
        if i == 0:  # Last event
            good_indices.append(i)
        else:
            if (events_pos[i] - events_pos[i - 1]) > minimum_time_delta:
                good_indices.append(i)
    dprint(
        "{} events after remove duplicates, {}".format(
            len(good_indices), events_pos[good_indices]
        )
    )
    return np.array(list(reversed(good_indices)), dtype=np.int64)


def _filter_event_start_duplicates(events_pos, minimum_time_delta):
    good_events = filter_duplicates(events_pos, minimum_time_delta)
    if len(good_events):
        events_pos = events_pos[good_events]
    else:
        print(
            "0 events after filtering by baseline time, before: {}".format(
                events_pos
            )
        )
    return events_pos


def _filter_peak_duplicates(events_pos, peaks_pos, minimum_time_delta):
    # Remove events with same peak time
    good_events = filter_duplicates(peaks_pos, minimum_time_delta)
    if len(good_events):
        events_pos = events_pos[good_events]
        peaks_pos = peaks_pos[good_events]
    else:
        dprint(
            "0 events after filtering by peak time. Before, bsl: {}, peak: {}".format(
                events_pos, peaks_pos
            )
        )
    return events_pos, peaks_pos


def _sd_filter(
    events_pos, n_pnts_high_pass_filter, n_sds, peak_ampls, peaks_pos, trace
):
    n_events_before_sd = len(events_pos)
    sd = get_sd(trace, n_pnts_high_pass_filter)
    good_events = filter_with_sd(peak_ampls, sd, n_sds)
    if len(good_events):
        events_pos = events_pos[good_events]
        n_events_after_sd = len(events_pos)
        dprint(
            "SD removed {} events ({} to {})".format(
                n_events_before_sd - n_events_after_sd,
                n_events_before_sd,
                n_events_after_sd,
            )
        )
        peaks_pos = peaks_pos[good_events]
        peak_ampls = peak_ampls[good_events]
    else:
        dprint("0 events after filtering by SD")
    return events_pos, peak_ampls, peaks_pos


def detect_trace(
    trace,
    threshold,
    n_pnts_bsl,
    n_pnts_peak,
    n_pnts_rise_t,
    n_pnts_for_peak_detection,
    n_sds,
    n_pnts_high_pass_filter,
    median_kernel_size=9,
    minimum_time_delta=60,
):  # FIXME: some \n left in trace

    trace, n_pnts_refractory_period = add_refractory_period(
        n_pnts_bsl, n_pnts_peak, n_pnts_rise_t, trace
    )  # FIXME: may not be correct NAME

    trace = signal.medfilt(trace, median_kernel_size)
    delta_wave = diff(
        box_smooth(trace, n_pnts_peak),
        box_smooth(trace, n_pnts_bsl),
        n_pnts_rise_t,
    )
    default_result = trace, delta_wave, False, False, False, False

    # DETECT
    events_start = _detect_events(
        trace, n_pnts_bsl, n_pnts_peak, n_pnts_rise_t, threshold
    )

    # FILTER
    events_start = _filter_event_start_duplicates(
        events_start, minimum_time_delta
    )
    if not len(events_start):
        return default_result

    peaks_pos = find_events_peaks(
        trace, events_start, n_pnts_for_peak_detection
    )
    events_start, peaks_pos = _filter_peak_duplicates(
        events_start, peaks_pos, minimum_time_delta
    )
    if not len(events_start):
        return default_result

    peak_ampls = find_events_amplitudes(trace, events_start, peaks_pos)

    events_start, peak_ampls, peaks_pos = _sd_filter(
        events_start,
        n_pnts_high_pass_filter,
        n_sds,
        peak_ampls,
        peaks_pos,
        trace,
    )
    if not len(events_start):
        return default_result

    half_rises = find_events_half_rises(trace, events_start, peaks_pos)

    (
        trace,
        delta_wave,
        events_start,
        half_rises,
        peaks_pos,
    ) = remove_refractory_period(
        trace,
        delta_wave,
        events_start,
        half_rises,
        peaks_pos,
        n_pnts_refractory_period,
    )
    return trace, delta_wave, events_start, peaks_pos, half_rises, peak_ampls
