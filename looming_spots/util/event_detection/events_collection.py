import operator
import warnings

import numpy as np
import pandas as pd


class EventsCollection(object):
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events[item]

    def __sub__(self, scalar):
        return EventsCollection([e - scalar for e in self])

    def to_df(self, trial_id=0):
        return pd.DataFrame(
            {
                "trial_id": [trial_id] * len(self),
                "start_t": self.start_times,
                "rise_t": self.rise_times,
                "peak_t": self.peak_times,
                "amplitude": self.amplitudes,
                "integral": self.integrals,
            }
        )

    def extend(self, events_collection):
        if events_collection.events:
            self.events.extend(events_collection.events)

    @staticmethod
    def from_concatenation_of_events_collections(events_collections):
        events = []
        for event_collection in events_collections:
            if event_collection is not None:
                events.extend(event_collection.events)
        return EventsCollection(events)

    def get_sorted(self, comparator=None):
        if comparator is None:
            return sorted(self)
        else:
            return sorted(self, key=operator.attrgetter(comparator))

    def in_point_range(self, range_start, range_end):
        return EventsCollection(
            [e for e in self if range_start <= e.peak_p < range_end]
        )

    def in_time_range(self, range_start, range_end):
        # TODO: use comparator instead
        return EventsCollection(
            [e for e in self if range_start <= e.peak_t < range_end]
        )

    def in_unordered_point_range(self, p1, p2):
        start = min(p1, p2)
        end = max(p1, p2)
        return EventsCollection([e for e in self if start <= e.peak_p < end])

    def in_unordered_time_range(self, t1, t2):
        start = min(t1, t2)
        end = max(t1, t2)
        return EventsCollection([e for e in self if start <= e.peak_t < end])

    def average_amplitude(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                result = np.mean([e.amplitude for e in self])
            except RuntimeWarning as warn:
                if str(warn).endswith("Mean of empty slice."):
                    print("Mean of empty slice")
                    result = 0
                else:
                    raise warn
        return result

    def weighted_amplitude(self, duration):
        amplitude = self.total_amplitude()
        if amplitude == 0:
            return 0
        weighted_ampl = amplitude / duration
        return weighted_ampl

    def total_amplitude(self):
        return np.sum([e.amplitude for e in self])

    @property
    def peak_positions(self):
        return np.array(sorted([e.peak_p for e in self]), dtype=np.float64)

    @property
    def peak_times(self):
        return np.array(sorted([e.peak_t for e in self]), dtype=np.float64)

    @property
    def start_times(self):
        return np.array(sorted([e.start_t for e in self]), dtype=np.float64)

    @property
    def rise_times(self):
        return np.array(
            sorted([e.half_rise_t for e in self]), dtype=np.float64
        )

    @property
    def amplitudes(self):
        return np.array(sorted([e.amplitude for e in self]), dtype=np.float64)

    @property
    def integrals(self):
        return np.array([e.integral for e in self], dtype=np.float64)

    @property
    def averages(self):
        return np.array([e.avg for e in self], dtype=np.float64)

    @property
    def cumulative_activity(self):
        events = self.in_point_range(200, 600)
        cumsum = 0
        cumsums = []
        times = []
        for e in events:
            # cumsum += e.integral
            # cumsum += e.average
            cumsum += e.total

            cumsums.append(cumsum)
            times.append(e.peak_p)
        return cumsums, times

    def get_nearest(self, p):
        vals, times = self.cumulative_activity
        times = np.array(times)
        time_differences = times - p
        time_differences[time_differences > 0] = -10000
        nearest_preceding_datapoint_idx = time_differences.argmax()
        return vals[nearest_preceding_datapoint_idx]

    def get_events_point_params(self):
        starts_pos = [e.start_p for e in self]  # OPTIMISE:
        peaks_pos = [e.peak_p for e in self]
        half_rise_pos = [e.half_rise_p for e in self]
        amplitudes = [e.amplitude for e in self]

        return starts_pos, peaks_pos, half_rise_pos, amplitudes

    def events_trace(self, start=200, end=400, event_latency=6):
        events = self.in_point_range(start, end)
        events_trace = np.zeros(600)
        for e in events:
            e_start = e.start_p
            e_end = e.get_end_p()
            if np.isnan(e_end):
                continue

            e_trace = e.trace[e_start:e_end]
            events_trace[e_start:e_end] = e_trace
            events_trace[e_start:e_end] -= e.estimate_baseline()

        events_trace[: (start + event_latency)] = 0
        return events_trace
