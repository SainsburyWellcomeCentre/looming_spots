import numpy as np
from margrie_libs.signal_processing.thresholding import find_level_decrease
from cached_property import cached_property
import matplotlib.pyplot as plt


class Event(object):
    ATTRIBUTES = ('start_p', 'peak_p', 'half_rise_p', 'amplitude')  # TODO: add end

    def __init__(self, start_p, peak_p, half_rise_p, amplitude, trace, sampling_interval, default_comparator='peak_p'):
        if default_comparator not in Event.ATTRIBUTES:
            raise AttributeError("Default_comparator should be one of {}, got {}".
                                 format(Event.ATTRIBUTES, default_comparator))
        self.start_p = start_p
        self.peak_p = peak_p
        self.half_rise_p = half_rise_p
        self.amplitude = amplitude
        self.sampling_interval = sampling_interval
        self.default_comparator = default_comparator
        self.trace = trace

    def __sub__(self, scalar):
        # if self.start - scalar < 0:
        #     raise ValueError("Event time becomes negative (from {} to {} with shift by {})".
        #                      format(self.start, self.start - scalar, scalar))
        return Event(self.start_p - scalar,
                     self.peak_p - scalar,
                     self.half_rise_p - scalar,
                     self.amplitude,
                     self.sampling_interval,
                     default_comparator=self.default_comparator)

    def __cmp__(self, other):
        return self.explicit_cmp(other, self.default_comparator)

    @cached_property
    def start_t(self):
        return self.start_p * self.sampling_interval

    @cached_property
    def peak_t(self):
        return self.peak_p * self.sampling_interval

    @cached_property
    def half_rise_t(self):
        return self.half_rise_p * self.sampling_interval

    @cached_property
    def integral(self):
        return self.get_integral(self.trace)

    @cached_property
    def average(self):
        return self.get_average(self.trace)

    @cached_property
    def total(self):
        return self.get_sum(self.trace)

    def bsl_start(self, n_bsl_p=10):
        return max(self.start_p - n_bsl_p, 0)

    def bsl_end(self):
        return self.start_p - 1

    def estimate_baseline(self):
        return np.median(self.trace[(self.bsl_end()-5):self.bsl_end()])
        #return self.trace[self.bsl_end()]  # np.mean(self.trace[self.bsl_start():self.bsl_end()])

    def get_average(self, trace):

        try:
            start_p_y = trace[self.start_p]
            end_p = find_level_decrease(trace[self.peak_p:],
                                        start_p_y + self.amplitude * 0.1)  # return to 10% amplitude
            end_p += self.peak_p

            event_trace = trace[self.start_p:end_p]
            event_trace -= self.estimate_baseline()

            avg = np.mean(event_trace)
        except StopIteration:
            avg = 0  # FIXME: should be NaN ?

        return avg

    def get_sum(self, trace):

        try:
            start_p_y = trace[self.start_p]
            end_p = find_level_decrease(trace[self.peak_p:],
                                        start_p_y + self.amplitude * 0.1)  # return to 10% amplitude
            end_p += self.peak_p

            event_trace = trace[self.start_p:end_p]
            event_trace -= self.estimate_baseline()

            total = sum(event_trace)
        except StopIteration:
            total = 0  # FIXME: should be NaN ?

        return total

    def get_end_p(self, threshold=0.1):
        try:
            start_p_y = self.trace[self.start_p]
            end_p = find_level_decrease(self.trace[self.peak_p:],
                                        start_p_y + self.amplitude * threshold)  # return to % amplitude
            end_p += self.peak_p
        except StopIteration:
            return np.nan
        return end_p

    def get_integral(self, trace):
        try:
            start_p_y = trace[self.start_p]
            end_p = find_level_decrease(trace[self.peak_p:],
                                        start_p_y + self.amplitude * 0.1)  # return to 10% amplitude
            end_p += self.peak_p

            event_trace = trace[self.start_p:end_p]
            event_trace -= self.estimate_baseline()

            plt.plot(np.arange(self.start_p, end_p, 1), event_trace)
            plt.plot(np.arange(self.bsl_start(), self.bsl_end(), 1), trace[self.bsl_start():self.bsl_end()], 'k')
            integral = np.trapz(event_trace, dx=self.sampling_interval)
        except StopIteration:
            integral = 0  # FIXME: should be NaN ?
        return integral

    def explicit_cmp(self, other, comparator):
        if not isinstance(other, Event):
            raise AttributeError("Event expected got: {}".format(other.type))
        if getattr(self, comparator) < getattr(other, comparator):
            return -1
        elif getattr(self, comparator) == getattr(other, comparator):
            return 0
        else:
            return 1


