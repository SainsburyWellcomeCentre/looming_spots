import numpy as np
from margrie_libs.signal_processing.thresholding import find_level_decrease
from cached_property import cached_property


class Event(object):
    ATTRIBUTES = (
        "start_p",
        "peak_p",
        "half_rise_p",
        "amplitude",
    )  # TODO: add end

    def __init__(
        self,
        start_p,
        peak_p,
        half_rise_p,
        amplitude,
        sampling_interval,
        default_comparator="peak_p",
    ):
        if default_comparator not in Event.ATTRIBUTES:
            raise AttributeError(
                "Default_comparator should be one of {}, got {}".format(
                    Event.ATTRIBUTES, default_comparator
                )
            )
        self.start_p = start_p
        self.peak_p = peak_p
        self.half_rise_p = half_rise_p
        self.amplitude = amplitude
        self.sampling_interval = sampling_interval
        self.default_comparator = default_comparator

    def __sub__(self, scalar):
        # if self.start - scalar < 0:
        #     raise ValueError("Event time becomes negative (from {} to {} with shift by {})".
        #                      format(self.start, self.start - scalar, scalar))
        return Event(
            self.start_p - scalar,
            self.peak_p - scalar,
            self.half_rise_p - scalar,
            self.amplitude,
            self.sampling_interval,
            default_comparator=self.default_comparator,
        )

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

    def get_integral(self, trace):
        try:
            start_p_y = trace[self.start_p]
            end_p = find_level_decrease(
                trace[self.peak_p :], start_p_y + self.amplitude * 0.2
            )  # return to 20% amplitude
            end_p += self.peak_p
            event_trace = trace[self.start_p : end_p]
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
