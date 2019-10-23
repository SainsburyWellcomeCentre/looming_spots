import time
import numpy as np
from matplotlib import pyplot as plt


def plot_events(trace, events_pos, peaks_pos, peak_ampls, half_rises):
    plt.ion()  # interactive

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(-(50 * 0.04), 0.04, 100)  # 2 secs
    y = x.copy()
    y.fill(0)
    line1, = ax.plot(
        x, y, "b-"
    )  # Returns a tuple of line objects, thus the comma

    cursors_y = (0, 0, 0)  # defaults
    cursors_x = (x[0], x[len(x) / 2], x[-1])
    line2, = ax.plot(cursors_x, cursors_y, "r*")

    for event_start, event_end, half_rise_t in zip(
        events_pos, peaks_pos, half_rises
    ):
        event = trace[
            (event_start - 50) : (event_start + 50)
        ]  # REFACTOR: why 50
        relative_event_start = 50
        relative_half_rise_t = 50 + half_rise_t - event_start
        relative_event_end = 50 + event_end - event_start
        line1.set_ydata(event)
        line2.set_xdata(
            (relative_event_start, relative_half_rise_t, relative_event_end)
        )
        line2.set_ydata(
            (
                event[relative_event_start],
                event[relative_half_rise_t],
                event[relative_event_end],
            )
        )
        fig.canvas.draw()
        time.sleep(0.05)


def _plot_trace(
    trace,
    trace_label,
    events_pos,
    peaks_pos,
    color=None,
    x=None,
    ax=None,
    y_shift=0.2,
    marker_size=100,
):
    if events_pos is False:
        return
    events_pos = np.array(events_pos, dtype=np.int64)
    peaks_pos = np.array(peaks_pos, dtype=np.int64)
    plot_element = ax if ax is not None else plt

    if color is None:
        plot_element.plot(x, trace, label=trace_label)
    else:
        plot_element.plot(x, trace, label=trace_label, color=color)

    if not len(events_pos):
        return

    sampling = x[1] - x[0]
    plt_y = np.take(trace, events_pos)
    plot_element.scatter(
        events_pos * sampling,
        plt_y,
        color="g",
        s=marker_size,
        zorder=5,
        label="event start",
    )
    plt_y = np.take(trace, peaks_pos) + y_shift
    plot_element.scatter(
        peaks_pos * sampling,
        plt_y,
        color="r",
        s=marker_size,
        zorder=4,
        label="event end",
    )

    plot_element.grid(color="blue", linestyle="--")
