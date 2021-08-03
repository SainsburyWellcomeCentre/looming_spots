import sys

from analysis.signal_processing_sepi.filters import box_smooth, diff
from matplotlib import pyplot as plt

from analysis.event_detection.calcium_trace_handling import load_fluo_profiles
from analysis.event_detection.event_detection import detect_trace
from analysis.event_detection.event_plot import _plot_trace


def plot_cell(
    trace, trace_label, delta_wave, events_pos, peaks_pos, half_rises
):
    fig = plt.figure()
    _plot_trace(trace, trace_label, events_pos, peaks_pos, half_rises)
    plt.plot(delta_wave + trace.max())
    plt.legend()
    fig.show()


def detect_cell(
    profile,
    cell_id,
    threshold,
    n_pnts_bsl,
    n_pnts_peak,
    n_pnts_rise_t,
    n_pnts_for_peak_detection,
    n_sds,
    n_pnts_high_pass_filter,
    minimum_time_delta=60,
):
    if profile.ndim == 3:
        trace = (profile[:, cell_id, 0]).copy()
    else:
        trace = (profile[:, cell_id]).copy()
    return detect_trace(
        trace,
        threshold,
        n_pnts_bsl,
        n_pnts_peak,
        n_pnts_rise_t,
        n_pnts_for_peak_detection,
        n_sds,
        n_pnts_high_pass_filter,
        minimum_time_delta,
    )


def main(args_list):
    profiles_path_base = args_list[0]
    header, green_profiles = load_fluo_profiles(profiles_path_base + "1.csv")
    header, red_profiles = load_fluo_profiles(profiles_path_base + "2.csv")
    threshold = float(args_list[1])
    n_pnts_bsl, n_pnts_peak, n_pnts_rise_t = [
        int(arg) for arg in args_list[2:5]
    ]
    n_sds = int(args_list[5])
    #    if len(args_list) == 7:
    #        sampleInterval = float(args_list[6])
    n_pnts_for_peak_detection = 100
    n_pnts_high_pass_filter = 100  # OPTIMISE: compute as function of (n_pnts_bsl + n_pnts_peak + n_pnts_rise_t)

    #    for i in range(2):
    for j in range(green_profiles.shape[1]):
        profile = green_profiles[j]
        events_pos, peaks_pos, half_rises, peak_ampls = detect_cell(
            profile,
            j,
            threshold,
            n_pnts_for_peak_detection,
            n_sds,
            n_pnts_high_pass_filter,
        )
        # PLOT
        delta_wave = diff(
            box_smooth(profile, n_pnts_peak),
            box_smooth(profile, n_pnts_bsl),
            n_pnts_rise_t,
        )
        plot_cell(
            profile,
            "GreenCell{}".format(j),
            delta_wave,
            events_pos,
            peaks_pos,
            half_rises,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
# FIXME: add refractory period
# TODO: do time version


# def delta_f_over_f(data):
#     f = data[:, :, 0]  # only cells
#     f0 = np.mean(f, 0)  # mean across time
#     return (f-f0) / f0
