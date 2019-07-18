import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter


def get_starts_and_ends(above_threshold, min_event_size=3):
    diff = np.diff(above_threshold.astype(int))
    unfiltered_starts = np.where(diff > 0)[0]
    unfiltered_ends = np.where(diff < 0)[0]

    if unfiltered_ends[0] < unfiltered_starts[0]:
        unfiltered_ends = unfiltered_ends[1:]
    if unfiltered_starts[-1] > unfiltered_ends[-1]:
        unfiltered_starts = unfiltered_starts[:-1]

    starts = [s for (s, e) in zip(unfiltered_starts, unfiltered_ends) if e - s > min_event_size]
    ends = [e for (s, e) in zip(unfiltered_starts, unfiltered_ends) if e - s > min_event_size]

    return starts, ends


def detect_events(trace, bsl_start, bsl_end):
    threshold = np.std(trace[bsl_start:bsl_end])
    above_threshold = trace > threshold
    starts, ends = get_starts_and_ends(above_threshold)

    peak_locs = np.array([np.argmax(trace[start:end]) + start for start, end in zip(starts, ends)])
    amplitudes = np.array([np.max(trace[start:end]) for start, end in zip(starts, ends)])
    half_rise = np.mean([starts, peak_locs], axis=0)
    return starts, ends, peak_locs, half_rise, amplitudes


def visualise_events(trial_group, other_trial_group, start=300, end=12600, axes=None):

    df1 = trial_group[0].session.delta_f[start:end]
    df2 = other_trial_group[0].session.delta_f[start:end]

    if axes is None:
        fig, axes = plt.subplots(3, 2)
    ylim = max([max(df) for df in [df1, df2]])

    for i, df in enumerate([df1, df2]):
        _, _, peak_locs, _, amplitudes = detect_events(df, 0, None)
        plt.sca(axes[0][i])
        plt.plot(df)
        plt.ylim([-0.001, ylim])
        plt.plot(peak_locs, np.zeros_like(peak_locs), 'o')
        maximum_peak_loc = peak_locs[np.argmax(amplitudes)]
        maximum_amplitude = max(amplitudes)

        plt.plot(maximum_peak_loc, maximum_amplitude, 'o')
        print('maximum event amplitude: {}, standard dev. event amplitude: {}, '
              'avg. event amplitude: {}, median event amplitude: {}'.format(max(amplitudes),
                                                                            np.std(amplitudes),
                                                                            np.mean(amplitudes),
                                                                            np.median(amplitudes)))

        print('signal:noise ratio: {}'.format(np.mean(amplitudes)/np.std(df)))

        plt.sca(axes[1][i])
        plt.hist(amplitudes, histtype='step')
        plt.sca(axes[2][i])
        plt.plot(df[(maximum_peak_loc-20):(maximum_peak_loc+20)])
        plt.ylim([-0.001, ylim])


def get_ca_event_metrics_from_trial_groups(trial_groups, labels, start, end):

    """

    :param trial_groups: a list of groups of trials
    :return:
    """

    df_dict = {}
    max_amplitudes = []
    std_amplitudes = []
    mean_amplitudes = []
    median_amplitudes = []
    n_events = []
    session_paths = []

    for tg in trial_groups:
        s = tg[0].session
        df = s.delta_f[start:end]
        _, _, peak_locs, _, amplitudes = detect_events(df, 0, None)

        max_amplitudes.append(max(amplitudes))
        std_amplitudes.append(np.std(amplitudes))
        mean_amplitudes.append(np.mean(amplitudes))
        median_amplitudes.append(np.median(amplitudes))
        n_events.append(len(amplitudes))
        session_paths.append(s.path)

    df_dict.setdefault('n events', n_events)  # TODO: extract
    df_dict.setdefault('amp. max', max_amplitudes)
    df_dict.setdefault('amp. std', std_amplitudes)
    df_dict.setdefault('amp. mean', mean_amplitudes)
    df_dict.setdefault('amp. median', median_amplitudes)
    df_dict.setdefault('condition', labels)
    df_dict.setdefault('session_path', session_paths)

    return pd.DataFrame.from_dict(df_dict)


def plot_all_events(trace, locs, n_samples_pre=10, n_samples_post=20):
    fig = plt.figure()
    all_events = np.array([trace[(loc-n_samples_pre):(loc+n_samples_post)] for loc in locs]).T
    plt.plot(all_events, color='k', alpha=0.3)
    plt.plot(np.mean(all_events, axis=1), 'r', linewidth=2)
    return fig


def get_rescale_factors_to_sessions_max(trial_group, hb_group, other_trial_group, bsl_start=300, bsl_end=None):
    """assumes two sessions only"""

    df1 = trial_group[0].session.delta_f
    df2 = hb_group[0].session.delta_f
    df3 = other_trial_group[0].session.delta_f

    max_amplitudes = []

    for df in [df1, df2, df3]:
        _, _, _, _, amplitudes = detect_events(df, bsl_start, bsl_end)
        max_amplitudes.append(max(amplitudes))

    return max_amplitudes


def fit_template(trace, peak_loc, template=None, default_path='/home/slenzi/template_event.npy'):
    if template is None:
        template = np.load(default_path)

    template_peak = np.argmax(template)
    template_end = len(template)

    rescaled_template = trace[peak_loc]/max(template) * template
    overlay = np.zeros_like(trace)
    start = peak_loc - template_peak
    end = peak_loc+template_end - template_peak
    overlay[start:end] = rescaled_template

    return overlay


def subtract_event(trace, peak_loc):
    return trace - fit_template(trace, peak_loc)
