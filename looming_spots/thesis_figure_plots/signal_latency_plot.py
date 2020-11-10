from looming_spots.db import loom_trial_group, experimental_log
import matplotlib.pyplot as plt
import numpy as np

ALL_SNL_KEYS = ['photometry_habituation_tre-GCaMP_24hr_pre',
                'photometry_habituation_tre-GCaMP_same_day_pre',
                'photometry_habituation_tre-GCaMP-contrasts']

LSIE_SNL_KEYS = ['photometry_habituation_tre-GCaMP_24hr_pre',
                'photometry_habituation_tre-GCaMP_same_day_pre',]


def get_pre_test_and_high_contrast_trials_mtg(mtg):
    trials = mtg.pre_test_trials()
    return [t for t in trials if t.contrast == 0]


def get_pre_test_and_high_contrast_trials(mtgs):
    all_trials = []
    for mtg in mtgs:
        if mtg.exp_key == 'photometry_habituation_tre-GCaMP-contrasts':
            trials = [t for t in mtg.all_trials[:18] if t.contrast == 0]
        else:
            trials = mtg.pre_test_trials()[:3]
        all_trials.extend(trials)
    return all_trials


def get_snl_pre_test_and_high_contrast_trials():
    mtgs = get_mtgs(ALL_SNL_KEYS)
    trials = get_pre_test_and_high_contrast_trials(mtgs)
    for t in trials:
        fig = plt.figure()
        title = f'deltaF_with_track__mouse_{t.mouse_id}__trial_{t.loom_number}'
        plt.title(title)
        t.plot_delta_f_with_track('k')
        fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.png')
        fig.close()


def get_mtgs(keys):
    mtgs = []
    for key in keys:
        mtgs.extend(experimental_log.get_mtgs_in_experiment(key))
    return mtgs


def calculate_theoretical_escape_threshold(mtg):
    pre_test_trials = mtg.pre_test_trials()[:3]
    post_test_trials = mtg.post_test_trials()[:3]
    pre_test_latency = np.nanmean([t.latency_peak_detect() for t in pre_test_trials])

    theoretical_escape_threshold = np.mean([t.integral_escape_metric(int(pre_test_latency)) for t in pre_test_trials])
    title = f'theoretical_threshold_{mtg.mouse_id}'
    fig=plt.figure()
    plt.title(title)
    plt.axhline(theoretical_escape_threshold)
    for t in post_test_trials:
        plt.plot(t.integral_downsampled)
    fig.savefig(f'/home/slenzi/thesis_latency_plots/{title}.png')
    fig.close()


def plot_all_theoretical_escape_thresholds():
    mtgs = get_mtgs(LSIE_SNL_KEYS)
    for mtg in mtgs:
        calculate_theoretical_escape_threshold(mtg)


def main():
    #get_snl_pre_test_and_high_contrast_trials()
    plot_all_theoretical_escape_thresholds()


if __name__ == '__main__':
    main()
