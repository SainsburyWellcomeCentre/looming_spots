from looming_spots.db import loom_trial_group, experimental_log
from matplotlib import patches
import matplotlib.pyplot as plt

from looming_spots.db.loom_trial_group import (
    make_trial_heatmap_location_overlay,
)

ICHLOC_MIDS = experimental_log.get_mouse_ids_in_experiment(
    "block_DA_during_pretest_flexichloc_SNLfibre"
)


def plot_ichloc_pretest_disruption(mids=ICHLOC_MIDS):
    fig, axes = plt.subplots(2, 1)
    plt.suptitle(
        "ichloc (15mW) inactivation of SNL during pre-test (same-day)"
    )
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        ax = plt.sca(axes[0])
        for t in mtg.pre_test_trials():
            t.plot_track(ax)
        t.plot_stimulus()
        ax = plt.sca(axes[1])
        r = patches.Rectangle((195, 0.95), 135, height=0.05, color="b")
        for t in mtg.post_test_trials():
            t.plot_track()
        t.plot_stimulus()
        axes[0].add_patch(r)

    plt.figure()
    for i, mid in enumerate(mids):
        trials = []
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        for t in mtg.lsie_trials():
            trials.extend([t])
        hm = make_trial_heatmap_location_overlay(trials)
        ax = plt.subplot(2, 3, i + 1)
        ax.title.set_text(f"ichloc {mid}")
        plt.imshow(hm, aspect="auto", vmax=2, vmin=0, interpolation="bilinear")
        ax.axis("off")
        plt.ylim(0, 300)
        plt.xlim(0, 400)
        ax2 = plt.subplot(2, 3, i + 4)
        for t in mtg.post_test_trials()[:3]:
            t.plot_track()
        t.plot_stimulus()
