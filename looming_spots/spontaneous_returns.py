from looming_spots.db import loom_trial_group
import matplotlib.pyplot as plt

mids = [
        "CA325_1",
        "CA319_1",
        "CA319_2",
        "CA319_3",
        "CA319_5",
        "CA320_4",
        "CA320_5",
        "CA284_2",
        "CA284_3",
        "CA284_4",
        "CA284_5",
        "CA285_1",
        "CA285_2",

        "CA473_1",
        "CA473_2",
        "CA475_4",
        "CA475_5",
        "CA476_1",

        ]
plt.figure()
for mid in mids:
    mtg = loom_trial_group.MouseLoomTrialGroup(mid)
    track = mtg.get_first_7min_normalised_x_track()
    entries = mtg.get_first_7min_all_tz_entries()
    [plt.plot(track[entry-200:entry+400], color='k') for entry in entries[-3:]]
plt.show()
