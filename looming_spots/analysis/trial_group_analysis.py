import numpy as np

"""
for functions that operate on lists of trials
"""


def make_trial_heatmap_location_overlay(trials):
    track_heatmap = np.zeros((480, 640))
    for t in trials:
        x, y = np.array(t.raw_track[0][200:400]), np.array(t.raw_track[1][200:400])
        for coordinate in zip(x, y):
            if not np.isnan(coordinate).any():
                track_heatmap[int(coordinate[1]), int(coordinate[0])] += 1
    return track_heatmap

#
# def sorted_tracks(self, values_to_sort_by=None):
#     if values_to_sort_by is None:
#         return self.all_tracks()
#     else:
#         args = np.argsort(values_to_sort_by)
#         order = [np.where(args == x)[0][0] for x in range(len(self.all_tracks()))]
#         sorted_tracks = []
#         for item, arg, sort_var in zip(order, args, values_to_sort_by):
#             trial_distances = self.all_tracks()[arg]
#             sorted_tracks.append(trial_distances[:400])
#         return sorted_tracks
#
# def plot_hm(self, values_to_sort_by):
#     fig = plt.figure(figsize=(7, 5))
#     tracks = sorted_tracks(values_to_sort_by)
#     plt.imshow(tracks, cmap='coolwarm_r', aspect='auto', vmin=-0.05, vmax=0.05)
#     title = '{}, {} flees out of {} trials, n={} mice'.format(self.label, self.n_flees, self.n_trials, self.n_mice)
#     plt.title(title)
#     plt.axvline(200, color='k')
#     cbar = plt.colorbar()
#     cbar.set_label('velocity in x axis a.u.')
#     plt.ylabel('trial number')
#     plt.xlabel('n frames')
#     return fig
