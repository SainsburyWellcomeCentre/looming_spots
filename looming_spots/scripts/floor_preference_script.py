
floor_preference_group = ['CA180_1', 'CA180_2', 'CA180_3', 'CA180_4', 'CA180_5', 'CA181_1', 'CA181_2', 'CA181_3', 'CA181_4', 'CA181_5']
floor_sessions = []

for mid in floor_preference_group:
    sessions = load.load_sessions(mid)
    for s in sessions:
        if s.n_looms == 0:
            floor_sessions.append(s)

all_pref = []
for s in floor_sessions:
    preferences=floor_preference.plot_binned_preference(s, grid_location=s.grid_location, start=None, n_bins=10)

    plt.plot([x for x in preferences], linewidth=0.5, alpha=0.7)
    all_pref.append(preferences)
mean_preferences = np.nanmean(all_pref, axis=0)
#[plt.bar(i, 1-pref-0.5, alpha=0.2, color='k') for i, pref in enumerate(mean_preferences)]

plt.plot(mean_preferences, color='k', linewidth=2)
plt.xticks(range(0,10), range(1,11))
plt.xlabel('time bin (min)')
plt.ylabel('% time spent on smooth')


plt.close('all')
plt.figure()
color='k'
label='pooled 10 minutes'
plt.title('mouse floor preference in first 10 minutes (n=10)')
pooled= np.nanmean(all_pref, axis=0)
overall_mean = np.mean(pooled)
std = np.std(pooled)
plt.scatter(1,overall_mean, zorder=20, color=color, s=50, label=label)
plt.errorbar(1, overall_mean, std)
plt.hlines(0.5,0,2,alpha=0.4,linestyle='--')
plt.ylim([0,1])
plt.xlim([-1,3])
plt.ylabel('% time spent on smooth')
plt.xticks([])

color = 'b'
label = 'pooled first 2 min'
all_pref_first_2 = [pref[0:2] for pref in all_pref]
pooled= np.nanmean(all_pref_first_2, axis=0)
overall_mean = np.mean(pooled)
std = np.std(pooled)
plt.scatter(0,overall_mean, zorder=20, color=color, s=50, label=label)
plt.errorbar(0, overall_mean, std)

color = 'r'
label = 'pooled last 2 min'
all_pref_last_2 = [pref[8:] for pref in all_pref]
pooled = np.nanmean(all_pref_last_2, axis=0)
overall_mean = np.mean(pooled)
std = np.std(pooled)
plt.scatter(2,overall_mean, zorder=20, color=color, s=50, label=label)
plt.errorbar(2, overall_mean, std, color=color)
plt.legend()