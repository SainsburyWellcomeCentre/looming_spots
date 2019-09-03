import os
import numpy as np
import matplotlib.pyplot as plt

from looming_spots.db import load, experimental_log
from looming_spots.analysis import randomised_contrast_escape_curves


df = experimental_log.load_df()
mids = experimental_log.get_mouse_ids_in_experiment('background_contrast_cossel_curve')
root_dir = '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/'

for mid in mids:
    sessions = load.load_sessions(mid)
    if sessions is not None:
        for s in sessions:
            save_path = os.path.join(s.path, 'contrasts.npy')
            print(save_path)
            contrast = df[df['mouse_id'] == mid]['contrast']
            print(mid, float(contrast))
            np.save(save_path, contrast)


plt.close('all')
OHDA = ['CA451A_2', 'CA451A_3', 'CA451A_4']
NMDA = ['276585A', '276585B', 'CA439_1', 'CA439_4']
CONTROL = ['276585D', '276585E', 'CA452_1', 'CA439_5', 'CA451A_5', 'CA459A_2']
for l, c, mids in zip(['OHDA', 'NMDA', 'CONTROL'], ['g', 'b', 'k'], [OHDA, NMDA, CONTROL]):
    randomised_contrast_escape_curves.plot_block_escape_curves_with_avg(mids, c)

