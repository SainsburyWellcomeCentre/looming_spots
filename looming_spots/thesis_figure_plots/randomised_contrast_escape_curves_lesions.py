import os
import numpy as np
import matplotlib.pyplot as plt

from looming_spots.analysis.photometry_habituations import get_signal_metric_dataframe_variable_contrasts, \
    get_behaviour_metric_dataframe
from looming_spots.db import load, experimental_log, loom_trial_group
from looming_spots.analysis import randomised_contrast_escape_curves
import pandas as pd


# df = experimental_log.load_df()
# mids = experimental_log.get_mouse_ids_in_experiment('background_contrast_cossel_curve')
# root_dir = '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/'
#
# for mid in mids:
#     sessions = load.load_sessions(mid)
#     if sessions is not None:
#         for s in sessions:
#             save_path = os.path.join(s.path, 'contrasts.npy')
#             print(save_path)
#             contrast = df[df['mouse_id'] == mid]['contrast']
#             print(mid, float(contrast))
#             np.save(save_path, contrast)
#
#
# plt.close('all')
# OHDA = ['CA451A_2', 'CA451A_3', 'CA451A_4', 'CA478_2', 'CA476_4']
# NMDA = ['276585A', '276585B', 'CA439_1', 'CA439_4']
# CONTROL = ['276585D', '276585E', 'CA452_1', 'CA439_5', 'CA451A_5', 'CA459A_2', 'CA478_3']
#
# for l, c, mids in zip(['OHDA', 'NMDA', 'CONTROL'], ['g', 'b', 'k'], [OHDA, NMDA, CONTROL]):
#     randomised_contrast_escape_curves.plot_block_escape_curves_with_avg(mids, c)
#
GROUPS ={'OHDA':    ['CA451A_1', 'CA451A_2', 'CA451A_3', 'CA451A_4', 'CA478_2', 'CA476_4'],
         'NMDA':    ['276585A', '276585B', 'CA439_1', 'CA439_4'],
         'CONTROL': ['276585D', '276585E', 'CA452_1', 'CA439_5', 'CA451A_5', 'CA459A_2', 'CA478_3']
         }

def get_df(metric, groups=GROUPS):
    all_df = pd.DataFrame()
    for label, mids in groups.items():
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
        df = get_behaviour_metric_dataframe(mtgs, metric, 'variable_contrast')
        df['experimental group'] = [label]*len(df)
        all_df = all_df.append(df)
    return all_df
