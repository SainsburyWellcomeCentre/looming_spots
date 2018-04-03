import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session

from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY
from looming_spots.db import session_group, experiment


def load_sessions(mouse_id):
    sessions_path = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    for s in os.listdir(sessions_path):
        if not looming_spots.util.generic_functions.is_datetime(s):
            continue
        date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
        s = session.Session(dt=date, mouse_id=mouse_id)
        session_list.append(s)
    return sorted(session_list)


def load_experiment_from_groups(groups, group_labels):
    sgs = []
    for ids, label in zip(groups, group_labels):
        sessions = []
        for mid in ids:
            ms = session_group.MouseSessionGroup(mid)
            s = ms.nth_post_test(0)
            sessions.append(s)
        sgs.append(session_group.SessionGroup(sessions, group_key=label))
    exp = experiment.Experiment(sgs)
    return exp