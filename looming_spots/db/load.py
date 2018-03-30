import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session
from looming_spots.db.metadata import experiment_metadata
from looming_spots.db.metadata import get_context, get_session_label_from_loom_idx

PROCESSED_DATA_PATH = '/home/slenzi/spine_shares/loomer/processed_data'


def load_sessions(mouse_id):
    sessions_path = os.path.join(PROCESSED_DATA_PATH, mouse_id)
    session_list = []
    for s in os.listdir(sessions_path):
        if not looming_spots.util.generic_functions.is_datetime(s):
            continue
        date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
        s_path = os.path.join(sessions_path, s)
        context = get_context(s_path)
        loom_idx = experiment_metadata.get_loom_idx(s_path)
        protocol = get_session_label_from_loom_idx(loom_idx)
        s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context, mouse_id=mouse_id)
        session_list.append(s)
        print('session path: {}, context: {}'.format(s.path, s.context))
    return session_list
