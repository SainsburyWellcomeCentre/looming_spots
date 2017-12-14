import os
from datetime import datetime

import looming_spots.db.experiment_metadata
import looming_spots.preprocess.extract_looms
import looming_spots.preprocess.photodiode
from looming_spots.preprocess import extract_looms
from ploom import mouse, session


def main(n):
    all_mice = []
    directory = '/home/slenzi/spine_shares/loomer/data_working_copy/'
    for (mouse_id, i) in zip(os.listdir(directory), n):
        m = mouse.Mouse(mouse_id)
        sessions_path = os.path.join(directory, mouse_id)
        for s in os.listdir(sessions_path):
            if not looming_spots.preprocess.extract_looms.is_datetime(s):
                continue
            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s_path = os.path.join(sessions_path, s)
            context = looming_spots.db.experiment_metadata.get_context_from_stimulus_mat(s_path)
            if context == 'n/a':
                return all_mice  # FIXME:
            loom_idx = extract_looms.get_loom_idx_from_raw(s_path)
            protocol = looming_spots.db.experiment_metadata.get_session_label_from_loom_idx(loom_idx)
            s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context)
            m.sessions.append(s)
        all_mice.append(m)
    return all_mice


if __name__ == '__main__':
    main()
