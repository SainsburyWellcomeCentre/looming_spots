import os
from ploom import mouse, session
from looming_spots.analysis import extract_looms
from datetime import datetime


def main(n):
    all_mice = []
    directory = '/home/slenzi/spine_shares/loomer/data_working_copy/'
    for (mouse_id, i) in zip(os.listdir(directory), n):
        m = mouse.Mouse(mouse_id)
        sessions_path = os.path.join(directory, mouse_id)
        for s in os.listdir(sessions_path):
            if not extract_looms.is_datetime(s):
                continue
            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s_path = os.path.join(sessions_path, s)
            context = extract_looms.get_context_from_stimulus_mat(s_path)
            if context == 'n/a':
                return all_mice  # FIXME:
            protocol = extract_looms.get_session_label(s_path)
            s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context)
            m.sessions.append(s)
        all_mice.append(m)
    return all_mice


if __name__ == '__main__':
    main()
