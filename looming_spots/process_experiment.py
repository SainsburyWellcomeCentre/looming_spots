from pathlib import Path

from looming_spots.constants import RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
from looming_spots.tracking_dlc.track_mouse import track_mouse
from looming_spots.util import video_processing


def get_mouse_ids():
    mouse_ids = ['1097633',
                 '1097634',
                 '1097635',
                 '1097639']

    return mouse_ids


mouse_ids = get_mouse_ids()




for m_id in mouse_ids:
    process_mouse(m_id)
    track_mouse(m_id)

