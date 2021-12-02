import os
from pathlib import Path

import deeplabcut

from looming_spots.constants import RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY
from looming_spots.tracking_dlc import track_5label, process_DLC_output
import pathlib
import numpy as np
from looming_spots.tracking_dlc.constants import DLC_FNAMES, CONFIG_PATHS
from looming_spots.util import video_processing


def get_video_paths_from_mouse_id(mid):
    p = pathlib.Path(RAW_DATA_DIRECTORY) / mid
    vpaths = p.rglob("*.avi")
    return [str(v) for v in vpaths]


def get_tracks_path(directory, config_path_label, get_all_paths=False):
    fname = DLC_FNAMES[config_path_label]
    h5_files = list(directory.glob(f"{fname}"))

    if h5_files:
        if get_all_paths:
            return h5_files
        for f in h5_files:
            if "filtered" in str(f):
                return f
        return h5_files[0]

    return None


def save_npy_tracks(directory, config_path_label):
    tracks_path = get_tracks_path(directory, config_path_label)
    print(tracks_path)
    if tracks_path is not None:
        print(f"saving tracks to {tracks_path}")
        mouse_xy_tracks = process_DLC_output.process_DLC_output(
            str(tracks_path), config_path_label
        )
        mouse_positions_x_path = str(tracks_path.parent / "dlc_x_tracks.npy")
        mouse_positions_y_path = str(tracks_path.parent / "dlc_y_tracks.npy")

        np.save(mouse_positions_x_path, mouse_xy_tracks["x"])
        np.save(mouse_positions_y_path, mouse_xy_tracks["y"])


def get_paths(source_path, video_file_name='camera', video_fmt='avi'):
    return source_path.rglob(f'*{video_file_name}.{video_fmt}')


def process_mouse(m_id, config_path_label='bg_5_label', overwrite=False, video_file_name='camera', input_video_fmt='avi', output_video_fmt='mp4'):
    raw_mouse_path = RAW_DATA_DIRECTORY / m_id
    processed_mouse_path = PROCESSED_DATA_DIRECTORY / m_id

    raw_paths = get_paths(raw_mouse_path, video_file_name=video_file_name, video_fmt=input_video_fmt)

    for path in list(raw_paths):
        video_processing.convert_avi_to_mp4(path, dest=Path(str(path).replace('raw_data', 'processed_data')))

    processed_paths = get_paths(processed_mouse_path, video_file_name=video_file_name, video_fmt=output_video_fmt)

    for path in list(processed_paths):
        dlc_track_video(
            pathlib.Path(str(path)),
            config_path_label=config_path_label,
            overwrite=overwrite,
        )


def dlc_track_video(
    vpath, config_path_label, label_video=False, overwrite=False
):

    directory = vpath.parent
    all_paths = get_tracks_path(
        directory, config_path_label, get_all_paths=True
    )

    if not os.path.isdir(directory):
        os.mkdir(str(directory))

    config = CONFIG_PATHS[config_path_label]
    tracks_path = get_tracks_path(directory, config_path_label)

    if not os.path.isfile(str(tracks_path)):
        deeplabcut.analyze_videos(config, [str(vpath)])

    if overwrite:

        if all_paths is not None:
            for p in all_paths:
                print(f"deleting... {str(p)}")
                os.remove(str(p))

        deeplabcut.analyze_videos(config, [str(vpath)])

    deeplabcut.filterpredictions(config, [str(vpath)])

    tracks_path = get_tracks_path(vpath.parent, config_path_label)

    if tracks_path is not None:
        print(f"valid track path {tracks_path} processing and output")
        mouse_xy_tracks = process_DLC_output.process_DLC_output(
            tracks_path, config_path_label
        )

        mouse_positions_x_path = directory / "dlc_x_tracks.npy"
        mouse_positions_y_path = directory / "dlc_y_tracks.npy"
        print(
            f"saving tracks to {mouse_positions_x_path} and {mouse_positions_y_path}"
        )
        np.save(str(mouse_positions_x_path), mouse_xy_tracks["x"])
        np.save(str(mouse_positions_y_path), mouse_xy_tracks["y"])

        if label_video:
            print("creating labeled video")
            labelled_vid = deeplabcut.create_labeled_video(
                config, [vpath], save_frames=True
            )
            return mouse_xy_tracks, labelled_vid

    save_npy_tracks(directory, config_path_label)


if __name__ == "__main__":

    mids = ["1097643"]
    for mid in mids:
        process_mouse(mid, overwrite=True)

