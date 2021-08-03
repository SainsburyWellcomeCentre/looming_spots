import os
from pathlib import Path

import deeplabcut
import numpy as np

from looming_spots.tracking_dlc import process_DLC_output
from looming_spots.tracking_dlc.constants import CONFIG_PATHS
from looming_spots.tracking_dlc.track_mouse import (
    get_tracks_path,
    save_npy_tracks,
)


def track_5_label(
    vpath, config_path_label="bg_5_label", label_video=False, overwrite=False
):
    all_paths_raw = get_tracks_path(
        vpath.parent, config_path_label, get_all_paths=True
    )

    d = Path(str(vpath.parent).replace("raw_data", "processed_data"))
    directory = d / "5_label"
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
        if all_paths_raw is not None:
            for p in all_paths_raw:
                print(f"deleting... {str(p)}")
                os.remove(str(p))

        deeplabcut.analyze_videos(config, [str(vpath)])

    deeplabcut.filterpredictions(config, [str(vpath)])

    tracks_path = get_tracks_path(vpath.parent, config_path_label)
    tracks_path.replace(directory / tracks_path.parts[-1])

    tracks_path = get_tracks_path(directory, config_path_label)

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
