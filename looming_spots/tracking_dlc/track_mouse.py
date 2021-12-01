from looming_spots.constants import RAW_DATA_DIRECTORY
from looming_spots.tracking_dlc import track_5label, process_DLC_output
import pathlib
import numpy as np
from looming_spots.tracking_dlc.constants import DLC_FNAMES


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


def track_mouse(mid, config_path_label="bg_5_label_SL_new", overwrite=False):
    vpaths = get_video_paths_from_mouse_id(mid)

    for vpath in vpaths:
        track_5label.track_5_label(
            pathlib.Path(vpath),
            config_path_label=config_path_label,
            overwrite=overwrite,
        )


if __name__ == "__main__":

    mids = ["1114088", "1114089", "1114185", "1114186", "1114188", "1114189"]
    for mid in mids:
        track_mouse(mid, overwrite=True)
