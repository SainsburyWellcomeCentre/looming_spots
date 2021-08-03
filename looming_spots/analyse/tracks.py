import os
import pathlib

import numpy as np
from looming_spots.analyse import arena_region_crossings
from looming_spots.constants import (
    FRAME_RATE,
    N_SAMPLES_BEFORE,
    CLASSIFICATION_WINDOW_START,
    CLASSIFICATION_WINDOW_END,
    ARENA_SIZE_CM,
    BOX_CORNER_COORDINATES,
    LOOMING_STIMULUS_ONSET,
    N_SAMPLES_TO_SHOW,
    SHELTER_FRONT,
    ARENA_LENGTH_PX,
    ARENA_WIDTH_PX,
)
from looming_spots.util.transformations import (
    get_inverse_projective_transform,
    get_box_coordinates_from_file,
)
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy import signal


def load_raw_track(session_directory, name, start, end, loom_folder=None):
    if get_tracking_method(session_directory) == "old_school":
        x, y = load_track_csv(loom_folder)
    else:
        x_path = pathlib.Path(session_directory) / name.format("x")
        y_path = pathlib.Path(session_directory) / name.format("y")
        x = np.load(str(x_path))[start:end]
        y = np.load(str(y_path))[start:end]
    return x, y


def normalised_x_track(x, frame_rate=30, target_frame_rate=30):
    """
    Positional traces are reported in pixels, this function normalises x traces into values between 0 and 1 for later
    conversion to real world units.

    :param x:
    :param frame_rate:
    :param target_frame_rate:
    :return:
    """
    normalised_track = 1 - (x / ARENA_LENGTH_PX)
    if frame_rate != target_frame_rate:
        normalised_track = downsample_track(normalised_track, frame_rate)
    return normalised_track


def normalised_y_track(y, frame_rate=30, target_frame_rate=30):
    """
    Positional traces are reported in pixels, this function normalises y traces into values between 0 and 1 for later
    conversion to real world units. Since the y range is from 0 - 0.4 everything is scaled down for the y-position.

    :param y:
    :param frame_rate:
    :param target_frame_rate:
    :return:
    """
    normalised_track = (y / ARENA_WIDTH_PX) * 0.4
    if frame_rate != target_frame_rate:
        normalised_track = downsample_track(normalised_track, frame_rate)
    return normalised_track


def load_box_corner_coordinates(session_directory):

    """
    Loads box corner coordinates that have been manually determined using the box_corner_gui. See projective transform
    for details on the coordinate order.

    :param session_directory:
    :return:
    """

    box_path = pathlib.Path(session_directory).glob(
        "box_corner_coordinates.npy"
    )
    if len(list(box_path)) == 0:
        print("no box coordinates found...")

    return get_box_coordinates_from_file(
        str(
            list(
                pathlib.Path(session_directory).glob(
                    "box_corner_coordinates.npy"
                )
            )[0]
        )
    )


def track_in_standard_space(
    session_directory, tracking_method, start, end, loom_folder=None
):
    """
    This function loads positional xy traces in standard space. For manual tracks and dlc tracks, these have been
    generated using videos that were transformed using the same projective transform approach. However, for
    older data, and cricket data rather than re-transforming all videos we simply transform the tracks themselves.

    :param session_directory:
    :param tracking_method:
    :param start:
    :param end:
    :param loom_folder:
    :return: x and y positional traces
    """

    p = pathlib.Path(session_directory)
    lab5 = p / "5_label"

    if tracking_method == "manual":
        print("loading manually tracked")
        x, y = load_raw_track(p, "{}_manual.npy", start, end)

    elif tracking_method == "dlc_1_label":
        print("loading tracking results")
        x, y = load_raw_track(p, "dlc_{}_tracks.npy", start, end)

    elif tracking_method == "dlc_5_label":
        print("loading 5 label tracking results")
        x, y = load_raw_track(lab5, "dlc_{}_tracks.npy", start, end)

    elif tracking_method == "old_school":
        print(f"loading from folders {str(loom_folder)}")
        x, y = load_raw_track(
            session_directory, None, start, end, loom_folder=loom_folder
        )
        x, y = projective_transform_tracks(
            x,
            y,
            load_box_corner_coordinates(session_directory),
            BOX_CORNER_COORDINATES,
        )
    else:
        raise NotImplementedError()

    return np.array(x), np.array(y)


def get_tracking_method(session_directory):
    p = pathlib.Path(session_directory)
    lab5 = p / "5_label"

    if "x_manual.npy" in os.listdir(str(p)):
        method = "manual"

    elif "dlc_x_tracks.npy" in os.listdir(str(p)):
        method = "dlc_1_label"

    elif len(list(lab5.glob("dlc_x_tracks.npy"))) > 0:
        method = "dlc_5_label"

    elif len(list(p.glob("loom0"))) > 0:
        method = "old_school"

    else:
        method = None

    return method


def latency_peak_detect(normalised_x_track, n_stds=2.5):
    """
    Calculates the onset, in samples, of escape using the normalised positional x-trace.

    :param normalised_x_track:
    :return:
    """
    speed = -smooth_speed_from_track(normalised_x_track)[N_SAMPLES_BEFORE:]
    std = np.nanstd(speed[:N_SAMPLES_TO_SHOW])
    all_peak_starts = signal.find_peaks(speed, std * n_stds, width=1)[1][
        "left_ips"
    ]
    if len(all_peak_starts) > 0:
        return all_peak_starts[0] + 200


def latency_peak_detect_s(normalised_x_track):
    """
    Calculates the onset, in seconds, of escape using the normalised positional x-trace.

    :param normalised_x_track:
    :return:
    """
    latency_pd = latency_peak_detect(normalised_x_track)
    if latency_pd is not None:
        latency_pd -= N_SAMPLES_BEFORE
        return latency_pd / FRAME_RATE


def time_to_shelter(normalised_x_track):
    """
    Calculates the time, in seconds, between stimulus onset and arrival at the shelter,
    using the normalised positional x-trace.

    :param normalised_x_track:
    :return:
    """

    smoothed_track = smooth_track(normalised_x_track)
    n_samples_to_shelter = arena_region_crossings.get_next_entry_from_track(
        smoothed_track,
        "shelter",
        "middle",
        LOOMING_STIMULUS_ONSET,
    )
    if n_samples_to_shelter is None:
        return n_samples_to_shelter

    return (n_samples_to_shelter - N_SAMPLES_BEFORE) / FRAME_RATE


def time_in_shelter(normalised_x_track):
    """
    Calculates  the time, in seconds, between arrival at shelter and emerging from it,
     using the normalised positional x-trace.

     :param normalised_x_track:
     :return:
    """
    smoothed_x_track = smooth_track(normalised_x_track)

    start = n_samples_to_reach_shelter(smoothed_x_track)
    if start is None:
        print(
            "mouse never returns to shelter, not computing time to leave shelter"
        )
        return None
    return arena_region_crossings.get_next_entry_from_track(
        smoothed_x_track, "middle", "shelter", start
    )


def n_samples_to_reach_shelter(normalised_x_track):

    """
    Calculates the number of samples between stimulus onset and arrival at the shelter,
    using the normalised positional x-trace.

    :param normalised_x_track:
    :return:
    """
    smoothed_x_track = smooth_track(normalised_x_track)
    n_samples = arena_region_crossings.get_next_entry_from_track(
        smoothed_x_track,
        "shelter",
        "middle",
        LOOMING_STIMULUS_ONSET,
    )
    return n_samples


def n_samples_to_tz_reentry(self):
    """
    Calculates the number of samples between stimulus onset and next tz entry, using the normalised positional x-trace.

    :param normalised_x_track:
    :return:
    """
    return arena_region_crossings.get_next_entry_from_track(
        self.smoothed_x_track, "tz", "middle", LOOMING_STIMULUS_ONSET
    )


def downsample_track(normalised_track, frame_rate):
    """
    Downsamples tracks acquired at >30Hz to allow the display and comparison o escape trajectories from data acquired
    at different frame rates.

    :param normalised_track:
    :param frame_rate:
    :return: downsampled normalised track
    """
    n_points_ori = len(normalised_track)
    n_points_new = int(n_points_ori * (FRAME_RATE / frame_rate))
    track_timebase = (np.arange(len(normalised_track))) / frame_rate
    new_timebase = (np.arange(n_points_new)) / FRAME_RATE
    normalised_track = np.interp(
        new_timebase, track_timebase, normalised_track
    )
    return normalised_track


def normalised_speed_from_track(normalised_track):
    normalised_speed = np.concatenate([[np.nan], np.diff(normalised_track)])
    return normalised_speed


def smooth_track(normalised_track):
    smoothed_track = gaussian_filter(normalised_track, 2)
    return smoothed_track


def smooth_speed_from_track(normalised_track):
    smoothed_track = smooth_track(normalised_track)
    return np.diff(smoothed_track)


def smooth_acceleration_from_track(normalised_track):
    smoothed_speed = smooth_speed_from_track(normalised_track)
    return np.roll(np.diff(smoothed_speed), 2)


def get_peak_speed(normalised_x_track, return_loc=False):

    """
    Calculates the peak speed (towards shelter) reached in the analysis window following presentation of a stimulus.
    :param normalised_x_track:
    :param return_loc:
    :return:
    """

    peak_speed, arg_peak_speed = get_peak_speed_and_latency(normalised_x_track)
    peak_speed = peak_speed * FRAME_RATE * ARENA_SIZE_CM

    if return_loc:
        return peak_speed, arg_peak_speed

    return peak_speed


def load_track_csv(loom_folder):
    track_path = os.path.join(loom_folder, "tracks.csv")
    df = pd.read_csv(track_path, sep="\t")
    x_pos = np.array(df["x_position"])
    y_pos = np.array(df["y_position"])
    return x_pos, y_pos


def get_peak_speed_and_latency(normalised_track):
    """
    :return peak_speed:
    :return arg_peak: the frame number of the peak speed
    """
    filtered_track = gaussian_filter(normalised_track, 3)
    distances = np.diff(filtered_track)
    peak_speed = np.nanmin(
        distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]
    )
    arg_peak = np.argmin(
        distances[CLASSIFICATION_WINDOW_START:CLASSIFICATION_WINDOW_END]
    )
    return -peak_speed, arg_peak + CLASSIFICATION_WINDOW_START


def projective_transform_tracks(
    Xin,
    Yin,
    observed_box_corner_coordinates,
    target_box_corner_coordinates=BOX_CORNER_COORDINATES,
):

    """
    To correct for camera angle artifacts, coordinates of the arena and its known real geometry are used to
    get a projective transform that can be applied to positional tracks or raw videos.


        x2y2-------------x4y4        b--------------d
       /                  /          |              |
      /                  /           |              |
     /                  /            |              |
    x1y1-------------x3y3            a--------------c

    target_box_corner_coordinates = [a=[0, 240],
                                     b=[0, 0],
                                     c=[600, 240],
                                     d=[600, 0]]


    observed_box_corner_coordinates = [e=[ 19.59140375, 296.09195599],
                                       f=[ 64.97987079, 124.3052263],
                                       g=[628.02667714, 284.60120484],
                                       h=[596.42711148, 102.47279911]]
    :param Xin:
    :param Yin:
    :param observed_box_corner_coordinates:
    :param target_box_corner_coordinates:
    :return: x and y positional traces transformed into standard space
    """
    p = get_inverse_projective_transform(
        dest=observed_box_corner_coordinates,
        src=np.array(target_box_corner_coordinates),
    )
    new_track_x = []
    new_track_y = []
    for x, y in zip(Xin, Yin):
        inverse_mapped = p.inverse([x, y])[0]
        new_track_x.append(inverse_mapped[0])
        new_track_y.append(inverse_mapped[1])
    return new_track_x, new_track_y
