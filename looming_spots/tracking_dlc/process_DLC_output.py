import numpy as np
import pandas as pd

from looming_spots.tracking_dlc.constants import BODYPART_LABELS


def process_DLC_output(file_path, config_path_label):
    clean_data = load_clean_data(file_path)
    median_label_tracks = get_median_position_from_labels(
        clean_data, config_path_label
    )
    return median_label_tracks


def load_clean_data(file_path):
    """
    first function, loads data from os, converts it to usable form
    universal processing to all tracks regardless of where or what they came from
    """
    raw_dlc_hdf5_df = load_tracks_from_h5(file_path)
    tracks_df = raw_dlc_hdf5_df[raw_dlc_hdf5_df.keys()[0][0]]

    return tracks_df


def load_tracks_from_h5(path):
    """
    loads dataframe from hdf5 file
    :param path:
    :return:
    """
    loaded_path = pd.read_hdf(path)
    return loaded_path


def replace_low_likelihood_as_nan(
    extracted_tracks, likelihood_threshold=0.9999
):
    body_part_labels = np.unique([k[0] for k in extracted_tracks.keys()])
    for body_part_label in body_part_labels:
        body_part = extracted_tracks[body_part_label]
        mask = body_part["likelihood"] <= likelihood_threshold
        extracted_tracks.loc[mask, body_part_label] = np.nan
    return extracted_tracks


def get_first_and_last_likely_frame(
    extracted_tracks, body_part_label, n_samples=500
):
    body_part = extracted_tracks[body_part_label]
    mask = body_part["likelihood"] == 1
    for i in range(len(mask)):
        end = min(i + n_samples, len(mask))
        window = mask[i:end]
        if np.mean(window) == 1:
            cricket_entry = i
            return cricket_entry, cricket_entry + 90000


def get_median_position_from_labels(
    dlc_clean_data, config_path_label="one_label_transform"
):
    body_part_labels = BODYPART_LABELS[config_path_label]
    body_parts = {
        body_part_label: dlc_clean_data[body_part_label]
        for body_part_label in body_part_labels
    }

    df_y = pd.DataFrame(
        {
            body_part_label: body_part["y"]
            for body_part_label, body_part in body_parts.items()
        }
    )
    df_x = pd.DataFrame(
        {
            body_part_label: body_part["x"]
            for body_part_label, body_part in body_parts.items()
        }
    )

    median_y_df = df_y.median(axis=1)
    median_x_df = df_x.median(axis=1)
    median_positions_df = pd.DataFrame({"x": median_x_df, "y": median_y_df})
    return median_positions_df
