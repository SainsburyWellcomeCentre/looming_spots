import os
import subprocess
from pathlib import Path

import numpy as np
from datetime import datetime

from looming_spots import exceptions
from looming_spots.constants import (
    ORDERED_ACQUISITION_CHANNEL_LABELS,
    PROCESSED_OUTPUT_VARIABLE_LABELS,
    AUDITORY_STIMULUS_CHANNEL_ADDED_DATE, RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY)
from nptdms import TdmsFile


if "photodetector" in ORDERED_ACQUISITION_CHANNEL_LABELS:
    from photometry import demodulation


def load_all_channels_raw(
    directory, data_labels=ORDERED_ACQUISITION_CHANNEL_LABELS
):
    if "AI.tdms" in os.listdir(directory):  # TODO: handle corrupt tdms files
        path = os.path.join(directory, "AI.tdms")
        tdms_file = TdmsFile(path)
        all_channels = tdms_file.group_channels("acq_task")
        return {key: c.data for c, key in zip(all_channels, data_labels)}
    elif "AI.bin" in os.listdir(directory):
        return load_from_ai_bin(directory)


def load_from_ai_bin(directory):

    path = os.path.join(directory, "AI.bin")
    raw_ai = np.fromfile(path, dtype="double")

    recording_date = datetime.strptime(
        os.path.split(directory)[-1], "%Y%m%d_%H_%M_%S"
    )
    raw_dict = {}
    if recording_date > datetime(2019, 1, 25):
        raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 3), 3)
        raw_dict.setdefault("photodiode", raw_ai[:, 0])
        raw_dict.setdefault("clock", raw_ai[:, 1])
        raw_dict.setdefault("auditory_stimulus", raw_ai[:, 2])
        return raw_dict

    raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 2), 2)
    raw_dict.setdefault("photodiode", raw_ai[:, 0])
    raw_dict.setdefault("clock", raw_ai[:, 1])
    raw_dict.setdefault("auditory_stimulus", np.zeros(len(raw_ai[:, 1])))

    return raw_dict


def load_all_channels_on_clock_ups(directory):
    try:
        return _load_downsampled_data(directory)
    except FileNotFoundError as e:
        print(e)
        print("attempting to get data from raw...")

        raw_data_dict = load_all_channels_raw(directory)
        clock = raw_data_dict["clock"]

        clock_ups = get_clock_ups(clock, threshold=2.5)
        processed_data_dict = {
            k: data[clock_ups] for k, data in raw_data_dict.items()
        }
        if "photodetector" in raw_data_dict:  # TODO: extract
            (
                signal,
                background,
                bg_fit,
                delta_f,
            ) = demodulation.lerner_deisseroth_preprocess(
                raw_data_dict["photodetector"],
                raw_data_dict["led211"],
                raw_data_dict["led531"],
            )
            processed_data_dict.update(
                [
                    ("signal", signal[clock_ups]),
                    ("background", background[clock_ups]),
                    ("bg_fit", bg_fit[clock_ups]),
                    ("delta_f", delta_f[clock_ups]),
                ]
            )

        save_downsampled_data(directory, processed_data_dict)

        return processed_data_dict


def _load_downsampled_data(directory):
    data_keys = (
        ORDERED_ACQUISITION_CHANNEL_LABELS + PROCESSED_OUTPUT_VARIABLE_LABELS
    )
    return {k: np.load(get_full_path(directory, k)) for k in data_keys}


def get_full_path(directory, key):
    return os.path.join(directory, f"{key}.npy")


def save_downsampled_data(directory: str, data_dict: dict):
    print("saving downsampled data to files")
    for k, v in data_dict.items():
        full_path = get_full_path(directory, k)
        if not os.path.isfile(full_path):
            np.save(full_path, v)
    for k in set(list(data_dict.keys())).symmetric_difference(
        set(
            ORDERED_ACQUISITION_CHANNEL_LABELS
            + PROCESSED_OUTPUT_VARIABLE_LABELS
        )
    ):
        full_path = get_full_path(directory, k)
        np.save(full_path, [0])


def get_clock_ups(clock, threshold=2.5):
    clock_on = (clock > threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    return clock_ups


def load_pd_and_clock_raw(
    directory,
):
    if "AI.tdms" in os.listdir(directory):

        path = os.path.join(directory, "AI.tdms")
        tdms_file = TdmsFile(path)
        all_channels = tdms_file.groups()[0].channels()
        pd, clock, auditory, pmt, led211, led531 = (
            c.data for c in all_channels
        )
        return pd, clock, auditory

    else:
        path = os.path.join(directory, "AI.bin")
        raw_ai = np.fromfile(path, dtype="double")

    recording_date = datetime.strptime(
        os.path.split(directory)[-1], "%Y%m%d_%H_%M_%S"
    )

    if recording_date > AUDITORY_STIMULUS_CHANNEL_ADDED_DATE:
        raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 3), 3)
        pd = raw_ai[:, 0]
        clock = raw_ai[:, 1]
        auditory = raw_ai[:, 2]
        return pd, clock, auditory

    raw_ai = raw_ai.reshape(int(raw_ai.shape[0] / 2), 2)
    pd = raw_ai[:, 0]
    clock = raw_ai[:, 1]
    return pd, clock, []  # FIXME: hack


def load_pd_on_clock_ups(directory, pd_threshold=2.5):
    if "AI_corrected.npy" in os.listdir(directory):
        print("loading corrected/downsampled ai")
        path = os.path.join(directory, "AI_corrected.npy")
        downsampled_processed_ai = np.load(path)
        return downsampled_processed_ai
    else:
        pd, clock, auditory = load_pd_and_clock_raw(directory)
        clock_ups = get_clock_ups(clock, pd_threshold)
        if len(clock_ups) < 12:
            raise exceptions.PdTooShortError()
        return pd[clock_ups]


def load_auditory_on_clock_ups(directory, pd_threshold=2.5):
    auditory_path = Path(directory) / "auditory_stimulus.npy"
    if not os.path.isfile(str(auditory_path)):
        pd, clock, auditory = load_pd_and_clock_raw(directory)
        clock_ups = get_clock_ups(clock, pd_threshold)
        np.save(str(auditory_path), auditory[clock_ups])
        return auditory[clock_ups]
    else:
        return np.load(str(auditory_path))


def sync_raw_and_processed_data(
    raw_directory=RAW_DATA_DIRECTORY,
    processed_directory=PROCESSED_DATA_DIRECTORY,
):
    """
    Runs rsync command from python to copy across necessary data from raw to processed directories. May require
    modification for use on Windows. Tested on Linux.

    :param raw_directory:
    :param processed_directory:
    :return:
    """
    cmd = "rsync -tvr --chmod=D2775,F664 --exclude='*.avi' --exclude='*.imec*' --exclude='.mp4' {}/* {}".format(
        raw_directory, processed_directory
    )
    subprocess.call(cmd, shell=True)