import os
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from nptdms import TdmsFile

from looming_spots.constants import (
    RAW_DATA_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    AUDITORY_STIMULUS_CHANNEL_ADDED_DATE,
)
from looming_spots import exceptions

"""

For all functions relating to moving raw data and processed files around

"""


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


def load_pd_and_clock_raw(directory):  # TODO: raw loading should be entirely extracted to load module
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
    auditory_path = Path(directory) / 'auditory_stimulus.npy'
    if not os.path.isfile(str(auditory_path)):
        pd, clock, auditory = load_pd_and_clock_raw(directory)
        clock_ups = get_clock_ups(clock, pd_threshold)
        np.save(str(auditory_path), auditory[clock_ups])
        return auditory[clock_ups]
    else:
        return np.load(str(auditory_path))


def get_clock_ups(clock, threshold=2.5):
    clock_on = (clock > threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    return clock_ups
