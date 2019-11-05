import os
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import copyfile
import numpy as np
from nptdms import TdmsFile

from looming_spots.db.constants import (
    RAW_DATA_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    AUDITORY_STIMULUS_CHANNEL_ADDED_DATE,
    OLD_RAW_DIRECTORY,
)
from looming_spots import exceptions

"""

for all functions relating to moving raw data and processed files around

"""


def sync_raw_and_processed_data(
    raw_directory=RAW_DATA_DIRECTORY,
    processed_directory=PROCESSED_DATA_DIRECTORY,
):
    cmd = "rsync -tvr --chmod=D2775,F664 --exclude='*.avi' --exclude='*.imec*' --exclude='.mp4' {}/* {}".format(
        raw_directory, processed_directory
    )
    subprocess.call(cmd, shell=True)


def sync_raw_spine_and_winstor(
    raw_directory=OLD_RAW_DIRECTORY, processed_directory=RAW_DATA_DIRECTORY
):
    cmd = "rsync -tvr --chmod=D2775,F664 {}/* {}".format(
        raw_directory, processed_directory
    )
    subprocess.call(cmd, shell=True)


def manually_correct_ai(directory, start, end):
    ai = load_pd_on_clock_ups(directory)
    ai[start:end] = np.median(ai)
    save_path = os.path.join(directory, "AI_corrected")
    np.save(save_path, ai)


def auto_fix_ai(
    directory, n_samples_to_replace=500, screen_off_threshold=0.02
):
    ai = load_pd_on_clock_ups(directory)
    screen_off_locs = np.where(ai < screen_off_threshold)[
        0
    ]  # TODO: remove hard var

    if len(screen_off_locs) == 0:
        return

    start = screen_off_locs[0]
    end = start + n_samples_to_replace
    ai[start:end] = np.median(ai)
    save_path = os.path.join(directory, "AI_corrected")
    np.save(save_path, ai)
    auto_fix_ai(directory, n_samples_to_replace=n_samples_to_replace)


def load_pd_and_clock_raw(
    directory
):  # TODO: raw loading should be entirely extracted to load module
    if "AI.tdms" in os.listdir(directory):
        path = os.path.join(directory, "AI.tdms")
        tdms_file = TdmsFile(path)
        all_channels = tdms_file.group_channels("acq_task")
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
        print(directory)
        pd, clock, auditory = load_pd_and_clock_raw(directory)
        clock_ups = get_clock_ups(clock, pd_threshold)
        print(f"number of clock ups found: {len(clock_ups)}")
        if len(clock_ups) < 12:
            raise exceptions.PdTooShortError()
        return pd[clock_ups]


def load_auditory_on_clock_ups(directory, pd_threshold=2.5):
    pd, clock, auditory = load_pd_and_clock_raw(directory)
    clock_ups = get_clock_ups(clock, pd_threshold)
    print(f"number of clock ups found: {len(clock_ups)}")
    return auditory[clock_ups]


def get_clock_ups(clock, threshold=2.5):
    clock_on = (clock > threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    return clock_ups


def get_all_tracks(raw_directory=RAW_DATA_DIRECTORY, dry=False):
    """
    this just allows dlc tracks to be copied over to processed directory after runnning
    deprecated, use rsync function

    :param raw_directory:
    :param dry:
    :return:
    """
    p = Path(raw_directory)
    track_paths = p.rglob("*tracks.npy")
    for track_path in track_paths:
        old_path = str(track_path)
        mouse_id = old_path.split("/")[-3]
        date = old_path.split("/")[-2]
        fname = old_path.split("/")[-1]
        new_path = f"{PROCESSED_DATA_DIRECTORY}/{mouse_id}/{date}/{fname}"
        if not os.path.isfile(new_path):
            print(f"copying {old_path} to {new_path}")
            if not dry:
                copyfile(old_path, new_path)
