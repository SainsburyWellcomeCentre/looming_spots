import os
import subprocess

import looming_spots.util.generic_functions
from looming_spots.constants import PROCESSED_DATA_DIRECTORY
from configobj import ConfigObj


PYTHON_PATH = "/home/slenzi/miniconda3/envs/pyper_env/bin/python2.7"
PYPER_COMMAND = "pyper.cli.tracking_cli"


def pyper_cli_track(directory, ref_fname="ref.npy"):
    for fname in os.listdir(directory):
        if "loom" in fname and ".h264" in fname:
            video_path = os.path.join(directory, fname)
            print(video_path)
            subprocess.check_call(
                "{} -m {} {} --bg-fname {}".format(
                    PYTHON_PATH, PYPER_COMMAND, video_path, ref_fname
                ),
                shell=True,
            )


def pyper_cli_track_trial(video_path, ref_fname="ref.npy"):
    subprocess.check_call(
        "{} -m {} {} --bg-fname {}".format(
            PYTHON_PATH, PYPER_COMMAND, video_path, ref_fname
        ),
        shell=True,
    )


def pyper_cli_track_video_path(
    session_path, start=0, n_frames=18000, video_name="camera.mp4"
):
    video_path = os.path.join(session_path, video_name)
    mtd_path = os.path.join(session_path, "metadata.cfg")
    config = ConfigObj(mtd_path, unrepr=False)

    if start is None:
        start = int(config["track_start"])

    end = start + n_frames
    subprocess.check_call(
        "{} -m {} {} -f {} -t {}".format(
            PYTHON_PATH, PYPER_COMMAND, video_path, start, end
        ),
        shell=True,
    )


def track_all_sessions_all_mice(root=PROCESSED_DATA_DIRECTORY):
    for dirName, subdirList, fileList in os.walk(root):
        for subdir in subdirList:
            if looming_spots.util.generic_functions.is_datetime(subdir):
                target_dir = os.path.join(root, dirName, subdir)
                if "ref.npy" in os.listdir(target_dir):
                    pyper_cli_track(target_dir)
