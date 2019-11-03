import os
import subprocess
import sys
import numpy as np
import pims

import looming_spots.preprocess.io


class NoPdError(Exception):
    pass


def compare_pd_and_video(directory):
    pd_trace = looming_spots.preprocess.io.load_pd_on_clock_ups(directory)
    n_samples_pd = len(pd_trace)
    video_path = os.path.join(directory, "camera.mp4")
    n_samples_video = len(pims.Video(video_path))

    print(
        "pd found {} samples, there are {} frames in the video".format(
            n_samples_pd, n_samples_video
        )
    )

    if n_samples_pd != n_samples_video:
        if n_samples_pd == 0:
            raise NoPdError
        n_samples_ratio = round(n_samples_pd / n_samples_video, 2)
        if n_samples_ratio.is_integer():
            print("downsampling by factor {}".format(n_samples_ratio))
            print(n_samples_pd, n_samples_video, n_samples_ratio)
            downsampled_ai = pd_trace[:: int(n_samples_ratio)]
            save_path = os.path.join(directory, "AI_corrected")
            np.save(save_path, downsampled_ai)


def convert_to_mp4(
    name, directory, remove_avi=False
):  # TODO: remove duplication
    mp4_path = os.path.join(directory, name[:-4] + ".mp4")
    avi_path = os.path.join(directory, name)
    if os.path.isfile(mp4_path):
        print("{} already exists".format(mp4_path))
        if remove_avi:  # TEST this
            if os.path.isfile(avi_path):
                print(
                    "{} present in processed data, deleting...".format(
                        avi_path
                    )
                )
                os.remove(avi_path)
    else:
        print("Creating: " + mp4_path)
        convert_avi_to_mp4(avi_path)


def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path[:-4] + ".mp4"
    print("avi: {} mp4: {}".format(avi_path, mp4_path))

    supported_platforms = ["linux", "windows"]

    if sys.platform == "linux":
        cmd = "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
            avi_path, mp4_path
        )

    elif sys.platform == "windows":  # TEST: on windows
        cmd = "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
            avi_path, mp4_path
        ).split(
            " "
        )

    else:
        raise (
            OSError(
                "platform {} not recognised, expected one of {}".format(
                    sys.platform, supported_platforms
                )
            )
        )

    subprocess.check_call([cmd], shell=True)


class NoProcessedVideoError(Exception):
    def __str__(self):
        print("there is no mp4 video")
