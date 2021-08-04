import os
import subprocess
import sys

import cv2
import numpy as np
import skvideo
import skvideo.io
import pims

from looming_spots.io.load import load_all_channels_on_clock_ups


def load_video_from_path(vid_path):
    rdr = skvideo.io.vreader(vid_path)
    vid = load_video_from_rdr(rdr, ref=None)
    return vid


def load_video_from_rdr(rdr, ref=None):
    video = []
    if ref is not None:
        video.append(ref)
    for frame in rdr:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.append(gray_frame)
    return np.array(video)


def get_frames(rdr_path, idx):
    frames = []
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i in idx:
            frames.append(frame)
    return frames


def save_video(video, path):
    skvideo.io.vwrite(path, video)


def crop_video(video, width, height, origin=(0, 0)):
    n_frames = video.shape[0]
    new_video = np.full(
        (n_frames, height - origin[1], width - origin[0]), np.nan
    )

    for i, frame in enumerate(video):
        new_video[i, :, :] = frame[origin[1] : height, origin[0] : width]
    return new_video


def compare_pd_and_video(directory):
    photodiode_trace = load_all_channels_on_clock_ups(directory)['photodiode']
    n_samples_pd = len(photodiode_trace)
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
            downsampled_ai = photodiode_trace[:: int(n_samples_ratio)]
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
        cmd = (
            "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
                avi_path, mp4_path
            )
        )

    elif sys.platform == "windows":  # TEST: on windows
        cmd = (
            "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
                avi_path, mp4_path
            ).split(" ")
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


def extract_video(fin_path, fout_path, start, end):
    v = pims.Video(fin_path)
    out_video = v[start:end]
    print(f"writign tp {fout_path}")
    skvideo.io.vwrite(fout_path, out_video)


class NoPdError(Exception):
    pass


class NoProcessedVideoError(Exception):
    def __str__(self):
        print("there is no mp4 video")
