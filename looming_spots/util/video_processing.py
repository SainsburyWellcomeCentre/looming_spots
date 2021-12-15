import subprocess
import sys

import cv2
import numpy as np
import skvideo
import skvideo.io
import pims


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
        new_video[i, :, :] = frame[origin[1]: height, origin[0]: width]
    return new_video


def convert_avi_to_mp4(source, dest, overwrite=False, crf=18):
    if not dest.exists() or overwrite:

        print(f"avi: {source} mp4: {dest}")

        supported_platforms = ["linux", "windows"]

        if sys.platform == "linux":
            cmd = (
                f"ffmpeg -i {str(source)} -c:v mpeg4 -preset veryfast -crf {crf} -b 5000k -c:v copy -c:a copy {str(dest)}"
            )

        elif sys.platform == "windows":  # TEST: on windows
            cmd = (
                f"ffmpeg -i {str(source)} -c:v mpeg4 -preset veryfast -crf {crf} -b 5000k {str(dest)}".split(" ")
            )

        else:
            raise (
                OSError(
                    f"platform {sys.platform} not recognised, expected one of {supported_platforms}"
                )
            )

        subprocess.check_call(cmd, shell=True)
    else:
        print(f"file exists at {dest} and overwrite set to false... skipping...")


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
