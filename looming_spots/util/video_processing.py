import os
import cv2
import numpy as np
import scipy.misc
import skvideo
import skvideo.io


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


def save_frame_as_array(frame, directory):
    ref_array = np.mean(frame, axis=2)
    ref_path = os.path.join(directory, 'ref.npy')
    print('refarray:{}'.format(ref_array))
    print('saving reference frame to: {}'.format(ref_path))
    np.save(ref_path, ref_array)


def save_frame_as_png(frame, directory):
    ref_array = np.mean(frame, axis=2)
    save_fpath = os.path.join(directory, 'ref.png')
    print('saving reference frame to: {}'.format(save_fpath))
    scipy.misc.imsave(save_fpath, ref_array, format='png')


def get_frame(rdr_path, idx):
    idx = int(idx)
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i == idx:
            return frame


def get_frames(rdr_path, idx):
    frames = []
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i in idx:
            frames.append(frame)
    return frames


def save_video(video, path):
    skvideo.io.vwrite(path, video)
