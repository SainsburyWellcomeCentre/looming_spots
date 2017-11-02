import os
import skvideo
import numpy as np
import configobj
import cv2


def get_frame(rdr_path, idx):
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i == idx:
            return frame


def get_reference_frames(rdr_path, frame_idx_left, frame_idx_right):
    mouse_left_frame = get_frame(rdr_path, frame_idx_left)
    mouse_right_frame = get_frame(rdr_path, frame_idx_right)
    return mouse_left_frame, mouse_right_frame


def make_reference_frame(left_frame, right_frame, x_pos):
    composite_frame = np.zeros_like(left_frame)
    composite_frame[:, :x_pos, :] = left_frame[:, :x_pos, :]
    composite_frame[:, x_pos:, :] = right_frame[:, x_pos:, :]
    return composite_frame


def get_ref_index(directory):
    meta_path = os.path.join(directory + '/metadata.txt')
    print(meta_path)
    if os.path.isfile(meta_path):
        metadata = configobj.ConfigObj(meta_path)
        ref_left = int(metadata['left'])
        ref_right = int(metadata['right'])
    else:
        msg = 'The indices cannot be obtained from a metafile'
        msg += "please enter values for left and right images as a tuple"
        answer = input(msg)
        ref_left, ref_right = (int(e) for e in answer.strip('[]() ').split(','))
    return ref_left, ref_right


def make_ref(directory, video_fname, mirror_plane=450):
    left_idx, right_idx = get_ref_index(directory)
    video_path = os.path.join(directory, video_fname)
    left_frame, right_frame = get_reference_frames(video_path, left_idx, right_idx)
    ref = make_reference_frame(left_frame, right_frame, mirror_plane)
    return np.mean(ref, axis=2)


def load_video_with_ref(rdr, ref):
    video = [ref]
    for frame in rdr:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.append(gray_frame)
    return video


def save_video(video, path):
    skvideo.io.vwrite(path, np.array(video))


def add_ref_to_all(mouse_dir):
    for dirName, subdirList, fileList in os.walk(mouse_dir):
        for subdir in subdirList:
            if is_datetime(subdir):
                path = os.path.join(mouse_dir, subdir)
                for fname in os.listdir(path):
                    if 'loom' in fname and '.h264' in fname:
                        vid_path = os.path.join(path, fname)
                        l_idx, r_idx = extract_ref_frame.get_ref_index(path)
                        ref = extract_ref_frame.make_ref(path, fname)
                        rdr = skvideo.io.vreader(vid_path)
                        vid = extract_ref_frame.load_video_with_ref(rdr, ref)
                        extract_ref_frame.save_video(vid, vid_path)