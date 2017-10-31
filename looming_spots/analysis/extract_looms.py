import os
import numpy as np
import skvideo
from datetime import datetime
import scipy.signal, scipy.misc
import skvideo.io
import configobj
import cv2
from looming_spots.analysis import loom_exceptions
# from looming_spots.ref_builder.viewer import Viewer


def is_datetime(folder_name):
    try:
        date_time = datetime.strptime(folder_name, '%Y%m%d_%H_%M_%S')
        print('string is in date_time format: {}'.format(date_time))
        return True
    except ValueError:  # FIXME: custom exception required
        return False


def get_fpath(directory, extension):
    for item in os.listdir(directory):
        if extension in item:
            return os.path.join(directory, item)

    raise loom_exceptions.FileNotPresentError('there is no file with extension: {}'
                                              ' in directory {}'.format(extension, directory))


def load_ai(directory, pd_threshold=2.5):
    path = get_fpath(directory, '.bin')
    raw = np.fromfile(path, dtype='double')
    raw_reshaped = raw.reshape(int(raw.shape[0]/2), 2)
    raw_ai = raw_reshaped[:, 0]
    raw_clock = raw_reshaped[:, 1]
    clock_on = (raw_clock > pd_threshold).astype(int)
    clock_ups = np.where(np.diff(clock_on) == 1)[0]
    print('number of clock ups found: {}'.format(len(clock_ups)))
    return raw_ai[clock_ups]


def get_context_from_stimulus_mat(directory):
    stimulus_path = os.path.join(directory, 'stimulus.mat')
    stimulus_params = scipy.io.loadmat(stimulus_path)['params']
    dot_locations = [x[0] for x in stimulus_params[0][0] if len(x[0]) == 2]  # only spot position has length 2
    return 'B' if any(1240 in x for x in dot_locations) else 'A'  # b is the only condition where a position can be 1240


def filter_raw_pd_trace(pd_trace, fs=10000):
    b1, a1 = scipy.signal.butter(3, 1000/fs*2, 'low')
    pd_trace = scipy.signal.lfilter(b1, a1, pd_trace)
    return pd_trace


def get_loom_idx(filtered_ai):
    loom_on = (filtered_ai > 1).astype(int)
    loom_ups = np.diff(loom_on) == 1
    # loom_downs = np.diff(loom_on) == -1
    loom_starts = np.where(loom_ups)[0]
    return loom_starts


def get_session_label(directory, n_habituation_looms=120):
    ai = load_ai(directory)
    filtered_ai = filter_raw_pd_trace(ai)
    loom_starts = get_loom_idx(filtered_ai)
    print("there are {} looms".format(len(loom_starts)))
    if len(loom_starts) == n_habituation_looms:
        return 'habituation_only'
    elif len(loom_starts) < 50:
        return 'test_only'
    elif len(loom_starts) > n_habituation_looms:
        return 'habituation_and_test'


def get_manual_looms(loom_idx, n_looms_per_stimulus=5, n_auto_looms=120, ILI_ignore_n_samples=1300):

    ilis = np.roll(loom_idx, -1) - loom_idx
    if len(ilis) > n_auto_looms:
        first_loom_idx = n_auto_looms
        n_manual_looms = len(ilis)-n_auto_looms
    elif len(ilis) == n_auto_looms:
        print('HABITUATION ONLY, {} looms detected'.format(len(ilis)))
        return []
    else:
        first_loom_idx = 0
        n_manual_looms = len(ilis)

    remainder = n_manual_looms % 5
    if remainder != 0:
        print("expected looms to be in multiple of: {}, got remainder: {}, skipping".format(n_looms_per_stimulus,
                                                                                            remainder))
        return []
    manual_looms = np.arange(first_loom_idx, first_loom_idx+n_manual_looms, 5)
    if len(manual_looms) > 5:
        print('way too many stimuli to be correct: {}, skipping'.format(len(manual_looms)))
        return []
    return loom_idx[manual_looms]


def extract_loom_videos(directory, manual_loom_indices):
    for loom_number, loom_idx in enumerate(manual_loom_indices):
        extract_loom_video(directory, loom_idx, loom_number)


def extract_loom_video(directory, loom_start, loom_number, n_samples_before=200, n_samples_after=200, shape=(480, 640)):
    loom_video_path = os.path.join(directory, 'loom{}.h264'.format(loom_number))
    if os.path.isfile(loom_video_path):
        return
    video_path = get_fpath(directory, '.mp4')
    rdr = skvideo.io.vreader(video_path)
    loom_video = np.zeros((n_samples_before+n_samples_after+1, shape[0], shape[1], 3))
    a = 0
    for i, frame in enumerate(rdr):
        if (loom_start-n_samples_before) < i < (loom_start + n_samples_after):
            loom_video[a, :, :, :] = frame
            a += 1
    skvideo.io.vwrite(loom_video_path, loom_video)


def auto_extract_all(directory, overwrite=False):
    if any('loom' in fname for fname in os.listdir(directory)) and not overwrite:
        raise loom_exceptions.LoomVideosAlreadyExtractedError(directory)
    ai = load_ai(directory)
    ai_filtered = filter_raw_pd_trace(ai)
    all_loom_idx = get_loom_idx(ai_filtered)
    manual_loom_indices = get_manual_looms(all_loom_idx)
    extract_loom_videos(directory, manual_loom_indices)


def get_frame(rdr_path, idx):
    idx = int(idx)
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i == idx:
            return frame


def make_ref(directory, video_fname='camera.mp4', mirror_plane=450):
    left_idx, right_idx = get_ref_index(directory)
    video_path = os.path.join(directory, video_fname)
    left_frame, right_frame = get_reference_frames(video_path, left_idx, right_idx)
    ref = make_reference_frame(left_frame, right_frame, mirror_plane)
    return np.mean(ref, axis=2)


def get_reference_frames(rdr_path, frame_idx_left, frame_idx_right):
    mouse_left_frame = get_frame(rdr_path, frame_idx_left)
    mouse_right_frame = get_frame(rdr_path, frame_idx_right)
    return mouse_left_frame, mouse_right_frame


def make_reference_frame(left_frame, right_frame, mirror_plane=450):
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(left_frame[:, :mirror_plane, :])
    # plt.subplot(122)
    # plt.imshow(right_frame[:, mirror_plane:, :])
    # plt.show()
    composite_frame = np.zeros_like(left_frame)
    composite_frame[:, :mirror_plane, :] = left_frame[:, :mirror_plane, :]
    composite_frame[:, mirror_plane:, :] = right_frame[:, mirror_plane:, :]
    return composite_frame


def get_ref_index(directory):
    meta_path = os.path.join(directory + '/metadata.txt')
    print(meta_path)
    if not os.path.isfile(meta_path):
        msg = 'The indices cannot be obtained from a metafile, would you like to make a reference frame manually?'
        msg += "WARNING: THIS WILL OVERWRITE THE METADATA FILE"
        answer = input(msg)
        if answer == 'yes':
            pass
            # Viewer(directory, video_name='loom{}.h264')

    metadata = configobj.ConfigObj(meta_path)
    ref_left = int(metadata['reference_frame']['left_frame_idx'])
    ref_right = int(metadata['reference_frame']['right_frame_idx'])
    return ref_left, ref_right


def get_reference_frame_details(directory):
    meta_path = os.path.join(directory + '/metadata.txt')
    metadata = configobj.ConfigObj(meta_path)
    left_idx = int(metadata['reference_frame']['left_frame_idx'])
    left_path = metadata['reference_frame']['left_video_name']
    right_idx = int(metadata['reference_frame']['right_frame_idx'])
    right_path = metadata['reference_frame']['right_video_name']
    return


def load_ref_from_metadata(directory):

    left_frame = get_frame(os.path.join(directory, left_path), left_idx)
    right_frame = get_frame(os.path.join(directory, right_path), right_idx)

    ref = make_reference_frame(left_frame, right_frame)
    save_frame_as_png(ref)


def save_frame_as_png(frame, directory):
    ref_array = np.mean(frame, axis=2)
    save_fpath = os.path.join(directory, 'ref.png')
    print('saving reference frame to: {}'.format(save_fpath))
    scipy.misc.imsave(save_fpath, ref_array, format='png')


def load_video(rdr, ref=None):
    video = []
    if ref:
        video.append(ref)
    for frame in rdr:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.append(gray_frame)
    return video


def add_ref_to_all(mouse_dir):
    for dirName, subdirList, fileList in os.walk(mouse_dir):
        for subdir in subdirList:
            if is_datetime(subdir):
                path = os.path.join(mouse_dir, subdir)
                add_ref_to_all_loom_videos(path)


def add_ref_to_all_loom_videos(directory):
    if not any('loom' in fname for fname in os.listdir(directory)):
        return 'no loom videos in directory: {}'.format(directory)
    for fname in os.listdir(directory):
        if 'loom' in fname and '.h264' in fname:
            vid_path = os.path.join(directory, fname)
            ref = make_ref(directory, fname)
            rdr = skvideo.io.vreader(vid_path)
            vid = load_video(rdr, ref)
            save_video(vid, vid_path)


def save_video(video, path):
    skvideo.io.vwrite(path, np.array(video))
