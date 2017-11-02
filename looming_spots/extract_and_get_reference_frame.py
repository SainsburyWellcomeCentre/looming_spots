import os
from looming_spots.analysis import extract_looms


def main(directory, video_fname='camera.mp4'):
    for dirName, subdirList, fileList in os.walk(directory):
        for subdir in subdirList:
            if extract_looms.is_datetime(subdir):
                vid_dir = os.path.join(directory, dirName, subdir)
                vid_path = os.path.join(vid_dir, video_fname)
                if os.path.isfile(vid_path):
                    print(vid_dir)
                    os.chdir(vid_dir)
                    extract_looms.auto_extract_all(vid_dir)
                    os.chdir(directory)

                # extract_looms.add_ref_to_all_loom_videos(directory)

if __name__ == '__main__': main('/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/171030/')
