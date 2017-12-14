import os

import looming_spots.analysis.extract_looms
import looming_spots.analysis.photodiode
from looming_spots.analysis import extract_looms
from looming_spots.ref_builder import viewer


def loop_through_directories(root, function):
    for dirName, subdirList, fileList in os.walk(root):
        for subdir in subdirList:
            if looming_spots.analysis.extract_looms.is_datetime(subdir):
                vid_dir = os.path.join(root, dirName, subdir)
                print(vid_dir)
                vid_path = os.path.join(vid_dir, 'loom0.h264')
                if os.path.isfile(vid_path):
                    os.chdir(vid_dir)
                    function(vid_dir)
                    os.chdir(root)


def main(directory):
    loop_through_directories(directory, viewer.Viewer)


if __name__ == '__main__': main('/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/171030/')
