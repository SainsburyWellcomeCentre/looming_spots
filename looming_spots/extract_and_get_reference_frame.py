import os
from looming_spots.analysis import extract_looms


def main(directory, video_path='camera.mp4'):
    for dirName, subdirList, fileList in os.walk(directory):
        for subdir in subdirList:
            if extract_looms.is_datetime(subdir):
                if os.path.isfile(video_path):
                    extract_looms.auto_extract_all(directory)

                extract_looms.add_ref_to_all_loom_videos(directory)

if __name__ == '__main__': main()
