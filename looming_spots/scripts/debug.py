def main():
    import os
    from looming_spots.ref_builder import viewer
    import skvideo.io

    directory = '/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/171030/CA84_2/20171030_11_28_19/'
    os.chdir(directory)
    #vid = skvideo.io.vread(os.path.join(directory, 'loom0.h264'), num_frames=400)
    viewer.Viewer(directory, video=None, video_fname='loom0.h264')

if __name__ == '__main__': main()