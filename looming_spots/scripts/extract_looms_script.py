def main():
    import os
    from looming_spots.analysis import extract_looms

    directory = '/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/171030/CA84_2/20171030_11_28_19/'
    os.chdir(directory)
    # ref = viewer.Ref()
    extract_looms.auto_extract_all(directory, overwrite=True)

if __name__ == '__main__': main()