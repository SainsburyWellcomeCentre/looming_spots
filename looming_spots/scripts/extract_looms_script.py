def main():
    import os
    from looming_spots.preprocess import extract_looms

    directory = '/home/slenzi/spine_shares/loomer/data_working_copy/CA105_5/20171204_21_44_05'
    os.chdir(directory)
    # ref = viewer.Ref()
    extract_looms.auto_extract_all(directory, overwrite=True)


if __name__ == '__main__': main()


# CA51_3
# CA51_4
# CA51_5
# CA52_1
# CA52_2
# CA52_3
# CA52_4
# CA52_5
# CA50_3
