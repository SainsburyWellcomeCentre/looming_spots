def main():
    from configobj import ConfigObj
    from looming_spots import mdata

    directory = '/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/171030/CA84_2/20171030_11_28_19/metadata.cfg'
    conf = ConfigObj(directory)
    a = mdata.load_nest_level_metadata(conf, 'second')
    print(a)

if __name__ == '__main__': main()
