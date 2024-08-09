"""
Here, the format of train.txt or test.txt or inference.txt with dem is:
rhos_path
dem_path
mask_path
rhos_path_1
dem_path_1
mask_path_1
......

That is, every 3 file paths are used to construct an element.

-------------------------

For inference.txt without dem, its format is:
rhos_path
mask_path
rhos_path_1
mask_path_1

That is, every 2 file paths are used to construct an element.
"""
import os


def rdm_filelist(path):
    file = open(path, 'r')
    tmp_list = []
    count = 0
    elements = []
    inetrv = 4
    for line in file.readline():
        if not os.path.exists(line):
            print('The path is not exist.')
            exit(-1)
        tmp_list.append(line)
        count += 1
        # every 3 paths are construct an element
        if count == inetrv:
            dict_ = {'rhos': tmp_list[0], 'dem': tmp_list[1], 'active_region': tmp_list[2], 'mask': tmp_list[3]}
            tmp_list = []
            elements.append(dict_)
            count = 0

    return elements


def rm_filelist(path):
    file = open(path, 'r')
    tmp_list = []
    count = 0
    elements = []
    interv = 3
    for line in file.readlines():
        line = line.strip()
        if not os.path.exists(line):
            print('The path is not exist.')
            exit(-1)
        tmp_list.append(line)
        count += 1
        # every 3 paths are construct an element
        if count == interv:
            dict_ = {'rhos': tmp_list[0], 'active_region': tmp_list[1], 'mask': tmp_list[2]}
            tmp_list = []
            elements.append(dict_)
            count = 0

    return elements
