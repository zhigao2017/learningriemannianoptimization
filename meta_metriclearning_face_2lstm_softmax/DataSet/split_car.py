from __future__ import absolute_import
import os
import errno

import scipy.io as sio
import os.path as osp
import shutil


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

path = 'Car196/cars_annos.mat'
file = sio.loadmat(path)
annotations = file['annotations'][0]


# print(annotations)

to_root = ['Car196/train', 'Car196/test']
start_root = 'Car196'
for i in range(2):
    mkdir_if_missing(to_root[i])

labels = []
for annotation in annotations:
    # print(annotation)
    label = annotation[-2][0][0]
    frame = annotation[0][0]
    # print(40*'#', label, frame)
    # print(label)

    origin_img = osp.join(start_root, frame)
    # print(origin_img)
    if label < 99:
        to_path = osp.join(to_root[0], '%d' % label)
    else:
        to_path = osp.join(to_root[1], '%d' % label)
    # print(to_path)
    mkdir_if_missing(to_path)
    shutil.copy(origin_img, to_path)
