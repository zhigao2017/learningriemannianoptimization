from __future__ import absolute_import
import os
import errno

# import scipy.io as sio
import os.path as osp
import shutil


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

train_txt = 'Stanford_Online_Products/Ebay_train.txt'
test_txt = 'Stanford_Online_Products/Ebay_test.txt'

train_root = 'Products/train'
test_root = 'Products/test'

start_root = 'Products'

mkdir_if_missing(start_root)

# train
f = open(train_txt)
annotations = f.readlines()

for annotation in annotations[1:]:
    annotation = annotation.split(' ')
    label = int(annotation[1])
    origin_img = osp.join('Stanford_Online_Products', annotation[-1][:-1])
    to_path = osp.join(train_root, '%d' % label)
    mkdir_if_missing(to_path)
    shutil.copy(origin_img, to_path)


# test
f = open(test_txt)
annotations = f.readlines()

for annotation in annotations[1:]:
    annotation = annotation.split(' ')
    label = int(annotation[1])
    origin_img = osp.join('Stanford_Online_Products', annotation[-1][:-1])
    to_path = osp.join(test_root, '%d' % label)
    mkdir_if_missing(to_path)
    shutil.copy(origin_img, to_path)
